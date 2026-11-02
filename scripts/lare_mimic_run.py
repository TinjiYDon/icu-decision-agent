"""LARE · MIMIC 运行器（两套死亡标签 × 四种复核策略 × N seeds）。

两种模式：

1. `--smoke`（默认自检、**不连 DB**）：用 `np.random.default_rng(0)` 合成
   `p ~ Beta(0.4, 6.0)`、`y_death ~ Bernoulli(0.15p+0.02)`、
   `y_dod = build_coarse_label(y_death, p, add_rate=0.5)`，n=2000，
   走与真实模式**完全相同**的管线（`run_lare_pipeline`），写
   `artifacts/lare/smoke_lare.json`。
2. 真实模式（需要 DUA/CITI 确认 + 环境变量 `LARE_MIMIC_ALLOW=1`）：
   读 `paper_feat.sample_matrix` join `paper_label.mortality_12h`（精确口径）与
   `paper_label.mortality_12h_dod`（日期级粗化口径），用 LightGBM 现成模型算 `p`，
   再跑同一管线，写 `artifacts/lare/mimic_lare.json`。

管线（`run_lare_pipeline`，纯函数、可单测）在每个 seed 上做一次 **bootstrap 重采样**，
在重采样上拟合/评估四种策略，并在两套标签下分别读指标：

    fixed      固定双阈值 (0.1, 0.3)（label-free 基线）
    double     select_double_threshold 在精确标签上拟合 + 两套标签交叉评估
    sbr        selective_review.solve_sbr 的终端策略表（横断面近似，每病例 ≤1 次复核）
    deferral   cost-aware deferral：按预算 K 把 top-K 风险直接转人工（单阈值近似）

另外每个 seed 跑一次 `fit_lar_policy(bound="none")` 与 `bound="hoeffding"`
（记录 tau / robust_cost / max_regret / feasible），并记录“标签专属阈值漂移”与
复核集合 Jaccard。

用法：

    python scripts/lare_mimic_run.py --smoke --seeds 2
    LARE_MIMIC_ALLOW=1 python scripts/lare_mimic_run.py --seeds 5 --hours 0,1,2,4,6

退出码：0 成功 / 2 输入或数据错误（含模型缺失，原因写入 `*.error.json`）/
3 F1 门未放行（`.env` 缺 `LARE_MIMIC_ALLOW=1`）。

约束：只依赖 numpy + 标准库（真实模式另用仓内既有 lightgbm/sqlalchemy）；
`--smoke` 不读 DB；脚本自身不跑 MIMIC、不访问网络、不 commit/push。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from domain.models.lar_policy import fit_lar_policy, label_specific_policies, max_regret
from domain.models.lare import (
    build_coarse_label,
    double_policy,
    evaluate_policy,
    reviewed_jaccard,
)
from domain.models.pac_safety_net import select_double_threshold
from domain.models.selective_review import (
    ALERT,
    RELEASE,
    REVIEW,
    CostModel,
    Reviewer,
    default_transition,
    solve_sbr,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "lare"

DEFAULT_DSN = "postgresql://icu_dev:icu_dev@127.0.0.1:5432/mimic"
DEFAULT_MODEL = "artifacts/models/lgbm_mortality_12h.txt"
DEFAULT_OUT = ARTIFACT_DIR / "mimic_lare.json"
SMOKE_OUT = ARTIFACT_DIR / "smoke_lare.json"

DEFAULT_SEEDS = 5
DEFAULT_HOURS = "0,1,2,4,6"
DEFAULT_K = 0.2
DEFAULT_ALPHA = 0.05
DEFAULT_DELTA = 0.1
N_GRID = 50

FIXED_TAU_LOW = 0.1
FIXED_TAU_HIGH = 0.3

STRATEGIES = ("fixed", "double", "sbr", "deferral")
LABEL_FREE = frozenset({"fixed", "sbr", "deferral"})
BOUNDS = ("none", "hoeffding")

SBR_GRID = 21
SBR_BUDGET = 1  # 横断面近似：每个病例最多 1 次复核

GATE_ENV = "LARE_MIMIC_ALLOW"
GATE_MSG = "F1-GATE: 需要 DUA/CITI 已确认"

STRATEGY_METRICS = (
    "review_yield_dod",
    "review_yield_death",
    "missed_imminent",
    "release_risk",
    "release_risk_dod",
    "release_risk_death",
    "reviewed_jaccard",
    "budget",
)


# ── 小工具 ────────────────────────────────────────────────────────────────


def _json_number(x) -> float | None:
    """inf / nan / None → None（严格 JSON）。"""
    if x is None:
        return None
    value = float(x)
    return value if math.isfinite(value) else None


def _mean(values) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return float(np.mean(finite))


def _as_vector(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} 不能为空")
    return arr


def _parse_seeds(seeds) -> list[int]:
    """接受 int / 可迭代 int；负数、空列表、非整数 → ValueError。"""
    if isinstance(seeds, (int, np.integer)) and not isinstance(seeds, bool):
        seeds = [int(seeds)]
    try:
        out = [int(s) for s in seeds]
    except TypeError as exc:  # 不可迭代
        raise ValueError(f"seeds 必须是 int 序列，得到 {seeds!r}") from exc
    if not out:
        raise ValueError("seeds 不能为空")
    if any(s < 0 for s in out):
        raise ValueError("seed 必须 >= 0（np.random.default_rng 的限制）")
    return out


def _validate(p, y_dod, y_death, *, K, alpha, n_grid, delta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = _as_vector(p, "p")
    if not np.all(np.isfinite(scores)):
        raise ValueError("p 含 NaN/inf")
    scores = np.clip(scores, 0.0, 1.0)
    dod = _as_vector(y_dod, "y_dod")
    death = _as_vector(y_death, "y_death")
    for name, arr in (("y_dod", dod), ("y_death", death)):
        if arr.size != scores.size:
            raise ValueError(f"{name} 与 p 长度不一致: {arr.size} vs {scores.size}")
        if not np.all(np.isin(arr, (0.0, 1.0))):
            raise ValueError(f"{name} 只接受 0/1 标签")
    if not 0.0 <= float(K) <= 1.0:
        raise ValueError("K 必须落在 [0, 1]")
    if not float(alpha) > 0.0:
        raise ValueError("alpha 必须 > 0")
    if int(n_grid) < 2:
        raise ValueError("n_grid 至少为 2")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta 必须落在 (0, 1)")
    return scores, dod, death


# ── 合成数据（smoke / 单测） ─────────────────────────────────────────────


def make_smoke_dataset(n: int = 2000, seed: int = 0, add_rate: float = 0.5):
    """`--smoke` 用的合成数据：与 `scripts/lare_demo.py` 同一生成机制。"""
    if int(n) < 10:
        raise ValueError("n 至少为 10")
    rng = np.random.default_rng(int(seed))
    p = rng.beta(0.4, 6.0, size=int(n))
    y_death = (rng.random(int(n)) < 0.15 * p + 0.02).astype(float)
    y_dod = build_coarse_label(
        y_death, p, add_rate=float(add_rate), rng=np.random.default_rng(int(seed))
    )
    return p, y_dod, y_death


# ── 四种策略的三区掩码 ───────────────────────────────────────────────────


def _sbr_masks(p: np.ndarray) -> dict:
    """SBR 的横断面近似：解一次 DP，取**终端时刻**、`k=SBR_BUDGET` 的动作表按最近邻作用到 p。

    真实 SBR 需要病例轨迹（`simulate_case`）；这里只有 (p, y) 快照，因此用
    “horizon 内最后一个决策点 + 每病例最多 1 次复核”作为可比较的近似，文档里显式标注。
    """
    grid = np.linspace(0.0, 1.0, SBR_GRID)
    cost = CostModel()
    reviewer = Reviewer()
    policy = solve_sbr(
        grid, default_transition(grid), cost, reviewer, horizon=1, budget=SBR_BUDGET
    )
    idx = np.argmin(np.abs(grid[None, :] - p[:, None]), axis=1)
    action = policy.action[-1][idx, SBR_BUDGET]
    return {
        "release": action == RELEASE,
        "review": action == REVIEW,
        "alert": action == ALERT,
    }


def _deferral_masks(p: np.ndarray, K: float) -> dict:
    """Cost-aware deferral 的固定阈值近似：把 top-`K` 风险整体转人工（复核），其余放行。"""
    if float(K) >= 1.0:
        tau_low = 0.0
    else:
        tau_low = float(np.quantile(p, 1.0 - float(K)))
    tau_low = min(max(tau_low, 0.0), 1.0)
    return double_policy(p, tau_low, 1.0)


def _deferral_tau(p: np.ndarray, K: float) -> float:
    """deferral 的转人工阈值（与 `_deferral_masks` 同口径，便于落盘）。"""
    if float(K) >= 1.0:
        return 0.0
    return min(max(float(np.quantile(p, 1.0 - float(K))), 0.0), 1.0)


def strategy_masks(name: str, p: np.ndarray, y_fit: np.ndarray, *,
                   K: float, alpha: float, n_grid: int, delta: float) -> tuple[dict, dict]:
    """构造某策略的三区掩码；返回 `(masks, meta)`。

    `y_fit` 只被**依赖标签**的策略（`double`）使用；`fixed / sbr / deferral` 与标签无关，
    因此两套标签下掩码完全相同（Jaccard 恒为 1.0，代表“无口径漂移”）。
    """
    if name == "fixed":
        masks = double_policy(p, FIXED_TAU_LOW, FIXED_TAU_HIGH)
        meta = {"tau_low": FIXED_TAU_LOW, "tau_high": FIXED_TAU_HIGH}
    elif name == "double":
        res = select_double_threshold(
            p, y_fit, alpha=float(alpha), delta=float(delta),
            n_low=int(n_grid), n_high=int(n_grid),
        )
        masks = double_policy(p, res.tau_low, res.tau_high)
        meta = {
            "tau_low": float(res.tau_low),
            "tau_high": float(res.tau_high),
            "feasible": bool(res.feasible),
            "risk_upper_bound": _json_number(res.risk_upper_bound),
        }
    elif name == "sbr":
        masks = _sbr_masks(p)
        meta = {"review_slots": SBR_BUDGET}
    elif name == "deferral":
        masks = _deferral_masks(p, K)
        meta = {"tau_low": _deferral_tau(p, K), "tau_high": 1.0}
    else:
        raise ValueError(f"未知策略 {name!r}；可用: {list(STRATEGIES)}")
    return masks, meta


def _strategy_row(name: str, p, dod, death, *, K, alpha, n_grid, delta) -> dict:
    """一个策略在一个 seed 上的指标（两套标签 + 口径漂移 Jaccard）。"""
    masks_fit, meta = strategy_masks(
        name, p, death, K=K, alpha=alpha, n_grid=n_grid, delta=delta
    )
    if name in LABEL_FREE:
        masks_alt = masks_fit
    else:
        masks_alt, _ = strategy_masks(
            name, p, dod, K=K, alpha=alpha, n_grid=n_grid, delta=delta
        )
    ev = evaluate_policy(p, dod, death, masks_fit, K=K, alpha=alpha)
    risk_dod = float(ev["release_risk_dod"])
    risk_death = float(ev["release_risk_death"])
    row = {
        "review_yield_dod": float(ev["yield_dod"]),
        "review_yield_death": float(ev["yield_death"]),
        "missed_imminent": float(ev["missed_imminent"]),
        "release_risk": max(risk_dod, risk_death),
        "release_risk_dod": risk_dod,
        "release_risk_death": risk_death,
        "reviewed_jaccard": float(
            reviewed_jaccard(masks_fit["review"], masks_alt["review"])
        ),
        "budget": float(ev["budget"]),
        "n_release": int(ev["n_release"]),
        "n_review": int(ev["n_review"]),
        "n_alert": int(ev["n_alert"]),
        "compliance": bool(ev["compliance"]),
        "label_free": name in LABEL_FREE,
    }
    row.update(meta)
    return row


# ── 核心管线（纯函数） ───────────────────────────────────────────────────


def run_lare_pipeline(
    p,
    y_dod,
    y_death,
    seeds,
    *,
    K: float = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
    n_grid: int = N_GRID,
    delta: float = DEFAULT_DELTA,
    bootstrap: bool = True,
) -> dict:
    """在两套标签上跑 `seeds` 次 bootstrap 的四种策略 + 两个 LAR 口径，返回汇总 dict。

    参数
    ----
    p, y_dod, y_death : 等长向量；`y_*` 只接受 0/1；`p` 截断到 [0,1]
    seeds : int 序列（每个 seed 一次 bootstrap 重采样；`bootstrap=False` 时不重采样）
    K / alpha / n_grid / delta : 与 `fit_lar_policy` / `evaluate_policy` 同义

    返回
    ----
    `{"config": {...}, "seeds": [{...per seed...}], "summary": {...五种子均值...}}`；
    summary 的数字均为有限值均值（`inf`/`nan` → 跳过），`feasible_rate` 为可行比例。
    """
    seed_list = _parse_seeds(seeds)
    scores, dod, death = _validate(
        p, y_dod, y_death, K=K, alpha=alpha, n_grid=n_grid, delta=delta
    )
    n = int(scores.size)

    rows: list[dict] = []
    for seed in seed_list:
        rng = np.random.default_rng(int(seed))
        if bootstrap:
            idx = rng.integers(0, n, size=n)
            p_s, dod_s, death_s = scores[idx], dod[idx], death[idx]
        else:
            p_s, dod_s, death_s = scores, dod, death

        strategies = {
            name: _strategy_row(
                name, p_s, dod_s, death_s,
                K=K, alpha=alpha, n_grid=n_grid, delta=delta,
            )
            for name in STRATEGIES
        }

        lar: dict[str, dict] = {}
        for bound in BOUNDS:
            policy = fit_lar_policy(
                p_s, dod_s, death_s, K=K, alpha=alpha, n_grid=n_grid,
                bound=bound, delta=delta,
            )
            regret = max_regret(
                p_s, dod_s, death_s, policy, K=K, alpha=alpha, n_grid=n_grid
            )
            lar[bound] = {
                "tau_low": float(policy.tau_low),
                "tau_high": float(policy.tau_high),
                "feasible": bool(policy.feasible),
                "robust_cost": _json_number(policy.robust_cost),
                "cost_dod": _json_number(policy.cost_dod),
                "cost_death": _json_number(policy.cost_death),
                "risk_ub_dod": _json_number(policy.risk_ub_dod),
                "risk_ub_death": _json_number(policy.risk_ub_death),
                "n_release": int(policy.n_release),
                "max_regret": _json_number(regret),
            }

        drift = label_specific_policies(
            p_s, dod_s, death_s, K=K, alpha=alpha, n_grid=n_grid
        )
        dod_masks = double_policy(p_s, drift["dod"]["tau_low"], drift["dod"]["tau_high"])
        death_masks = double_policy(
            p_s, drift["death"]["tau_low"], drift["death"]["tau_high"]
        )

        rows.append(
            {
                "seed": int(seed),
                "n": n,
                "base_rate_dod": float(np.mean(dod_s)),
                "base_rate_death": float(np.mean(death_s)),
                "n_positive_dod": int(np.count_nonzero(dod_s)),
                "n_positive_death": int(np.count_nonzero(death_s)),
                "strategies": strategies,
                "lar": lar,
                "label_drift": {
                    "tau_low_dod": float(drift["dod"]["tau_low"]),
                    "tau_high_dod": float(drift["dod"]["tau_high"]),
                    "tau_low_death": float(drift["death"]["tau_low"]),
                    "tau_high_death": float(drift["death"]["tau_high"]),
                    "delta_tau_low": float(
                        drift["dod"]["tau_low"] - drift["death"]["tau_low"]
                    ),
                    "delta_tau_high": float(
                        drift["dod"]["tau_high"] - drift["death"]["tau_high"]
                    ),
                    "reviewed_jaccard": float(
                        reviewed_jaccard(dod_masks["review"], death_masks["review"])
                    ),
                    "feasible_dod": bool(drift["dod"]["feasible"]),
                    "feasible_death": bool(drift["death"]["feasible"]),
                },
            }
        )

    summary = {
        "n_seeds": len(rows),
        "bootstrap": bool(bootstrap),
        "strategies": {
            name: {
                **{key: _mean(r["strategies"][name][key] for r in rows)
                   for key in STRATEGY_METRICS},
                "n_release": _mean(r["strategies"][name]["n_release"] for r in rows),
                "n_review": _mean(r["strategies"][name]["n_review"] for r in rows),
                "n_alert": _mean(r["strategies"][name]["n_alert"] for r in rows),
                "compliance_rate": _mean(
                    1.0 if r["strategies"][name]["compliance"] else 0.0 for r in rows
                ),
            }
            for name in STRATEGIES
        },
        "lar": {
            bound: {
                "tau_low": _mean(r["lar"][bound]["tau_low"] for r in rows),
                "tau_high": _mean(r["lar"][bound]["tau_high"] for r in rows),
                "robust_cost": _mean(r["lar"][bound]["robust_cost"] for r in rows),
                "cost_dod": _mean(r["lar"][bound]["cost_dod"] for r in rows),
                "cost_death": _mean(r["lar"][bound]["cost_death"] for r in rows),
                "risk_ub_dod": _mean(r["lar"][bound]["risk_ub_dod"] for r in rows),
                "risk_ub_death": _mean(r["lar"][bound]["risk_ub_death"] for r in rows),
                "n_release": _mean(r["lar"][bound]["n_release"] for r in rows),
                "max_regret": _mean(r["lar"][bound]["max_regret"] for r in rows),
                "feasible_rate": _mean(
                    1.0 if r["lar"][bound]["feasible"] else 0.0 for r in rows
                ),
            }
            for bound in BOUNDS
        },
        "label_drift": {
            "tau_low_dod": _mean(r["label_drift"]["tau_low_dod"] for r in rows),
            "tau_high_dod": _mean(r["label_drift"]["tau_high_dod"] for r in rows),
            "tau_low_death": _mean(r["label_drift"]["tau_low_death"] for r in rows),
            "tau_high_death": _mean(r["label_drift"]["tau_high_death"] for r in rows),
            "delta_tau_low": _mean(r["label_drift"]["delta_tau_low"] for r in rows),
            "delta_tau_high": _mean(r["label_drift"]["delta_tau_high"] for r in rows),
            "reviewed_jaccard": _mean(r["label_drift"]["reviewed_jaccard"] for r in rows),
        },
        "base_rate_dod": _mean(r["base_rate_dod"] for r in rows),
        "base_rate_death": _mean(r["base_rate_death"] for r in rows),
    }

    return {
        "config": {
            "n": n,
            "seeds": seed_list,
            "bootstrap": bool(bootstrap),
            "K": float(K),
            "alpha": float(alpha),
            "n_grid": int(n_grid),
            "delta": float(delta),
            "strategies": list(STRATEGIES),
            "bounds": list(BOUNDS),
            "fixed_policy": {"tau_low": FIXED_TAU_LOW, "tau_high": FIXED_TAU_HIGH},
            "sbr_grid": SBR_GRID,
            "sbr_budget": SBR_BUDGET,
            "coarse_superset_violations": int(
                np.count_nonzero((death == 1.0) & (dod != 1.0))
            ),
        },
        "seeds": rows,
        "summary": summary,
    }


# ── 真实模式：读 DB + 模型打分（本轮不跑） ───────────────────────────────


def _load_real_frame(dsn: str, *, feat_table: str, death_table: str, dod_table: str,
                     hours: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读 `paper_feat.sample_matrix` + 两套标签 + LightGBM 打分，返回 `(p, y_dod, y_death)`。

    表名默认值对应五时刻全量矩阵（`paper_feat.sample_matrix` / `paper_label.mortality_12h` /
    `paper_label.mortality_12h_dod`）；`feat.sample_matrix` 仅 h=0。缺表时报错（CLI 可覆盖）。
    """
    import re

    from sqlalchemy import bindparam, create_engine, text

    from domain.features.build import FEATURE_COLS

    for name in (feat_table, death_table, dod_table):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?", name):
            raise ValueError(f"非法表名: {name!r}（只允许 schema.table 形式）")

    sql = text(
        f"""
        SELECT f.stay_id, f.hour_index, f.feature_json,
               d.label AS y_death, o.label AS y_dod
        FROM {feat_table} f
        JOIN {death_table} d
          ON f.stay_id = d.stay_id AND f.hour_index = d.hour_index
        JOIN {dod_table} o
          ON f.stay_id = o.stay_id AND f.hour_index = o.hour_index
        WHERE f.hour_index IN :hours
        """
    ).bindparams(bindparam("hours", expanding=True))
    engine = create_engine(dsn, pool_pre_ping=True)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"hours": list(hours)}).mappings().all()
    if not rows:
        raise RuntimeError(
            f"查询无数据：{feat_table} × {death_table} × {dod_table}（hours={hours}）"
        )

    import json as _json

    matrix = np.full((len(rows), len(FEATURE_COLS)), np.nan, dtype=float)
    y_death = np.empty(len(rows), dtype=float)
    y_dod = np.empty(len(rows), dtype=float)
    for i, row in enumerate(rows):
        raw = row["feature_json"]
        if not isinstance(raw, dict):
            raw = _json.loads(raw)
        for j, col in enumerate(FEATURE_COLS):
            value = raw.get(col)
            if value is not None:
                try:
                    matrix[i, j] = float(value)
                except (TypeError, ValueError):
                    pass
        y_death[i] = float(row["y_death"])
        y_dod[i] = float(row["y_dod"])
    return _predict_scores(matrix), y_dod, y_death


def _predict_scores(matrix: np.ndarray, model_path: str | Path = DEFAULT_MODEL) -> np.ndarray:
    """用现成 LightGBM 模型给特征矩阵打分；模型缺失 → `FileNotFoundError`。"""
    import os as _os

    import lightgbm as lgb

    path = Path(model_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(str(path))
    old_cwd = _os.getcwd()
    try:
        _os.chdir(path.parent)  # LightGBM 对含空格/中文的绝对路径不友好
        booster = lgb.Booster(model_file=path.name)
    finally:
        _os.chdir(old_cwd)
    return np.clip(np.asarray(booster.predict(matrix), dtype=float).reshape(-1), 0.0, 1.0)


# ── CLI ─────────────────────────────────────────────────────────────────


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="LARE MIMIC runner（--smoke 不连 DB；真实模式需 F1 门放行）"
    )
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                    help="bootstrap 重采样次数（seed = 0..N-1）")
    ap.add_argument("--hours", default=DEFAULT_HOURS, help="预测时刻，逗号分隔")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true", help="合成数据自检（不连 DB）")
    ap.add_argument("--K", dest="K", type=float, default=DEFAULT_K,
                    help="复核+升级预算上限（默认 0.2）")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    ap.add_argument("--feat-table", default="paper_feat.sample_matrix")
    ap.add_argument("--death-table", default="paper_label.mortality_12h")
    ap.add_argument("--dod-table", default="paper_label.mortality_12h_dod")
    args = ap.parse_args(argv)

    if args.seeds < 1:
        print("--seeds 至少为 1", file=sys.stderr)
        return 2
    try:
        hours = [int(h) for h in args.hours.split(",") if h.strip()]
    except ValueError:
        print(f"--hours 需要逗号分隔的整数，得到 {args.hours!r}", file=sys.stderr)
        return 2
    if not hours:
        print("--hours 不能为空", file=sys.stderr)
        return 2
    seeds = list(range(args.seeds))

    if args.smoke:
        out_path = Path(args.out) if args.out else SMOKE_OUT
        try:
            p, y_dod, y_death = make_smoke_dataset(2000, 0, 0.5)
            payload = run_lare_pipeline(
                p, y_dod, y_death, seeds,
                K=args.K, alpha=args.alpha, delta=args.delta,
            )
        except ValueError as exc:
            print(f"smoke 失败: {exc}", file=sys.stderr)
            return 2
        payload["config"]["mode"] = "smoke"
        payload["config"]["hours"] = hours
        payload["note"] = "synthetic only (p~Beta(0.4,6.0)); no MIMIC data, no clinical claim"
        _write_json(out_path, payload)
        _print_summary(payload)
        print(f"\nwrote {out_path}")
        return 0

    # ── 真实模式：F1 门 + 数据 + 模型 ──
    print(GATE_MSG)
    if os.environ.get(GATE_ENV, "") not in ("1", "true", "TRUE", "yes"):
        print(
            f"未放行：请先确认 DUA/CITI，并设置环境变量 {GATE_ENV}=1 后重跑。",
            file=sys.stderr,
        )
        return 3

    out_path = Path(args.out) if args.out else DEFAULT_OUT
    try:
        p, y_dod, y_death = _load_real_frame(
            args.dsn,
            feat_table=args.feat_table,
            death_table=args.death_table,
            dod_table=args.dod_table,
            hours=hours,
        )
        payload = run_lare_pipeline(
            p, y_dod, y_death, seeds, K=args.K, alpha=args.alpha, delta=args.delta
        )
    except FileNotFoundError as exc:
        reason = f"模型文件缺失: {exc}"
        _write_json(out_path.with_name(out_path.stem + ".error.json"),
                    {"status": "error", "reason": reason})
        print(reason, file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — CLI 边界：DB 连不上 / 表缺失 / 查询为空
        reason = f"{type(exc).__name__}: {exc}"
        _write_json(out_path.with_name(out_path.stem + ".error.json"),
                    {"status": "error", "reason": reason})
        print(reason, file=sys.stderr)
        return 2

    payload["config"].update({
        "mode": "mimic",
        "dsn": args.dsn,
        "hours": hours,
        "model": args.model,
        "feat_table": args.feat_table,
        "death_table": args.death_table,
        "dod_table": args.dod_table,
    })
    _write_json(out_path, payload)
    _print_summary(payload)
    print(f"\nwrote {out_path}")
    return 0


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    value = float(x)
    return "inf" if not math.isfinite(value) else f"{value:.4f}"


def _print_summary(payload: dict) -> None:
    summary = payload["summary"]
    print("\n## per-strategy summary (seed 均值)")
    print("| strategy | yield_dod | yield_death | missed | release_risk | jaccard | budget |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name in STRATEGIES:
        s = summary["strategies"][name]
        print(
            f"| {name} | {_fmt(s['review_yield_dod'])} | {_fmt(s['review_yield_death'])} | "
            f"{_fmt(s['missed_imminent'])} | {_fmt(s['release_risk'])} | "
            f"{_fmt(s['reviewed_jaccard'])} | {_fmt(s['budget'])} |"
        )
    print("\n## LAR-Policy")
    print("| bound | tau_low | tau_high | feasible_rate | robust_cost | max_regret |")
    print("|---|---:|---:|---:|---:|---:|")
    for bound in BOUNDS:
        s = summary["lar"][bound]
        print(
            f"| {bound} | {_fmt(s['tau_low'])} | {_fmt(s['tau_high'])} | "
            f"{_fmt(s['feasible_rate'])} | {_fmt(s['robust_cost'])} | "
            f"{_fmt(s['max_regret'])} |"
        )
    drift = summary["label_drift"]
    print(
        f"\nlabel drift: Δtau_low={_fmt(drift['delta_tau_low'])} "
        f"Δtau_high={_fmt(drift['delta_tau_high'])} "
        f"jaccard={_fmt(drift['reviewed_jaccard'])}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
