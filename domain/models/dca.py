"""Decision Curve Analysis (DCA) for ICU mortality prediction.

高内聚低耦合设计：
- 净受益计算（含成本比）
- Bootstrap 置信区间
- 工作点寻优（F1 / 临床成本比）
- 全量 DCA 曲线导出
- 报告生成（JSON + Markdown）

用法：
    from domain.models.dca import run_dca, compute_net_benefit_curve
    result = run_dca(y_true, y_prob, cost_ratio=4.0, n_bootstrap=1000)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from domain.models.evaluation import net_benefit_at_threshold

# ─────────────────────────────────────────────────────────────────────────────
# 数据类（不可变，单一职责）
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "dca"


@dataclass(frozen=True)
class PointResult:
    """单阈值净受益计算结果。"""
    threshold: float
    net_benefit_model: float
    net_benefit_treat_all: float
    net_benefit_treat_none: float
    prevalence: float
    tp: int
    fp: int
    fn: int
    tn: int
    cost_ratio: float          # FP:FN 代价比


@dataclass(frozen=True)
class WorkingPoint:
    """工作点（最优阈值）汇总。"""
    method: str                # "f1" | "clinical_cost"
    threshold: float
    precision: float
    recall: float
    specificity: float
    f1: float
    net_benefit: float
    tp: int
    fp: int
    fn: int
    tn: int
    cost_ratio: float
    note: str = ""


@dataclass(frozen=True)
class CurveResult:
    """完整 DCA 曲线 + Bootstrap CI。"""
    thresholds: list[float]
    nb_model: list[float]
    nb_treat_all: list[float]
    nb_treat_none: list[float]
    nb_model_ci_lower: list[float] | None
    nb_model_ci_upper: list[float] | None
    n: int
    prevalence: float
    working_point: WorkingPoint | None
    cost_ratio: float
    n_bootstrap: int
    elapsed_s: float
    status: str = "ok"
    message: str = ""


@dataclass(frozen=True)
class DCAResult:
    """DCA 完整结果（供 export_json / export_md 使用）。"""
    model_name: str
    curve: CurveResult
    points: list[PointResult]
    report_path: str
    elapsed_s: float


# ─────────────────────────────────────────────────────────────────────────────
# 核心计算函数
# ─────────────────────────────────────────────────────────────────────────────

def compute_net_benefit(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    cost_ratio: float = 1.0,
) -> PointResult:
    """在指定阈值和代价比下计算单点净受益。

    Args:
        y_true: 真实标签 (0/1)
        probability: 预测概率
        threshold: 决策阈值 t
        cost_ratio: FP:FN 代价比（默认 1.0 = 等代价；ICU 场景建议 3~5）

    Returns:
        PointResult：含 tp/fp/fn/tn 和三类净受益
    """
    y = y_true.astype(int)
    prob = probability.astype(float)
    t = float(threshold)
    if not (0.0 < t < 1.0):
        raise ValueError("threshold must be in (0, 1)")
    n = len(y)
    pred = (prob >= t).astype(int)

    cm = _confusion_matrix(y, pred)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    prevalence = float(y.mean()) if n else 0.0
    odds = t / max(1.0 - t, 1e-9)

    # 广义净受益：nb = (TP/n) - (FP/n) * cost_ratio * odds
    # cost_ratio=1 → 标准 DCA；cost_ratio>1 → 更保守（提高阈值）
    nb_model = tp / n - fp / n * cost_ratio * odds
    nb_all = prevalence - (1.0 - prevalence) * cost_ratio * odds

    return PointResult(
        threshold=t,
        net_benefit_model=nb_model,
        net_benefit_treat_all=nb_all,
        net_benefit_treat_none=0.0,
        prevalence=prevalence,
        tp=tp, fp=fp, fn=fn, tn=tn,
        cost_ratio=cost_ratio,
    )


def compute_net_benefit_curve(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    thresholds: Sequence[float] | None = None,
    cost_ratio: float = 1.0,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> CurveResult:
    """计算 DCA 曲线 + Bootstrap 置信区间。

    Args:
        y_true: 真实标签
        probability: 预测概率
        thresholds: 阈值网格（默认 0.01~0.50，50 点）
        cost_ratio: FP:FN 代价比
        n_bootstrap: Bootstrap 抽样次数
        seed: 随机种子
        ci_level: 置信水平（默认 0.95）

    Returns:
        CurveResult：含均值曲线和 CI 带
    """
    y = y_true.astype(int)
    prob = probability.astype(float)
    rng = np.random.default_rng(seed)
    n = len(y)

    if thresholds is None:
        # 针对低阳性率场景，聚焦低阈值区间
        thr = np.linspace(0.01, 0.50, 50)
    else:
        thr = np.asarray(list(thresholds), dtype=float)

    t0 = time.time()

    # 基础曲线
    nb_model_base = []
    nb_all_base = []
    for t in thr:
        pt = compute_net_benefit(y, prob, threshold=float(t), cost_ratio=cost_ratio)
        nb_model_base.append(pt.net_benefit_model)
        nb_all_base.append(pt.net_benefit_treat_all)

    # Bootstrap CI
    ci_lower: list[float] | None = None
    ci_upper: list[float] | None = None
    if n_bootstrap > 0 and n >= 30:
        nb_boot: list[list[float]] = [[] for _ in thr]
        for _b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            y_boot = y[idx]
            prob_boot = prob[idx]
            for i, t in enumerate(thr):
                pt = compute_net_benefit(y_boot, prob_boot, threshold=float(t),
                                         cost_ratio=cost_ratio)
                nb_boot[i].append(pt.net_benefit_model)
        alpha = 1.0 - ci_level
        ci_lower = [float(np.percentile(nb_boot[i], 100 * alpha / 2)) for i in range(len(thr))]
        ci_upper = [float(np.percentile(nb_boot[i], 100 * (1 - alpha / 2))) for i in range(len(thr))]

    # 工作点
    wp = find_working_point(y, prob, cost_ratio=cost_ratio)

    elapsed = time.time() - t0
    return CurveResult(
        thresholds=[float(x) for x in thr],
        nb_model=nb_model_base,
        nb_treat_all=nb_all_base,
        nb_treat_none=[0.0] * len(thr),
        nb_model_ci_lower=ci_lower,
        nb_model_ci_upper=ci_upper,
        n=n,
        prevalence=float(y.mean()) if n else 0.0,
        working_point=wp,
        cost_ratio=cost_ratio,
        n_bootstrap=n_bootstrap,
        elapsed_s=elapsed,
    )


def find_working_point(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    method: str = "f1",
    cost_ratio: float = 1.0,
) -> WorkingPoint:
    """寻优工作点阈值。

    Args:
        method: "f1"（最大 F1）或 "clinical_cost"（最大化净受益，考虑代价比）
    """
    y = y_true.astype(int)
    prob = probability.astype(float)
    n = len(y)

    if method == "clinical_cost":
        # 在阈值网格上搜索最大化净受益的阈值
        thr_grid = np.linspace(0.01, 0.50, 200)
        best_nb, best_t = -1e9, 0.01
        for t in thr_grid:
            pt = compute_net_benefit(y, prob, threshold=float(t), cost_ratio=cost_ratio)
            if pt.net_benefit_model > best_nb:
                best_nb = pt.net_benefit_model
                best_t = t
    else:
        # F1 最大
        best_t = _select_threshold_f1(y, prob)

    # 在工作点计算完整指标
    pt = compute_net_benefit(y, prob, threshold=best_t, cost_ratio=cost_ratio)
    cm = _confusion_matrix(y, (prob >= best_t).astype(int))

    prec = cm["tp"] / max(cm["tp"] + cm["fp"], 1)
    rec = cm["tp"] / max(cm["tp"] + cm["fn"], 1)
    spec = cm["tn"] / max(cm["tn"] + cm["fp"], 1)
    f1 = (2 * prec * rec / max(prec + rec, 1e-9))

    method_desc = "max_F1" if method == "f1" else "max_net_benefit(cost_ratio={})".format(cost_ratio)
    return WorkingPoint(
        method=method_desc,
        threshold=round(best_t, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        specificity=round(spec, 4),
        f1=round(f1, 4),
        net_benefit=round(pt.net_benefit_model, 4),
        tp=cm["tp"], fp=cm["fp"], fn=cm["fn"], tn=cm["tn"],
        cost_ratio=cost_ratio,
        note=f"threshold={best_t:.4f} via {method_desc}",
    )


def run_dca(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    model_name: str = "lgbm",
    cost_ratio: float = 4.0,
    n_bootstrap: int = 1000,
    seed: int = 42,
    save: bool = True,
) -> DCAResult:
    """运行完整 DCA 分析，导出 artifacts。

    Args:
        model_name: 模型名称（用于文件命名）
        cost_ratio: FP:FN 临床代价比（默认 4.0，符合 ICU 场景）
        n_bootstrap: Bootstrap 抽样次数
        seed: 随机种子
        save: 是否保存 JSON + Markdown 报告

    Returns:
        DCAResult
    """
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(probability, dtype=float)

    t0 = time.time()
    curve = compute_net_benefit_curve(y, prob, cost_ratio=cost_ratio,
                                      n_bootstrap=n_bootstrap, seed=seed)

    # 提取关键阈值点的净受益
    points = []
    key_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    for t in key_thresholds:
        pt = compute_net_benefit(y, prob, threshold=t, cost_ratio=cost_ratio)
        points.append(pt)

    ART.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = ART / f"dca_{model_name}_{ts}.json"
    md_path = ART / f"dca_{model_name}_{ts}.md"

    result = DCAResult(
        model_name=model_name,
        curve=curve,
        points=points,
        report_path=str(json_path),
        elapsed_s=time.time() - t0,
    )

    if save:
        _save_json(result, json_path)
        _save_md(result, md_path)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _select_threshold_f1(y_true: np.ndarray, probability: np.ndarray) -> float:
    """从 PR 曲线选取 F1 最大阈值。"""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, probability)
    if len(thr) == 0:
        return 0.5
    denom = prec[:-1] + rec[:-1]
    f1 = np.divide(
        2 * prec[:-1] * rec[:-1], denom,
        out=np.zeros_like(denom), where=denom > 0,
    )
    return float(thr[int(np.argmax(f1))]) if len(f1) > 0 else 0.5


def _save_json(result: DCAResult, path: Path) -> None:
    """保存 DCA 结果为 JSON。"""
    payload = {
        "model_name": result.model_name,
        "curve": asdict(result.curve),
        "points": [asdict(p) for p in result.points],
        "elapsed_s": result.elapsed_s,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_md(result: DCAResult, path: Path) -> None:
    """保存 DCA 结果为 Markdown 报告。"""
    c = result.curve
    wp = c.working_point
    lines = [
        "# DCA 决策曲线分析报告", "",
        f"**模型**: {result.model_name}",
        f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**耗时**: {result.elapsed_s:.1f}s",
        f"**Bootstrap 次数**: {c.n_bootstrap}",
        f"**代价比 FP:FN**: {c.cost_ratio}:1", "",
        "## 一、数据概况", "",
        f"- 样本量：{c.n:,}",
        f"- 阳性率：{c.prevalence:.2%}",
        f"- 阈值网格：{len(c.thresholds)} 点 ({c.thresholds[0]:.3f} ~ {c.thresholds[-1]:.3f})", "",
    ]
    if wp:
        lines += [
            "## 二、工作点", "",
            f"| 指标 | 值 |",
            f"|------|---|",
            f"| 方法 | {wp.method} |",
            f"| 阈值 | {wp.threshold:.4f} |",
            f"| 精确率 | {wp.precision:.2%} |",
            f"| 召回率 | {wp.recall:.2%} |",
            f"| 特异度 | {wp.specificity:.2%} |",
            f"| F1 | {wp.f1:.4f} |",
            f"| 净受益 | {wp.net_benefit:.4f} |",
            f"| TP / FP / FN / TN | {wp.tp} / {wp.fp} / {wp.fn} / {wp.tn} |",
        ]
    lines += ["", "## 三、关键阈值净受益", "",
              "| 阈值 | 模型 NB | 全部干预 NB | 代价比 |"]
    for p in result.points:
        lines.append(f"| {p.threshold:.2f} | {p.net_benefit_model:.4f} | {p.net_benefit_treat_all:.4f} | {p.cost_ratio:.1f} |")

    lines += ["", "## 四、备注", "",
              "- 净受益公式：NB = TP/n - FP/n * cost_ratio * odds(t)",
              f"- 当前场景：ICU 12h 死亡率，阳性率 ≈ {c.prevalence:.2%}",
              "- Bootstrap 95% CI：从重采样分布取 2.5% / 97.5% 分位数",
              "- 结论：若模型曲线在 [t_min, t_max] 区间高于 treat_all 和 treat_none，则在该阈值范围内有临床净受益。"]

    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 公开 API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "PointResult",
    "WorkingPoint",
    "CurveResult",
    "DCAResult",
    "compute_net_benefit",
    "compute_net_benefit_curve",
    "find_working_point",
    "run_dca",
    "ART",
]
