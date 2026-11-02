"""LARE + LAR-Policy demo（synthetic only）。

用法（不读数据库、不调 LLM、不跑 MIMIC）：

    python scripts/lare_demo.py --n 4000 --seed 0 --add-rate 0.5

输出：终端摘要 + `artifacts/lare/demo.json`。

打印三块：
1. 同一固定双阈值策略在两套标签（`y_dod` 粗化 / `y_death` 精确）下的
   review yield、放行区漏检质量、预算与差值；
2. 按各自标签单独选出的双阈值 —— 阈值差、复核集合 Jaccard；
3. `fit_lar_policy`（min-max 鲁棒）—— tau / robust_cost / cost_dod / cost_death，
   以及相对标签专属最优的 max regret；附 alpha=1%（严格口径）的可行性对照；
   并列出 `bound="none"`（经验风险）与 `bound="hoeffding"`（有限样本上界）的对比。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from domain.models.lar_policy import (
    LARPolicy,
    fit_lar_policy,
    label_specific_policies,
    max_regret,
)
from domain.models.lare import (
    build_coarse_label,
    double_policy,
    evaluate_policy,
    priority_spearman,
    reviewed_jaccard,
)

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "lare"

FIXED_TAU_LOW = 0.1
FIXED_TAU_HIGH = 0.3
DEFAULT_K = 0.2
DEFAULT_ALPHA = 0.05
STRICT_ALPHA = 0.01
DEFAULT_DELTA = 0.1
N_GRID = 50
BOUNDS = ("none", "hoeffding")


def make_synthetic(n: int, seed: int, add_rate: float):
    """p ~ Beta(0.4, 6.0)；y_death ~ Bernoulli(0.15p+0.02)；y_dod 为粗化超集。"""
    rng = np.random.default_rng(seed)
    p = rng.beta(0.4, 6.0, size=n)
    y_death = (rng.random(n) < 0.15 * p + 0.02).astype(float)
    y_dod = build_coarse_label(
        y_death, p, add_rate=add_rate, rng=np.random.default_rng(seed)
    )
    return p, y_dod, y_death


def _json_number(x: float):
    """inf/nan → None（严格 JSON）。"""
    return x if math.isfinite(x) else None


def _policy_json(policy: LARPolicy) -> dict:
    row = asdict(policy)
    for key in ("robust_cost", "cost_dod", "cost_death", "risk_ub_dod", "risk_ub_death"):
        row[key] = _json_number(float(row[key]))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="LARE / LAR-Policy demo (synthetic only)")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--add-rate", type=float, default=0.5)
    ap.add_argument("--bound", choices=BOUNDS, default="none",
                    help="可行性口径：none=经验风险，hoeffding=有限样本上界")
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA,
                    help="Hoeffding 上界的失败概率（仅 bound=hoeffding 生效）")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    if args.n < 10:
        raise SystemExit("--n 至少为 10")
    if not 0.0 <= args.add_rate <= 1.0:
        raise SystemExit("--add-rate 必须落在 [0, 1]")
    if not 0.0 < args.delta < 1.0:
        raise SystemExit("--delta 必须落在 (0, 1)")

    p, y_dod, y_death = make_synthetic(args.n, args.seed, args.add_rate)

    fixed_masks = double_policy(p, FIXED_TAU_LOW, FIXED_TAU_HIGH)
    fixed = evaluate_policy(
        p, y_dod, y_death, fixed_masks, K=DEFAULT_K, alpha=DEFAULT_ALPHA
    )
    fixed_spearman = priority_spearman(p, fixed_masks["review"])

    ls = label_specific_policies(
        p, y_dod, y_death, K=DEFAULT_K, alpha=DEFAULT_ALPHA, n_grid=N_GRID
    )
    dod_masks = double_policy(p, ls["dod"]["tau_low"], ls["dod"]["tau_high"])
    death_masks = double_policy(p, ls["death"]["tau_low"], ls["death"]["tau_high"])
    jaccard = reviewed_jaccard(dod_masks["review"], death_masks["review"])

    # 两种可行性口径的对照（bound=none 为经验风险，hoeffding 为有限样本上界）
    by_bound = {}
    for name in BOUNDS:
        pol = fit_lar_policy(
            p, y_dod, y_death, K=DEFAULT_K, alpha=DEFAULT_ALPHA, n_grid=N_GRID,
            bound=name, delta=args.delta,
        )
        reg = max_regret(
            p, y_dod, y_death, pol, K=DEFAULT_K, alpha=DEFAULT_ALPHA, n_grid=N_GRID
        )
        by_bound[name] = (pol, float(reg))
    policy, regret = by_bound[args.bound]

    strict = fit_lar_policy(
        p, y_dod, y_death, K=DEFAULT_K, alpha=STRICT_ALPHA, n_grid=N_GRID,
        bound=args.bound, delta=args.delta,
    )
    strict_regret = max_regret(
        p, y_dod, y_death, strict, K=DEFAULT_K, alpha=STRICT_ALPHA, n_grid=N_GRID
    )

    d_yield = fixed["yield_dod"] - fixed["yield_death"]
    d_risk = fixed["release_risk_dod"] - fixed["release_risk_death"]

    lines = [
        "# LARE · Label-Ambiguity Review Evaluation (synthetic only)",
        "",
        (
            f"config: n={args.n} seed={args.seed} add_rate={args.add_rate} "
            f"K={DEFAULT_K} alpha={DEFAULT_ALPHA} grid={N_GRID}x{N_GRID}"
        ),
        (
            f"base rate: y_death={y_death.mean():.4f} y_dod={y_dod.mean():.4f} "
            f"(dod 粗化超集, +{y_dod.mean() - y_death.mean():.4f})"
        ),
        "",
        "## 1) 同一固定策略在两套标签下",
        "",
        f"fixed policy: tau_low={FIXED_TAU_LOW} tau_high={FIXED_TAU_HIGH}",
        "",
        "| metric | y_dod (coarse) | y_death (exact) | diff |",
        "|---|---:|---:|---:|",
        (
            f"| review yield P(y=1|review) | {fixed['yield_dod']:.4f} | "
            f"{fixed['yield_death']:.4f} | {d_yield:+.4f} |"
        ),
        (
            f"| release risk E[1{{y=1}}·1{{release}}] | {fixed['release_risk_dod']:.4f} | "
            f"{fixed['release_risk_death']:.4f} | {d_risk:+.4f} |"
        ),
        f"| missed imminent (y_death) | - | {fixed['missed_imminent']:.4f} | - |",
        f"| budget (review+alert) | {fixed['budget']:.4f} | {fixed['budget']:.4f} | 0.0000 |",
        (
            f"| n_release / n_review / n_alert | {fixed['n_release']} / "
            f"{fixed['n_review']} / {fixed['n_alert']} | same | - |"
        ),
        f"| reviewed-set priority spearman | {fixed_spearman:.4f} | same | - |",
        "",
        (
            f"compliance (budget<={DEFAULT_K} & both risks<={DEFAULT_ALPHA}): "
            f"{fixed['compliance']}"
        ),
        "",
        f"## 2) 各自标签下单独选阈值（K={DEFAULT_K}, alpha={DEFAULT_ALPHA}）",
        "",
        "| label | tau_low | tau_high | cost | n_release | feasible |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in ("dod", "death"):
        row = ls[name]
        cost = row["cost"]
        cost_s = "inf" if not math.isfinite(cost) else f"{cost:.6f}"
        lines.append(
            f"| {name} | {row['tau_low']:.4f} | {row['tau_high']:.4f} | {cost_s} | "
            f"{row['n_release']} | {row['feasible']} |"
        )
    lines += [
        "",
        (
            f"reviewed-set Jaccard = {jaccard:.4f}；"
            f"Δtau_low = {ls['dod']['tau_low'] - ls['death']['tau_low']:+.4f}，"
            f"Δtau_high = {ls['dod']['tau_high'] - ls['death']['tau_high']:+.4f}"
        ),
        "",
        f"## 3) LAR-Policy（min-max 鲁棒，K={DEFAULT_K}, alpha={DEFAULT_ALPHA}）",
        "",
        "| field | value |",
        "|---|---:|",
        f"| bound (selected by --bound) | {args.bound} |",
        f"| delta | {args.delta} |",
        f"| tau_low | {policy.tau_low:.4f} |",
        f"| tau_high | {policy.tau_high:.4f} |",
        f"| feasible | {policy.feasible} |",
        (
            f"| robust_cost = max(cost_dod, cost_death) | "
            f"{_fmt(policy.robust_cost)} |"
        ),
        f"| cost_dod | {_fmt(policy.cost_dod)} |",
        f"| cost_death | {_fmt(policy.cost_death)} |",
        f"| risk_ub_dod | {_fmt(policy.risk_ub_dod)} |",
        f"| risk_ub_death | {_fmt(policy.risk_ub_death)} |",
        f"| n_release | {policy.n_release} |",
        f"| max_regret vs label-specific | {_fmt(regret)} |",
        "",
        f"### 3.1) 可行性口径对比（K={DEFAULT_K}, alpha={DEFAULT_ALPHA}, delta={args.delta}）",
        "",
        "| bound | tau_low | tau_high | feasible | robust_cost | risk_ub_dod | risk_ub_death | n_release | max_regret |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name in BOUNDS:
        pol, reg = by_bound[name]
        low_s = f"{pol.tau_low:.4f}"
        high_s = f"{pol.tau_high:.4f}"
        lines.append(
            f"| {name} | {low_s} | {high_s} | {pol.feasible} | "
            f"{_fmt(pol.robust_cost)} | {_fmt(pol.risk_ub_dod)} | "
            f"{_fmt(pol.risk_ub_death)} | {pol.n_release} | {_fmt(reg)} |"
        )
    lines += [
        "",
        (
            "note: bound=hoeffding 用 radius=sqrt(log(m/delta)/(2n))、"
            f"m={N_GRID * (N_GRID - 1) // 2}（n_grid={N_GRID} 的候选对数）；"
            "可行要求两套标签的 (经验风险+radius) 都 <= alpha，因此 \"none 可行\" 不能推出 hoeffding 可行，"
            "但 hoeffding 可行必然 none 可行。"
        ),
        "",
        (
            f"strict alpha={STRICT_ALPHA}: feasible={strict.feasible}, "
            f"robust_cost={_fmt(strict.robust_cost)}, max_regret={_fmt(strict_regret)}"
        ),
        "",
        (
            "note: synthetic only (p~Beta(0.4,6.0), y_death~Bernoulli(0.15p+0.02), "
            "y_dod=coarse superset)；no MIMIC data, no clinical claim."
        ),
    ]
    print("\n".join(lines))

    payload = {
        "config": {
            "n": args.n,
            "seed": args.seed,
            "add_rate": args.add_rate,
            "K": DEFAULT_K,
            "alpha": DEFAULT_ALPHA,
            "strict_alpha": STRICT_ALPHA,
            "n_grid": N_GRID,
            "bound": args.bound,
            "delta": args.delta,
            "fixed_policy": {"tau_low": FIXED_TAU_LOW, "tau_high": FIXED_TAU_HIGH},
            "base_rate_death": float(y_death.mean()),
            "base_rate_dod": float(y_dod.mean()),
        },
        "results": {
            "fixed_policy_two_labels": {
                **fixed,
                "priority_spearman": fixed_spearman,
                "diff_yield_dod_minus_death": d_yield,
                "diff_release_risk_dod_minus_death": d_risk,
            },
            "label_specific": {
                "dod": ls["dod"],
                "death": ls["death"],
                "reviewed_jaccard": jaccard,
                "delta_tau_low": ls["dod"]["tau_low"] - ls["death"]["tau_low"],
                "delta_tau_high": ls["dod"]["tau_high"] - ls["death"]["tau_high"],
            },
            "robust_policy": {
                **_policy_json(policy),
                "max_regret": _json_number(float(regret)),
            },
            "bounds": {
                name: {
                    **_policy_json(by_bound[name][0]),
                    "max_regret": _json_number(float(by_bound[name][1])),
                }
                for name in BOUNDS
            },
            "strict_alpha": {
                **_policy_json(strict),
                "max_regret": _json_number(float(strict_regret)),
            },
        },
        "note": "synthetic only; no MIMIC data, no clinical claim",
    }
    for key in ("dod", "death"):
        for field in ("tau_low", "tau_high", "cost"):
            value = payload["results"]["label_specific"][key][field]
            if isinstance(value, float) and not math.isfinite(value):
                payload["results"]["label_specific"][key][field] = None

    out_path = Path(args.output) if args.output else ARTIFACT_DIR / "demo.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {out_path}")
    return 0


def _fmt(x: float) -> str:
    return "inf" if not math.isfinite(float(x)) else f"{float(x):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
