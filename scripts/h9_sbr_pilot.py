"""H9 SBR pilot：在合成病例轨迹上对比复核策略。

用法（不读数据库、不调 LLM，可在没有 MIMIC 授权的环境下运行）：

    python scripts/h9_sbr_pilot.py --n 2000 --horizon 5 --budget 3 --seed 0

输出：终端表格 + artifacts/h9/sbr_pilot_<seed>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from domain.models.selective_review import (
    CostModel,
    DoubleThresholdPolicy,
    FixedThresholdPolicy,
    MyopicReviewPolicy,
    Reviewer,
    default_transition,
    evaluate_policy,
    make_synthetic_cases,
    solve_sbr,
)

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "h9"


def main() -> int:
    ap = argparse.ArgumentParser(description="H9 SBR pilot (synthetic only)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=5, help="决策时刻数（>=2）")
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--cost-fn", type=float, default=4.0)
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--review-cost", type=float, default=0.2)
    ap.add_argument("--sensitivity", type=float, default=0.80)
    ap.add_argument("--specificity", type=float, default=0.80)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    if args.horizon < 2:
        raise SystemExit("--horizon 至少为 2")

    cost = CostModel(args.cost_fn, args.cost_fp, args.review_cost)
    reviewer = Reviewer(args.sensitivity, args.specificity)
    trajectories, labels = make_synthetic_cases(args.n, args.horizon, args.seed)

    grid = np.linspace(0.0, 1.0, args.grid)
    transition = default_transition(grid)
    sbr = solve_sbr(grid, transition, cost, reviewer, horizon=args.horizon - 1, budget=args.budget)

    policies = {
        "release_all": FixedThresholdPolicy(1.01),
        "alert_all": FixedThresholdPolicy(-0.01),
        "fixed_0.5": FixedThresholdPolicy(0.5),
        "fixed_0.384": FixedThresholdPolicy(0.384),  # H4 旧 S2 工作点
        "double_threshold": DoubleThresholdPolicy(0.2, 0.6),
        "myopic_review": MyopicReviewPolicy(cost, reviewer, threshold=0.5),
        "sbr_dp": sbr,
    }

    rows = []
    for name, policy in policies.items():
        metrics = evaluate_policy(policy, trajectories, labels, cost, reviewer,
                                  budget=args.budget, seed=args.seed + 1)
        rows.append({"policy": name, **metrics})

    print(f"n={args.n} horizon={args.horizon} budget={args.budget} "
          f"reviewer sens/spec={args.sensitivity}/{args.specificity}")
    print("| policy | mean_cost | miss_rate | alert_rate | review_rate | review_yield | oracle_gap |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['policy']} | {r['mean_cost']:.4f} | {r['miss_rate']:.3f} | "
              f"{r['alert_rate']:.3f} | {r['review_rate']:.3f} | {r['review_yield']:.3f} | "
              f"{r['oracle_gap']:.4f} |")

    out = Path(args.output) if args.output else ARTIFACT_DIR / f"sbr_pilot_{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "rows": rows,
        "note": "synthetic trajectories; simulated reviewer; no clinical claim",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
