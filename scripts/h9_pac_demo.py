"""H9-I4 PAC 双阈值安全网 demo（synthetic only）。

用法（不读数据库、不调 LLM、不跑 MIMIC）：

    python scripts/h9_pac_demo.py --n 4000 --alpha 0.01 --delta 0.1 --seed 0

输出：终端摘要 + artifacts/h9/pac_demo.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from domain.models.pac_safety_net import (
    hoeffding_radius,
    missed_mass,
    release_mask,
    select_double_threshold,
)

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "h9"


def make_synthetic(n: int, seed: int):
    """p ~ Beta(0.4, 6.0)，y ~ Bernoulli(0.15*p + 0.02)。"""
    rng = np.random.default_rng(seed)
    p = rng.beta(0.4, 6.0, size=n)
    y = (rng.random(n) < 0.15 * p + 0.02).astype(float)
    return p, y


def main() -> int:
    ap = argparse.ArgumentParser(
        description="H9-I4 PAC double-threshold demo (synthetic only)"
    )
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-low", type=int, default=50)
    ap.add_argument("--n-high", type=int, default=50)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    if args.n < 1:
        raise SystemExit("--n 至少为 1")
    if args.alpha <= 0:
        raise SystemExit("--alpha 必须 > 0")
    if not 0.0 < args.delta < 1.0:
        raise SystemExit("--delta 必须落在 (0, 1)")

    p, y = make_synthetic(args.n, args.seed)
    res = select_double_threshold(
        p, y, alpha=args.alpha, delta=args.delta, n_low=args.n_low, n_high=args.n_high
    )

    n_pairs = sum(
        int(np.count_nonzero(np.linspace(0.0, 1.0, args.n_high) > lo))
        for lo in np.linspace(0.0, 1.0, args.n_low)
    )
    radius_m = hoeffding_radius(args.n, args.delta / max(n_pairs, 1))
    # 零风险下要让 U<=alpha 所需的最小校准集大小
    n_required = math.ceil(
        math.log(max(n_pairs, 1) / args.delta) / (2.0 * args.alpha**2)
    )

    # 对比：把固定单阈值 0.2 直接当 tau_low 的 Hoeffding 上界（m=1 与同 m 两种口径）
    fixed_low = 0.2
    fixed_release = release_mask(p, fixed_low)
    fixed_rate = float(np.mean(fixed_release))
    fixed_risk = missed_mass(p, y, fixed_low)
    fixed_upper_m1 = fixed_risk + hoeffding_radius(args.n, args.delta)
    fixed_upper_m = fixed_risk + radius_m

    lines = [
        "# H9-I4 PAC safety net (synthetic only)",
        "",
        (
            f"config: n={args.n} alpha={args.alpha} delta={args.delta} seed={args.seed} "
            f"grid={args.n_low}x{args.n_high} (pairs={n_pairs})"
        ),
        (
            f"hoeffding_radius (Bonferroni m={n_pairs}): {radius_m:.6f}  "
            f"| n_min(risk=0, U<=alpha): {n_required}"
        ),
        "",
        "| field | value |",
        "|---|---:|",
        f"| tau_low | {res.tau_low:.4f} |",
        f"| tau_high | {res.tau_high:.4f} |",
        f"| feasible | {res.feasible} |",
        f"| empirical_risk | {res.empirical_risk:.6f} |",
        f"| risk_upper_bound | {res.risk_upper_bound:.6f} |",
        f"| release / review / alert | {res.n_release} / {res.n_review} / {res.n_alert} |",
        f"| release_rate | {res.n_release / res.n_cal:.4f} |",
        f"| conditional_miss_rate | {res.conditional_miss_rate:.6f} |",
        "",
        (
            f"baseline fixed tau_low=0.2: release={int(fixed_release.sum())} "
            f"({fixed_rate:.4f}), R_hat={fixed_risk:.6f}, "
            f"U(m=1)={fixed_upper_m1:.6f}, U(m={n_pairs})={fixed_upper_m:.6f}"
        ),
        "",
        (
            "note: synthetic data (p~Beta(0.4,6.0), y~Bernoulli(0.15p+0.02)); "
            "no clinical evidence. feasible=False means no candidate passes the "
            "finite-sample bound at this n/alpha/grid."
        ),
    ]
    print("\n".join(lines))

    payload = {
        "config": {
            "n": args.n,
            "alpha": args.alpha,
            "delta": args.delta,
            "seed": args.seed,
            "n_low": args.n_low,
            "n_high": args.n_high,
            "n_pairs": n_pairs,
            "hoeffding_radius": radius_m,
            "n_required_zero_risk": n_required,
        },
        "result": asdict(res),
        "baseline_fixed_0_2": {
            "tau_low": fixed_low,
            "n_release": int(fixed_release.sum()),
            "release_rate": fixed_rate,
            "empirical_risk": fixed_risk,
            "risk_upper_bound_m1": fixed_upper_m1,
            "risk_upper_bound_m": fixed_upper_m,
        },
        "note": "synthetic only; no MIMIC data, no clinical claim",
    }

    out_path = Path(args.output) if args.output else ARTIFACT_DIR / "pac_demo.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
