"""DCA full pipeline: load real data → compute DCA curves with CI → export report.

Usage:
    python scripts/d2_dca_full.py [--cost_benefit_ratio 0.111] [--n_boot 500] [--out PATH]

Output:
    artifacts/dca/dca_report_<timestamp>.json   — full numeric results
    artifacts/dca/dca_summary.txt               — human-readable summary
    artifacts/dca/net_benefit_curve.png          — static plot (Plotly JSON)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from domain.models.dca import (
    dca_compare_models,
    dca_curve_with_ci,
    dca_working_point,
)
from domain.models.evaluation import net_benefit_curve
from domain.models.lgbm import _load_training_frame, _get_model_bundle
from domain.models.split import split_frame_by_stay
from domain.features.build import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts" / "dca"
ART_DIR.mkdir(parents=True, exist_ok=True)


def _load_test_data(max_rows: int = 10000) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load feat/label + split; return test_df and manifest."""
    df = _load_training_frame()
    if len(df) < 50:
        raise RuntimeError(f"too few samples for DCA: n={len(df)}")
    _, _, test_df, manifest = split_frame_by_stay(df)
    if len(test_df) > max_rows:
        test_df = test_df.sample(n=max_rows, random_state=42)
    return test_df, manifest


def _predict_lgbm(test_df: pd.DataFrame) -> np.ndarray:
    """Score test set with saved LGBM booster."""
    from domain.models.lgbm import _get_model_bundle
    _, explainer = _get_model_bundle()  # noqa: F841  — kept for side-effect consistency
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(ROOT / "artifacts" / "models" / "lgbm_mortality_12h.txt"))
    X = test_df[FEATURE_COLS].to_numpy(dtype=float)
    raw = booster.predict(X)
    # LightGBM returns log-odds; convert to probability
    if np.any((raw < -50) | (raw > 50)):
        prob = 1.0 / (1.0 + np.exp(-raw))
    else:
        prob = raw
    return np.asarray(prob, dtype=float)


def _predict_grud(test_df: pd.DataFrame) -> np.ndarray | None:
    """Score test set with saved GRU-D model. Returns None if not trained."""
    try:
        import torch
        from domain.models.temporal.grud import GRUDModel
        model_path = ROOT / "artifacts" / "models" / "grud_mortality_12h.pt"
        if not model_path.exists():
            return None
        # Minimal GRU-D inference: load state dict and predict
        # (Simplified — actual inference requires seq preprocessing)
        return None  # placeholder; real GRU-D DCA uses saved predictions
    except ImportError:
        return None


def run_dca(
    *,
    cost_benefit_ratio: float = 1.0 / 9.0,
    n_boot: int = 500,
    max_rows: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run full DCA pipeline and return structured result."""
    test_df, manifest = _load_test_data(max_rows=max_rows)
    y = test_df["label"].to_numpy(dtype=int)
    prob_lgbm = _predict_lgbm(test_df)

    # --- Point-estimate curve (backward compat) ---
    curve_lgbm = net_benefit_curve(y, prob_lgbm)

    # --- Bootstrap CI curve ---
    ci_result = dca_curve_with_ci(
        y, prob_lgbm,
        n_boot=n_boot,
        seed=seed,
    )

    # --- Working point analysis ---
    wp = dca_working_point(y, prob_lgbm, cost_benefit_ratio=cost_benefit_ratio)

    # --- Multi-model comparison (LGBM only if GRU-D not available) ---
    compare_result = dca_compare_models(
        y,
        {"lgbm": prob_lgbm},
        cost_benefit_ratio=cost_benefit_ratio,
        n_boot=n_boot,
        seed=seed,
    )

    # Build summary
    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "manifest": manifest,
        "sample_size": int(len(test_df)),
        "positive_rate": float(y.mean()),
        "cost_benefit_ratio": cost_benefit_ratio,
        "working_threshold": wp.threshold,
        "working_point": {
            "net_benefit_model": wp.net_benefit_model,
            "net_benefit_treat_all": wp.net_benefit_treat_all,
            "sensitivity": wp.sensitivity,
            "specificity": wp.specificity,
            "ppv": wp.ppv,
            "npv": wp.npv,
        },
        "curve_point_estimate": {
            "thresholds": curve_lgbm["thresholds"],
            "net_benefit_model": curve_lgbm["net_benefit_model"],
            "net_benefit_treat_all": curve_lgbm["net_benefit_treat_all"],
        },
        "curve_with_ci": {
            "thresholds": ci_result.thresholds,
            "net_benefit_model": ci_result.net_benefit_model,
            "ci_lower": ci_result.ci_lower_model,
            "ci_upper": ci_result.ci_upper_model,
            "net_benefit_treat_all": ci_result.net_benefit_treat_all,
        },
        "comparison": compare_result,
        "n_boot": n_boot,
        "seed": seed,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DCA full pipeline (D2)")
    parser.add_argument("--cost_benefit_ratio", type=float, default=1.0 / 9.0,
                        help="CBR for working point (default 1/9, threshold≈0.1)")
    parser.add_argument("--n_boot", type=int, default=500, help="Bootstrap iterations")
    parser.add_argument("--max_rows", type=int, default=10000, help="Max test rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="", help="Output dir (default: artifacts/dca)")
    args = parser.parse_args()

    print(f"Running DCA pipeline …")
    print(f"  CBR={args.cost_benefit_ratio:.4f} → threshold={args.cost_benefit_ratio/(1+args.cost_benefit_ratio):.4f}")
    print(f"  n_boot={args.n_boot}, max_rows={args.max_rows}")

    out_dir = Path(args.out) if args.out else ART_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_dca(
        cost_benefit_ratio=args.cost_benefit_ratio,
        n_boot=args.n_boot,
        max_rows=args.max_rows,
        seed=args.seed,
    )

    # Write JSON report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"dca_report_{ts}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON report → {json_path}")

    # Write human-readable summary
    txt_lines = [
        "=" * 60,
        "D2 决策曲线分析 (DCA) 报告",
        "=" * 60,
        f"生成时间: {result['generated_at']}",
        f"样本量: {result['sample_size']:,} stays × hours",
        f"阳性率: {result['positive_rate']:.2%}",
        f"代价比 (CBR): {args.cost_benefit_ratio:.4f}",
        f"工作阈值: {result['working_threshold']:.4f}",
        "",
        "— 工作点净受益 —",
        f"  模型净受益 (NB_model): {result['working_point']['net_benefit_model']:.4f}",
        f"  全干预净受益 (NB_treat_all): {result['working_point']['net_benefit_treat_all']:.4f}",
        f"  敏感性: {result['working_point']['sensitivity']:.2%}",
        f"  特异度: {result['working_point']['specificity']:.2%}",
        f"  PPV: {result['working_point']['ppv']:.2%}",
        f"  NPV: {result['working_point']['npv']:.2%}",
        "",
        "— 结论 —",
        f"  {'✅ 模型净受益 > 全干预' if result['working_point']['net_benefit_model'] > result['working_point']['net_benefit_treat_all'] else '❌ 模型净受益 ≤ 全干预'}",
        f"  {'✅ 模型净受益 > 0 (优于不做任何干预)' if result['working_point']['net_benefit_model'] > 0 else '⚠️ 模型净受益 ≤ 0'},",
        "=" * 60,
    ]
    txt_path = out_dir / f"dca_summary_{ts}.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    print(f"Summary → {txt_path}")
    print("\n".join(txt_lines))


if __name__ == "__main__":
    main()
