"""H4 CLI: compare max-F1 vs max-net-benefit thresholds on restored S2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb

from domain.features.build import FEATURE_COLS
from domain.models.lgbm import ARTIFACT_DIR, _load_training_frame
from domain.models.split import split_frame_by_stay
from domain.models.utility_calibrate import compare_f1_vs_utility

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "h4"


def main() -> None:
    p = argparse.ArgumentParser(description="H4 utility-calibrated threshold compare")
    p.add_argument("--cost-ratio", type=float, default=4.0)
    p.add_argument("--thr-min", type=float, default=0.05)
    p.add_argument("--thr-max", type=float, default=0.40)
    p.add_argument("--max-alert-rate", type=float, default=0.20)
    args = p.parse_args()

    model_path = ARTIFACT_DIR / "lgbm_mortality_12h.txt"
    if not model_path.is_file():
        raise SystemExit(f"missing model: {model_path}; run train --from-existing first")

    df = _load_training_frame()
    _train_df, val_df, test_df, _manifest = split_frame_by_stay(df)
    booster = lgb.Booster(model_file=str(model_path))
    p_val = booster.predict(val_df[FEATURE_COLS].to_numpy(dtype=float))
    p_test = booster.predict(test_df[FEATURE_COLS].to_numpy(dtype=float))
    y_val = val_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    payload = compare_f1_vs_utility(
        y_val,
        p_val,
        y_test,
        p_test,
        cost_ratio=float(args.cost_ratio),
        thr_min=float(args.thr_min),
        thr_max=float(args.thr_max),
        max_alert_rate=float(args.max_alert_rate),
    )
    payload["n_val"] = int(len(y_val))
    payload["n_test"] = int(len(y_test))
    payload["label_note"] = "old_S2_dump_dod_style_prevalence"

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "utility_compare.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    f1 = payload["f1_policy"]["test"]
    ut = payload["utility_policy"]["test"]
    free = payload["utility_unconstrained_diagnostic"]["test"]
    md = [
        "# H4 · Utility-Calibrated Early Warning",
        "",
        payload["note"],
        "",
        f"- cost_ratio FP:FN = {args.cost_ratio}:1",
        f"- search window = [{args.thr_min}, {args.thr_max}], max_alert_rate={args.max_alert_rate}",
        f"- n_val / n_test = {payload['n_val']} / {payload['n_test']}",
        "",
        "| policy | thr | test NB | alert_rate | F1 | PR-AUC | Brier |",
        "|--------|----:|--------:|-----------:|---:|-------:|------:|",
        (
            f"| max_F1 | {payload['f1_policy']['threshold']:.4f} | "
            f"{f1['net_benefit']:.4f} | {f1['alert_rate']:.4f} | {f1['f1']:.4f} | "
            f"{f1['pr_auc']:.4f} | {f1['brier']:.4f} |"
        ),
        (
            f"| max_NB constrained | {payload['utility_policy']['threshold']:.4f} | "
            f"{ut['net_benefit']:.4f} | {ut['alert_rate']:.4f} | {ut['f1']:.4f} | "
            f"{ut['pr_auc']:.4f} | {ut['brier']:.4f} |"
        ),
        (
            f"| max_NB unconstrained (diag) | {payload['utility_unconstrained_diagnostic']['threshold']:.4f} | "
            f"{free['net_benefit']:.4f} | {free['alert_rate']:.4f} | {free['f1']:.4f} | "
            f"{free['pr_auc']:.4f} | {free['brier']:.4f} |"
        ),
        "",
        f"- ΔNB (utility−F1) = {payload['test_delta']['net_benefit_utility_minus_f1']:+.4f}",
        f"- Δalert_rate = {payload['test_delta']['alert_rate_utility_minus_f1']:+.4f}",
        f"- pass: {payload['pass_criteria']}",
        "",
    ]
    (OUT / "utility_compare.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
