"""H5 CLI: sparsity-gated fusion of current-hour vs multi-hour LGBM trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from domain.features.build import FEATURE_COLS, prediction_hours
from domain.models.lgbm import ARTIFACT_DIR, _load_training_frame
from domain.models.sparsity_fusion import (
    compare_fusion_policies,
    feature_completeness,
    trajectory_from_hour_map,
)
from domain.models.split import split_frame_by_stay
from domain.models.utility_calibrate import select_threshold_by_net_benefit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "h5"


def main() -> None:
    p = argparse.ArgumentParser(description="H5 sparsity-gated multi-hour fusion")
    p.add_argument("--hour-index", type=int, default=6)
    p.add_argument("--gate-power", type=float, default=1.0)
    p.add_argument("--cost-ratio", type=float, default=4.0)
    args = p.parse_args()

    model_path = ARTIFACT_DIR / "lgbm_mortality_12h.txt"
    if not model_path.is_file():
        raise SystemExit(f"missing model: {model_path}; run train --from-existing first")

    df = _load_training_frame()
    _train, val_df, test_df, _m = split_frame_by_stay(df)
    booster = lgb.Booster(model_file=str(model_path))

    # Score all hours once
    X_all = df[FEATURE_COLS].to_numpy(dtype=float)
    probs = booster.predict(X_all)
    df = df.copy()
    df["prob"] = probs
    hour_prob = {
        (int(r.stay_id), int(r.hour_index)): float(r.prob) for r in df.itertuples()
    }

    h = int(args.hour_index)
    te = test_df[test_df["hour_index"] == h].copy()
    if len(te) < 50:
        raise SystemExit(f"too few test rows at hour_index={h}: {len(te)}")

    # Threshold from val at same hour (utility policy, aligned with H4)
    va = val_df[val_df["hour_index"] == h]
    p_va = booster.predict(va[FEATURE_COLS].to_numpy(dtype=float))
    thr = select_threshold_by_net_benefit(
        va["label"].to_numpy(dtype=int), p_va, cost_ratio=float(args.cost_ratio)
    )

    stay_ids = te["stay_id"].astype(int).tolist()
    p_cur = booster.predict(te[FEATURE_COLS].to_numpy(dtype=float))
    p_traj = trajectory_from_hour_map(
        stay_ids, h, hour_prob, hours=tuple(prediction_hours())
    )
    comp = feature_completeness(te, FEATURE_COLS)
    y = te["label"].to_numpy(dtype=int)

    payload = compare_fusion_policies(
        y,
        p_cur,
        p_traj,
        comp,
        threshold=float(thr),
        gate_power=float(args.gate_power),
    )
    payload["hour_index"] = h
    payload["threshold_source"] = "max_net_benefit_on_val_same_hour"
    payload["label_note"] = "old_S2_dump_dod_style_prevalence"

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sparsity_fusion.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# H5 · Sparsity-Gated Dual Fusion",
        "",
        payload["note"],
        "",
        f"- hour_index = {h}",
        f"- threshold (val max-NB, cost_ratio={args.cost_ratio}) = {thr:.4f}",
        f"- mean completeness = {payload.get('mean_completeness')}",
        f"- mean gate = {payload.get('mean_gate')}",
        "",
        "| policy | n | PR-AUC | Brier | ROC | F1 | NB |",
        "|--------|--:|-------:|------:|----:|---:|---:|",
    ]
    for r in payload["rows"]:
        md.append(
            f"| {r['policy']} | {r['n']} | {r['pr_auc']:.4f} | {r['brier']:.4f} | "
            f"{r['roc_auc']:.4f} | {r['f1']:.4f} | {r.get('net_benefit')} |"
        )
    md.append("")
    md.append(f"- pass: {payload['pass_criteria']}")
    (OUT / "sparsity_fusion.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
