"""H6 CLI: CUGEW suite — RWSG + SUFH + DEI/ANB/UEI on restored S2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb

from domain.features.build import FEATURE_COLS, prediction_hours
from domain.models.lgbm import ARTIFACT_DIR, _load_training_frame
from domain.models.sparsity_fusion import feature_completeness
from domain.models.split import split_frame_by_stay
from domain.models.sufh import recency_weighted_trajectory, run_cugew_compare
from domain.models.utility_calibrate import select_threshold_by_net_benefit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "h6"


def main() -> None:
    p = argparse.ArgumentParser(description="CUGEW innovation suite (H6)")
    p.add_argument("--hour-index", type=int, default=6)
    p.add_argument("--beta", type=float, default=0.35, help="recency decay")
    p.add_argument("--gate-power", type=float, default=1.0)
    p.add_argument("--cost-ratio", type=float, default=4.0)
    args = p.parse_args()

    model_path = ARTIFACT_DIR / "lgbm_mortality_12h.txt"
    if not model_path.is_file():
        raise SystemExit(f"missing model: {model_path}")

    df = _load_training_frame()
    _tr, val_df, test_df, _m = split_frame_by_stay(df)
    booster = lgb.Booster(model_file=str(model_path))
    df = df.copy()
    df["prob"] = booster.predict(df[FEATURE_COLS].to_numpy(dtype=float))
    hour_prob = {
        (int(r.stay_id), int(r.hour_index)): float(r.prob) for r in df.itertuples()
    }

    h = int(args.hour_index)
    hours = tuple(prediction_hours())
    va = val_df[val_df["hour_index"] == h].copy()
    te = test_df[test_df["hour_index"] == h].copy()
    if len(va) < 50 or len(te) < 50:
        raise SystemExit(f"too few rows at h={h}: val={len(va)} test={len(te)}")

    p_va_cur = booster.predict(va[FEATURE_COLS].to_numpy(dtype=float))
    thr = select_threshold_by_net_benefit(
        va["label"].to_numpy(dtype=int), p_va_cur, cost_ratio=float(args.cost_ratio)
    )

    def pack(frame):
        sids = frame["stay_id"].astype(int).tolist()
        p_cur = booster.predict(frame[FEATURE_COLS].to_numpy(dtype=float))
        p_rw = recency_weighted_trajectory(
            sids, h, hour_prob, hours=hours, beta=float(args.beta)
        )
        c = feature_completeness(frame, FEATURE_COLS)
        y = frame["label"].to_numpy(dtype=int)
        return y, p_cur, p_rw, c

    yv, pcv, prv, cv = pack(va)
    yt, pct, prt, ct = pack(te)

    payload = run_cugew_compare(
        y_val=yv,
        p_cur_val=pcv,
        p_rw_val=prv,
        c_val=cv,
        y_te=yt,
        p_cur_te=pct,
        p_rw_te=prt,
        c_te=ct,
        threshold=float(thr),
        cost_ratio=float(args.cost_ratio),
        gate_power=float(args.gate_power),
        beta=float(args.beta),
    )
    payload["hour_index"] = h
    payload["label_note"] = "old_S2_dump_dod_style_prevalence"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cugew_suite.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = [
        "# H6 · CUGEW Suite (RWSG + SUFH + DEI/ANB/UEI)",
        "",
        payload["note"],
        "",
        f"- hour_index={h} · β={args.beta} · cost_ratio={args.cost_ratio} · τ*={thr:.4f}",
        f"- best_by_UEI = **{payload['best_by_UEI']}**",
        f"- SUFH coefs = {payload['sufh_coefficients']}",
        "",
        "| policy | PR-AUC | Brier | NB | DEI | ANB | UEI | alert |",
        "|--------|-------:|------:|---:|----:|----:|----:|------:|",
    ]
    for r in payload["rows"]:
        md.append(
            f"| {r['policy']} | {r['pr_auc']:.4f} | {r['brier']:.4f} | "
            f"{r['net_benefit']:.4f} | {r['DEI']:.4f} | {r['ANB']:.4f} | "
            f"{r['UEI']:.4f} | {r['alert_rate']:.4f} |"
        )
    (OUT / "cugew_suite.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
