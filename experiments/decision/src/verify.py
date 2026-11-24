"""Recompute every headline number from the delivered prediction files and assert
that results/metrics.json / lare.json / extras.json agree.

    python experiments/decision/src/verify.py

Exit code 0 = all checks pass. Any mismatch is printed with the two values.
The script uses the same eval_lib functions as the training run, but it only needs
the delivered artefacts (preds_test.parquet + the JSON files), never the models.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
RES = OUT / "results"
sys.path.insert(0, str(HERE))
import eval_lib as E  # noqa: E402

FAILS: list[str] = []
CHECKS = [0]


def chk(name, a, b, tol=1e-9):
    CHECKS[0] += 1
    if a is None or b is None:
        ok = a is None and b is None
    elif isinstance(a, float) and (np.isnan(a) or np.isnan(b)):
        ok = np.isnan(a) and np.isnan(b)
    else:
        ok = abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))
    if not ok:
        FAILS.append(f"{name}: stored={a!r} recomputed={b!r}")
    return ok


def main():
    m = json.loads((RES / "metrics.json").read_text(encoding="utf-8"))
    p = pd.read_parquet(RES / "preds_test.parquet")
    y = p["y"].values.astype(int)
    sid = p["stay_id"].values
    print(f"preds_test.parquet: {p.shape}, positives={y.sum()}, stays={p['stay_id'].nunique()}")

    # ---------- 1. per-model point metrics / CI / budgets / cost ----------
    for name, mm in m["models"].items():
        col = f"p_{name}"
        if col not in p.columns:
            FAILS.append(f"missing column {col}")
            continue
        pv = p[col].values.astype(float)
        chk(f"{name}.n", mm["n"], len(y))
        chk(f"{name}.n_pos", mm["n_pos"], int(y.sum()))
        chk(f"{name}.auroc", mm["auroc"], E.roc_auc_score(y, pv), 1e-12)
        chk(f"{name}.auprc", mm["auprc"], E.average_precision_score(y, pv), 1e-12)
        chk(f"{name}.brier", mm["brier"], E.brier_score_loss(y, np.clip(pv, 0, 1)), 1e-12)
        chk(f"{name}.ece15", mm["ece15"], E.ece_score(y, pv), 1e-12)
        ic, sl = E.calibration_slope_intercept(y, pv)
        chk(f"{name}.cal_slope", mm["cal_slope"], sl, 1e-6)
        chk(f"{name}.cal_intercept", mm["cal_intercept"], ic, 1e-6)
        for b in mm["budgets"]:
            if b["sensitivity"] is None:
                continue
            r = E.budget_metrics(y, pv, b["budget_per_1000"])
            chk(f"{name}.sens@{b['budget_per_1000']}", b["sensitivity"], r["sensitivity"], 1e-9)
            chk(f"{name}.ppv@{b['budget_per_1000']}", b["ppv"], r["ppv"], 1e-9)
        c = E.cost_at_threshold(y, pv, mm["cost"]["threshold"])
        chk(f"{name}.cost_miss", mm["cost"]["miss"], c["miss"], 1e-12)
        chk(f"{name}.cost_fp", mm["cost"]["fp"], c["fp"], 1e-12)
        chk(f"{name}.cost_total", mm["cost"]["cost_total"], c["cost_total"], 1e-12)

    # ---------- 2. comparisons (paired bootstrap + DeLong) ----------
    for key, cmp in m["comparisons"].items():
        a, b = key.split("_vs_")
        pa, pb = p[f"p_{a}"].values, p[f"p_{b}"].values
        chk(f"{key}.auprc_abs", cmp["auprc_delta"]["abs"],
            m["models"][a]["auprc"] - m["models"][b]["auprc"], 1e-12)
        chk(f"{key}.auroc_abs", cmp["auroc_delta"]["abs"],
            m["models"][a]["auroc"] - m["models"][b]["auroc"], 1e-12)
        bb = E.BootBundle(y, pa)
        d = bb.paired_delta(E.BootBundle(y, pb), sid, "auprc", n_boot=200, seed=42)
        chk(f"{key}.de_long_p", cmp["delong"]["p"], E.delong_auc(y, pa, pb)["p"], 1e-9)
        chk(f"{key}.boot_mean_vs_200", cmp["auprc_delta"]["delta"], d["delta"], 0.05)

    # ---------- 3. LARE (thresholds from lare.json, applied to delivered preds) ----------
    lj = json.loads((RES / "lare.json").read_text(encoding="utf-8"))
    ydod = p["y_dod"].values.astype(int)
    pol = lj["policies_val"]
    rows = {}
    for k, cfg in pol.items():
        rows[k] = _eval_policy(p["p_B5"].values, y, ydod, cfg["tl"], cfg["th"])
    r = rows["R_robust"]
    chk("lare.robust_cost", lj["headline"]["robust_cost"], max(r["cost_death"], r["cost_dod"]), 1e-9)
    best_d = min(v["cost_death"] for v in rows.values())
    best_o = min(v["cost_dod"] for v in rows.values())
    mr = max(r["cost_death"] - best_d, r["cost_dod"] - best_o)
    chk("lare.max_regret", lj["headline"]["max_regret"], mr, 1e-9)
    jac = _jaccard(_review_mask(p["p_B5"].values, pol["A_death"]),
                   _review_mask(p["p_B5"].values, pol["B_dod"]))
    chk("lare.jaccard_review_A_vs_B", lj["headline"]["jaccard_review_A_vs_B"], jac, 1e-9)

    # ---------- 4. extras: ablation / per-hour / subgroup spot checks ----------
    ex_path = RES / "extras.json"
    if ex_path.exists():
        ex = json.loads(ex_path.read_text(encoding="utf-8"))
        pa = pd.read_parquet(RES / "preds_ablation.parquet")
        if "fold" not in pa.columns:      # older file: recover the fold from the split artefact
            rs = pd.read_parquet(OUT / "cache" / "rows_split.parquet")
            pa = pa.merge(rs[["stay_id", "hour_index", "fold"]],
                          on=["stay_id", "hour_index"], how="left")
        pa = pa[pa["fold"] == "test"].reset_index(drop=True)
        ya = pa["y"].values.astype(int)
        for row in ex["ablation"]:
            col = f"p_{row['variant']}"
            if col not in pa.columns:
                continue
            msk = ~np.isnan(pa[col].values)
            if msk.sum() < 100:
                continue
            chk(f"ablation.{row['variant']}.auprc", row["auprc"],
                E.average_precision_score(ya[msk], pa[col].values[msk]), 1e-9)
            chk(f"ablation.{row['variant']}.auroc", row["auroc"],
                E.roc_auc_score(ya[msk], pa[col].values[msk]), 1e-9)
        for row in ex["per_hour"]:
            msk = (p["hour_index"] == row["hour_index"]).values
            pv = p[f"p_{row['model']}"].values[msk]
            yh = y[msk]
            chk(f"per_hour.h{row['hour_index']}.{row['model']}.auprc", row["auprc"],
                E.average_precision_score(yh, pv), 1e-9)
    else:
        print("note: extras.json absent, skipping ablation checks")

    print(f"\n{CHECKS[0]} checks, {len(FAILS)} failures")
    for f in FAILS:
        print("  FAIL", f)
    if FAILS:
        sys.exit(1)
    print("VERIFY OK — every stored headline reproduces from preds_test.parquet")


def _review_mask(pv, cfg):
    return (pv >= cfg["tl"]) & (pv < cfg["th"])


def _eval_policy(pv, yd, yo, tl, th):
    release, review, alert = pv < tl, _review_mask(pv, ({"tl": tl, "th": th})), pv >= th
    out = {}
    for nm, yy in (("death", yd), ("dod", yo)):
        out[f"cost_{nm}"] = float(4.0 * (yy * release).mean() + 1.0 * ((1 - yy) * alert).mean()
                                  + 0.2 * review.mean())
        out[f"yield_{nm}"] = float(yy[review].mean()) if review.any() else float("nan")
    out["budget"] = float((review | alert).mean())
    return out


def _jaccard(a, b):
    u = int((a | b).sum())
    return 1.0 if u == 0 else float((a & b).sum()) / u


if __name__ == "__main__":
    main()
