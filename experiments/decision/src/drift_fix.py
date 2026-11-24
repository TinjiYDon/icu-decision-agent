"""Recompute ONLY the temporal-drift section of extras.json (bug fix pass).

The first extras run filtered B4's old-feature columns with the un-prefixed name, so the
drift comparison used a 1-2 column model. This recomputes the drift rows and the
per-era test table with the correct 19-column B4 matrix and merges them into
results/extras.json. Everything else in extras.json is left untouched; a full
`extras.py` rerun produces the same numbers (bug fixed there too).

    python experiments/decision/src/drift_fix.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
CACHE = OUT / "cache"
RES = OUT / "results"
import sys
sys.path.insert(0, str(HERE))
import eval_lib as E  # noqa: E402

LGB_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
                  min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, deterministic=True, num_threads=8, verbose=-1)
OLD19 = ("anchor_age", "careunit_other", "vital_temp", "vital_gcs_total", "vital_fio2",
         "vital_heart_rate", "vital_nbps", "vital_resp_rate", "lab_lactate", "lab_bun",
         "lab_ph", "lab_potassium", "lab_inr", "lab_albumin", "pre_icu_los_hours", "gcs_total",
         "vasopressor_1h", "shock_index", "spo2_fio2_ratio")
NEW_PREF = ("v_", "l_", "d_", "tr_", "sofa_", "qsofa", "news2", "sirs", "shock", "pf_",
            "sf_", "vaso_", "vent_", "uo_", "miss_", "n_")


def clean(X):
    X = np.array(X, dtype=np.float32, copy=True)
    X[~np.isfinite(X)] = np.nan
    return X


def fit(X, y, tr, va, te, seed=42):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": seed}).fit(X[tr], y[tr])
    return m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]


def main():
    preds = pd.read_parquet(CACHE / "preds_full.parquet")
    stays = pl.read_parquet(CACHE / "stays.parquet").to_pandas()
    stays = stays[["stay_id", "intime_year", "first_careunit", "admission_type", "race",
                   "gender", "anchor_age", "male", "race_white", "race_black", "race_other"]]
    df = preds.merge(stays, on="stay_id", how="left")
    feat = pl.read_parquet(CACHE / "feat_new.parquet").to_pandas()
    feat = feat.drop(columns=[c for c in ("anchor_age", "male", "race_white", "race_black",
                                          "race_other") if c in feat.columns], errors="ignore")
    old = pl.read_parquet(CACHE / "feat_old.parquet").to_pandas()
    old = old.rename(columns={c: f"old_{c}" for c in old.columns
                              if c not in ("stay_id", "hour_index")})
    df = df.merge(feat, on=["stay_id", "hour_index"], how="left")
    df = df.merge(old, on=["stay_id", "hour_index"], how="left")

    y = df["y"].values.astype(int)
    tr, va, te = ((df["fold"] == f).values for f in ("train", "val", "test"))
    yr = df["intime_year"].fillna(0).astype(int).values

    static_cols = ["anchor_age", "male", "race_white", "race_black", "race_other",
                   "pre_adm_hours", "careunit_micu", "careunit_sicu", "careunit_ccu",
                   "careunit_other", "admit_emergency", "admit_urgent", "admit_elective",
                   "ins_medicare", "ins_medicaid", "ins_private"] + \
                  [c for c in df.columns if c.startswith(("elix_", "hx_"))]
    new_cols = [c for c in df.columns if c.startswith(NEW_PREF)]
    cols_b5 = [c for c in static_cols + new_cols if c in df.columns]
    cols_b4 = [f"old_{c}" for c in OLD19 if f"old_{c}" in df.columns]
    print(f"B4 columns: {len(cols_b4)}  B5 columns: {len(cols_b5)}")

    hist = pd.Series(yr[te]).value_counts().sort_index()
    cum = hist.cumsum() / hist.sum()
    cut = int(cum[cum >= 0.70].index[0])
    print(f"drift cut year = {cut} (de-identified timeline)")

    X4, X5 = clean(df[cols_b4].to_numpy()), clean(df[cols_b5].to_numpy())
    rows = []
    for name, X in (("B4_old19", X4), ("B5_new", X5), ("B7_seedavg", X5)):
        tr_d = tr & (yr <= cut)
        va_d = va & (yr <= cut)
        te_d = te & (yr >= cut + 1)
        te_in = te & (yr <= cut)
        if name == "B7_seedavg":
            pv, pt = np.zeros(te_d.sum()), None
            acc, n = 0.0, 0
            for s in (42, 43, 44, 45, 46):
                _, p = fit(X, y, tr_d, va_d, te_d, seed=s)
                acc, n = acc + p, n + 1
            pt = acc / n
        else:
            pv, pt = fit(X, y, tr_d, va_d, te_d)
        m = E.point_metrics(y[te_d], pt)
        if name == "B7_seedavg":
            acc_in, n_in = 0.0, 0
            for s in (42, 43, 44, 45, 46):
                acc_in, n_in = acc_in + fit(X, y, tr_d, va_d, te_in, seed=s)[1], n_in + 1
            p_in = acc_in / n_in
        else:
            _, p_in = fit(X, y, tr_d, va_d, te_in)
        m_in = E.point_metrics(y[te_in], p_in)
        rows.append({"model": name, "train_era": f"<= {cut}", "test_era": f">= {cut+1}",
                     "n_train": int(tr_d.sum()), "n_test": int(te_d.sum()),
                     "n_pos_test": int(y[te_d].sum()),
                     "auprc": m["auprc"], "auroc": m["auroc"], "brier": m["brier"],
                     "ece15": m["ece15"], "cal_slope": m["cal_slope"],
                     "auprc_in_era_test": m_in["auprc"], "brier_in_era_test": m_in["brier"],
                     "ece_in_era_test": m_in["ece15"]})
        print(f"  {name}: late-era AUPRC {m['auprc']:.4f} | early-era {m_in['auprc']:.4f}")

    pd.DataFrame(rows).to_csv(RES / "tables" / "table_drift.csv", index=False)

    era = []
    for lo, hi, lbl in ((0, cut, f"early (<= {cut})"), (cut + 1, 9999, f"late (>= {cut+1})")):
        m_e = te & (yr >= lo) & (yr <= hi)
        for model in ("B4", "B5", "B7"):
            mm = E.point_metrics(y[m_e], df.loc[m_e, f"p_{model}"].values)
            era.append({"test_era": lbl, "model": model, "n": mm["n"], "n_pos": mm["n_pos"],
                        "auprc": mm["auprc"], "auroc": mm["auroc"], "brier": mm["brier"],
                        "ece15": mm["ece15"], "cal_slope": mm["cal_slope"]})
    pd.DataFrame(era).to_csv(RES / "tables" / "table_era_test.csv", index=False)

    ex = json.loads((RES / "extras.json").read_text(encoding="utf-8"))
    ex["drift_retrain"] = rows
    ex["era_test"] = era
    ex["drift_meta"] = {"cut_year": cut,
                        "timeline": "MIMIC-IV de-identified years (2110-2214); split on the "
                                    "year at the 70th percentile of test-slice volume",
                        "b4_columns": len(cols_b4), "b5_columns": len(cols_b5)}
    (RES / "extras.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1))
    print("extras.json drift section updated")


if __name__ == "__main__":
    main()
