"""Secondary analyses: feature-group ablation, temporal drift, subgroups, per-hour.

Trains extra LightGBM variants on the SAME subject-level split / hyper-parameters as
B5, so any difference is attributable to the feature block or the training era.

Outputs
  results/tables/table_ablation.csv
  results/tables/table_drift.csv
  results/tables/table_subgroup.csv
  results/tables/table_per_hour.csv
  results/extras.json
  results/preds_ablation.parquet   (stay_id, hour_index, y, p_<variant>)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
RES = OUT / "results"
CACHE = OUT / "cache"
sys.path.insert(0, str(HERE))
import eval_lib as E  # noqa: E402

LGB_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
                  min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, deterministic=True, num_threads=8, verbose=-1)


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def clean(X):
    X = np.array(X, dtype=np.float32, copy=True)
    X[~np.isfinite(X)] = np.nan
    return X


def fit_predict(X, y, tr, va, te, seed=42, **over):
    import lightgbm as lgb
    p = dict(LGB_PARAMS)
    p.update(over)
    p["random_state"] = seed
    m = lgb.LGBMClassifier(**p).fit(X[tr], y[tr])
    return m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1], m


def groups(cols):
    g = {
        "static": [c for c in cols if c in ("anchor_age", "male", "race_white", "race_black",
                                            "race_other", "pre_adm_hours", "careunit_micu",
                                            "careunit_sicu", "careunit_ccu", "admit_emergency",
                                            "admit_urgent", "admit_elective", "ins_medicare",
                                            "ins_medicaid", "ins_private", "hour_index")]
                  + [c for c in cols if c.startswith("hx_")],
        "vitals": [c for c in cols if c.startswith("v_")],
        "labs": [c for c in cols if c.startswith("l_")],
        "trend": [c for c in cols if c.startswith("tr_")],
        "derived": [c for c in cols if c.startswith("sofa_") or c in
                    ("qsofa", "news2", "sirs", "shock_index", "supp_o2", "pf_ratio", "sf_ratio",
                     "hr_range", "sbp_range", "temp_range", "hr_ratio_max_min", "n_obs_total",
                     "vaso_any", "vaso_rate_max", "vaso_rate_last", "vaso_active", "vent_active",
                     "uo_24h", "uo_n", "uo_rate")],
    }
    g["gcs"] = [c for c in cols if "gcs" in c]
    return g


def main():
    t0 = time.time()
    preds = pd.read_parquet(CACHE / "preds_full.parquet")
    stays = pl.read_parquet(CACHE / "stays.parquet").to_pandas()
    stays = stays[["stay_id", "intime_year", "first_careunit", "admission_type", "race",
                   "gender", "anchor_age", "male", "race_white", "race_black",
                   "race_other"]]
    df = preds.merge(stays, on="stay_id", how="left", suffixes=("", "_st"))
    feat = pl.read_parquet(CACHE / "feat_new.parquet").to_pandas()
    # col name collision guard: feat_new carries anchor_age/male/race_*; use preds' join
    feat = feat.drop(columns=[c for c in ("anchor_age", "male", "race_white", "race_black",
                                          "race_other") if c in feat.columns], errors="ignore")
    old = pl.read_parquet(CACHE / "feat_old.parquet").to_pandas()
    old = old.rename(columns={c: f"old_{c}" for c in old.columns
                              if c not in ("stay_id", "hour_index")})
    df = df.merge(feat, on=["stay_id", "hour_index"], how="left")
    df = df.merge(old, on=["stay_id", "hour_index"], how="left")
    tic(f"loaded preds+features {df.shape}")

    tr = (df["fold"] == "train").values
    va = (df["fold"] == "val").values
    te = (df["fold"] == "test").values
    y = df["y"].values.astype(int)
    sid = df["stay_id"].values
    yva = y[va]

    # Exactly B5's feature matrix (same order as train_eval: STATIC_COLS + NEW_COLS), so the
    # "full" ablation row reproduces the delivered B5 number instead of a look-alike.
    NEW_PREF = ("v_", "l_", "d_", "tr_", "sofa_", "qsofa", "news2", "sirs", "shock", "pf_",
                "sf_", "vaso_", "vent_", "uo_", "miss_", "n_")
    new_cols = [c for c in df.columns if c.startswith(NEW_PREF)]
    static_cols = ["anchor_age", "male", "race_white", "race_black", "race_other",
                   "pre_adm_hours", "careunit_micu", "careunit_sicu", "careunit_ccu",
                   "careunit_other", "admit_emergency", "admit_urgent", "admit_elective",
                   "ins_medicare", "ins_medicaid", "ins_private"] + \
                  [c for c in df.columns if c.startswith("elix_")] + \
                  [c for c in df.columns if c.startswith("hx_")]
    allcols = [c for c in static_cols + new_cols if c in df.columns]
    G = groups(allcols)
    tic("groups: " + json.dumps({k: len(v) for k, v in G.items()}))
    Xall = clean(df[allcols].to_numpy())

    rows_ab, preds_ab = [], {"stay_id": df["stay_id"].values,
                             "hour_index": df["hour_index"].values,
                             "fold": df["fold"].values, "y": y}

    def evaluate_variant(name, cols, **over):
        X = clean(df[cols].to_numpy()) if cols != allcols else Xall
        pv, pt, _ = fit_predict(X, y, tr, va, te, **over)
        m = E.point_metrics(y[te], pt)
        tau, _, _ = E.best_cost_threshold_on_val(yva, pv)
        cost = E.cost_at_threshold(y[te], pt, tau)["cost_total"]
        rows_ab.append({"variant": name, "n_features": len(cols), "auprc": m["auprc"],
                        "auroc": m["auroc"], "brier": m["brier"], "ece15": m["ece15"],
                        "cost_4to1": cost})
        preds_ab[f"p_{name}"] = np.full(len(y), np.nan)
        preds_ab[f"p_{name}"][va] = pv
        preds_ab[f"p_{name}"][te] = pt
        tic(f"  {name:28s} k={len(cols):4d} AUPRC={m['auprc']:.4f} AUROC={m['auroc']:.4f}")
        return m

    # ---- 1. additive feature-group ablation
    tic("additive ablation")
    order = ["static", "labs", "vitals", "derived", "trend"]
    cum = []
    for k in order:
        cum = cum + G[k]
        evaluate_variant("add_" + "+".join(order[:order.index(k) + 1]), cum)

    # ---- 2. leave-one-group-out from the full set
    tic("leave-one-group-out ablation")
    evaluate_variant("full", allcols)
    for k in ["vitals", "labs", "trend", "derived", "gcs", "hx"]:
        drop = set([c for c in allcols if c.startswith("hx_")]) if k == "hx" else set(G[k])
        cols = [c for c in allcols if c not in drop]
        evaluate_variant(f"drop_{k}", cols)

    # ---- 3. class weighting on/off
    tic("class weight ablation")
    evaluate_variant("full_classweight", allcols, scale_pos_weight=8.0, n_estimators=300)

    ab = pd.DataFrame(rows_ab)
    ab.to_csv(RES / "tables" / "table_ablation.csv", index=False)
    (RES / "_partial_ablation.json").write_text(json.dumps(rows_ab, ensure_ascii=False, indent=1))
    tic("ablation written")
    pd.DataFrame(preds_ab).to_parquet(RES / "preds_ablation.parquet", index=False)

    # ---- 4. per-hour metrics from the delivered predictions (no retrain)
    prow = []
    for h in (0, 1, 2, 4, 6):
        m_h = (df["hour_index"] == h).values & te
        if m_h.sum() == 0:
            continue
        for model in ("B4", "B5", "B6", "B7"):
            pc = f"p_{model}"
            m = E.point_metrics(y[m_h], df.loc[m_h, pc].values)
            prow.append({"hour_index": h, "model": model, "n": m["n"], "n_pos": m["n_pos"],
                         "prevalence": m["prevalence"], "auprc": m["auprc"],
                         "auroc": m["auroc"], "brier": m["brier"], "ece15": m["ece15"]})
    ph = pd.DataFrame(prow)
    ph.to_csv(RES / "tables" / "table_per_hour.csv", index=False)

    # ---- 5. temporal drift: train <=2016 -> test >=2017
    tic("drift")
    drift_rows = []
    yr = df["intime_year"].fillna(0).values
    yrs = df.loc[te, "intime_year"].dropna().astype(int)
    if len(yrs) == 0:
        raise SystemExit("intime_year missing: cannot run drift split")
    hist = yrs.value_counts().sort_index()
    cum = (hist.cumsum() / hist.sum())
    cut = int(cum[cum >= 0.70].index[0])   # last year of the "early" era on the test side
    tic(f"drift cut year (de-identified timeline) = {cut}")
    for era_train, era_test in ((cut, cut + 1),):
        tr_d = tr & (yr <= era_train)
        te_d = te & (yr >= era_test)
        if tr_d.sum() < 1000 or te_d.sum() < 100:
            continue
        for name, cols in (("B4_old19", ["old_" + c for c in ("anchor_age", "careunit_other", "vital_temp",
                                                   "vital_gcs_total", "vital_fio2", "vital_heart_rate",
                                                   "vital_nbps", "vital_resp_rate", "lab_lactate",
                                                   "lab_bun", "lab_ph", "lab_potassium", "lab_inr",
                                                   "lab_albumin", "pre_icu_los_hours", "gcs_total",
                                                   "vasopressor_1h", "shock_index",
                                                   "spo2_fio2_ratio") if "old_" + c in df.columns]),
                             ("B5_new", allcols)):
            if name == "B4_old19":
                X = clean(df[cols].to_numpy())
            else:
                X = Xall
            pv, pt, _ = fit_predict(X, y, tr_d, va & (yr <= era_train), te_d)
            m = E.point_metrics(y[te_d], pt)
            # in-era control: same model on the in-era test slice
            te_in = te & (yr <= era_train)
            p_in = fit_predict(X, y, tr_d, va & (yr <= era_train), te_in)[1]
            m_in = E.point_metrics(y[te_in], p_in)
            drift_rows.append({
                "model": name, "train_era": f"<= {era_train}", "test_era": f">= {era_test}",
                "n_train": int(tr_d.sum()), "n_test": int(te_d.sum()),
                "n_pos_test": int(y[te_d].sum()),
                "auprc": m["auprc"], "auroc": m["auroc"], "brier": m["brier"],
                "ece15": m["ece15"], "cal_slope": m["cal_slope"],
                "auprc_in_era_test": m_in["auprc"], "brier_in_era_test": m_in["brier"],
                "ece_in_era_test": m_in["ece15"],
            })
            tic(f"  drift {name}: AUPRC {m['auprc']:.4f} (in-era {m_in['auprc']:.4f})")
    dr = pd.DataFrame(drift_rows)
    dr.to_csv(RES / "tables" / "table_drift.csv", index=False)
    (RES / "_partial_drift.json").write_text(json.dumps(drift_rows, ensure_ascii=False, indent=1))
    tic(f"drift written ({len(drift_rows)} rows)")

    # also: no-retrain drift of B5/B4 across test eras
    era_rows = []
    for lo, hi, lbl in ((0, cut, f"early (<= {cut})"),
                        (cut + 1, 9999, f"late (>= {cut + 1})")):
        m_e = te & (yr >= lo) & (yr <= hi)
        if m_e.sum() < 100:
            continue
        for model in ("B4", "B5", "B7"):
            m = E.point_metrics(y[m_e], df.loc[m_e, f"p_{model}"].values)
            era_rows.append({"test_era": lbl, "model": model, "n": m["n"], "n_pos": m["n_pos"],
                             "auprc": m["auprc"], "auroc": m["auroc"], "brier": m["brier"],
                             "ece15": m["ece15"], "cal_slope": m["cal_slope"]})
    pd.DataFrame(era_rows).to_csv(RES / "tables" / "table_era_test.csv", index=False)

    # ---- 6. subgroups (delivered predictions only, no retrain)
    tic("subgroups")
    unit = df["first_careunit"].fillna("NA").astype(str)
    age = df["anchor_age"].fillna(-1).values
    grp_defs = {
        "age_lt65": age < 65, "age_ge65": (age >= 65) & (age >= 0),
        "male": (df["male"].fillna(0).values == 1), "female": (df["male"].fillna(0).values == 0),
        "race_white": df["race_white"].fillna(0).values == 1,
        "race_black": df["race_black"].fillna(0).values == 1,
        "race_other": df["race_other"].fillna(0).values == 1,
        "unit_MICU": unit.str.contains("MICU").values,
        "unit_SICU": unit.str.contains("SICU").values,
        "unit_CCU": (unit.str.contains("CCU") | unit.str.contains("CVICU")).values,
        "unit_other": (~(unit.str.contains("MICU") | unit.str.contains("SICU")
                         | unit.str.contains("CCU") | unit.str.contains("CVICU"))).values,
        "admit_emergency": df["admission_type"].fillna("").isin(
            ["EW EMER.", "DIRECT EMER."]).values,
        "admit_urgent": (df["admission_type"] == "URGENT").values,
        "admit_elective": df["admission_type"].fillna("").isin(
            ["ELECTIVE", "SURGICAL SAME DAY ADMISSION"]).values,
    }
    srows = []
    for gname, mask in grp_defs.items():
        m_g = mask & te
        if m_g.sum() < 200 or y[m_g].sum() < 10:
            continue
        for model in ("B4", "B5", "B6", "B7"):
            m = E.point_metrics(y[m_g], df.loc[m_g, f"p_{model}"].values)
            srows.append({"subgroup": gname, "model": model, "n": m["n"], "n_pos": m["n_pos"],
                          "prevalence": m["prevalence"], "auprc": m["auprc"],
                          "auroc": m["auroc"], "brier": m["brier"], "ece15": m["ece15"]})
    sg = pd.DataFrame(srows)
    sg.to_csv(RES / "tables" / "table_subgroup.csv", index=False)

    extras = {
        "meta": {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "script": "experiments/decision/src/extras.py",
                 "lgb_params": LGB_PARAMS,
                 "note": "ablation/drift retrain on the same split; subgroups use delivered preds"},
        "ablation": ab.to_dict(orient="records"),
        "drift_retrain": dr.to_dict(orient="records"),
        "era_test": era_rows,
        "per_hour": ph.to_dict(orient="records"),
        "subgroup": sg.to_dict(orient="records"),
        "feature_groups": {k: len(v) for k, v in G.items()},
    }
    (RES / "extras.json").write_text(json.dumps(extras, ensure_ascii=False, indent=1))
    tic(f"extras.json written ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
