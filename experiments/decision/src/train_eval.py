"""Train and evaluate the decision model zoo B0–B7 on one subject-level split.

Usage (project venv):
    python experiments/decision/src/train_eval.py --suite old     # provisional: B0/B3/B4
    python experiments/decision/src/train_eval.py --suite full    # B0–B7 + tables

The test fold is scored exactly once per model; thresholds/threshold-selection and
isotonic calibration are fitted on val only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
CACHE = OUT / "cache"
RES = OUT / "results"
DATA = OUT.parent / "data"
sys.path.insert(0, str(HERE))
import eval_lib as E  # noqa: E402

BUDGETS = (20, 50, 100)
LGB_PARAMS = dict(
    objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, deterministic=True, num_threads=8, verbose=-1,
)
SEEDS = (42, 43, 44, 45, 46)


def tic(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_all(suite="full"):
    stays = pl.read_parquet(CACHE / "stays.parquet")
    labels = pl.read_parquet(CACHE / "labels.parquet")
    fold = pl.read_parquet(CACHE / "rows_split.parquet").select(["stay_id", "hour_index", "fold"])
    old = pl.read_parquet(CACHE / "feat_old.parquet")
    meta_cols = ["stay_id", "subject_id", "intime_year", "first_careunit", "male",
                 "race_white", "race_black", "race_other", "anchor_age", "admission_type",
                 "insurance", "pre_adm_hours", "los_hours", "deathtime", "dod",
                 "hospital_expire_flag"]
    new = None
    if suite == "full":
        fn = CACHE / "feat_new.parquet"
        if not fn.exists():
            raise SystemExit(f"missing {fn}: run build_features.py first")
        new = pl.read_parquet(fn)
        # feat_new carries its own copies of the static descriptors (but not the join keys)
        meta_cols = ["stay_id"] + [c for c in meta_cols if c != "stay_id" and c not in new.columns]
    base = (labels.join(fold, on=["stay_id", "hour_index"], how="inner")
                  .join(old, on=["stay_id", "hour_index"], how="left")
                  .join(stays.select(meta_cols), on="stay_id", how="left")
                  .sort(["stay_id", "hour_index"]))
    assert base.height == 472290, base.height
    if new is not None:
        base = base.join(new, on=["stay_id", "hour_index"], how="left")
        sf = DATA / "sofa_hourly.parquet"
        if sf.exists():
            sh = (pl.read_parquet(sf).select(["stay_id", "hour_index", "sofa_total"])
                    .rename({"sofa_total": "sofa_total_layer"}))
            base = base.join(sh, on=["stay_id", "hour_index"], how="left")
            tic("joined ETL sofa_hourly (B1 clinical score)")
        tic(f"joined new features: {base.width} cols")
    return base


def split_arrays(df: pd.DataFrame, cols, fold):
    m = df["fold"].values == fold
    X = df.loc[m, cols].to_numpy(dtype=np.float32, na_value=np.nan)
    y = df.loc[m, "y_death"].to_numpy(dtype=np.int8)
    sid = df.loc[m, "stay_id"].to_numpy()
    return X, y, sid


def clean(X):
    X = np.array(X, dtype=np.float32, copy=True)
    X[~np.isfinite(X)] = np.nan
    return X


def lgb_fit_predict(Xtr, ytr, Xva, Xte, seed=42, params=None):
    import lightgbm as lgb
    p = dict(LGB_PARAMS)
    if params:
        p.update(params)
    p["random_state"] = seed
    m = lgb.LGBMClassifier(**p)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["old", "full"], default="old")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    t0 = time.time()
    df = load_all(args.suite).to_pandas()
    tag = args.tag or args.suite
    tic(f"data ready {df.shape}")

    OLD_COLS = json.loads((OUT / "cache" / "old_cols.json").read_text()) \
        if (CACHE / "old_cols.json").exists() else None
    if OLD_COLS is None:
        from build_base import OLD_FEATURES
        OLD_COLS = OLD_FEATURES
    NEW_COLS = [c for c in df.columns if c.startswith(("v_", "l_", "d_", "tr_", "sofa_", "qsofa",
                                                       "news2", "sirs", "shock", "pf_", "sf_",
                                                       "vaso_", "vent_", "uo_", "miss_", "n_"))]
    STATIC_COLS = ["anchor_age", "male", "race_white", "race_black", "race_other",
                   "pre_adm_hours", "careunit_micu", "careunit_sicu", "careunit_ccu",
                   "careunit_other", "admit_emergency", "admit_urgent", "admit_elective",
                   "ins_medicare", "ins_medicaid", "ins_private"] + \
                  [c for c in df.columns if c.startswith("elix_")] + \
                  [c for c in df.columns if c.startswith("hx_")]

    models = {}
    preds = pd.DataFrame({
        "stay_id": df["stay_id"].values,
        "hour_index": df["hour_index"].values,
        "subject_id": df["subject_id"].values,
        "fold": df["fold"].values,
        "y": df["y_death"].values,
        "y_dod": df["y_dod"].values,
    })

    tr = df["fold"].values == "train"
    va = df["fold"].values == "val"
    te = df["fold"].values == "test"
    ytr, yva, yte = df.loc[tr, "y_death"].values, df.loc[va, "y_death"].values, df.loc[te, "y_death"].values

    # ---------------- B0 base rate
    p0 = np.full(len(df), ytr.mean())
    preds["p_B0"] = p0
    tic(f"B0 base rate {ytr.mean():.5f}")

    # ---------------- B4 LightGBM old 19 features
    Xo = clean(df[OLD_COLS].to_numpy())
    pv, pt, m4 = lgb_fit_predict(Xo[tr], ytr, Xo[va], Xo[te], seed=42)
    preds["p_B4"], models["B4"] = np.nan, m4
    preds.loc[va, "p_B4"] = pv
    preds.loc[te, "p_B4"] = pt
    tic(f"B4 done ({time.time()-t0:.0f}s)")

    if args.suite == "full":
        # ---------------- B1 SOFA total
        sf_col = "sofa_total_layer" if "sofa_total_layer" in df.columns else "sofa_total"
        sf_all = df[sf_col].astype(float)
        n_missing = int(sf_all.isna().sum())
        sf_all = sf_all.fillna(float(np.nanmedian(sf_all[tr])))   # train median impute
        tic(f"B1 score={sf_col}, imputed {n_missing} missing")
        sv = sf_all[va].values
        st = sf_all[te].values
        str_ = sf_all[tr].values
        # map score -> probability with a 1-D isotonic fit on train (monotone, no test use)
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5)
        ir.fit(str_, ytr)
        preds.loc[va, "p_B1"] = ir.predict(sv)
        preds.loc[te, "p_B1"] = ir.predict(st)
        preds["p_B1"] = np.nan
        preds.loc[va, "p_B1"] = ir.predict(sv)
        preds.loc[te, "p_B1"] = ir.predict(st)
        tic(f"B1 SOFA done (score={sf_col})")

        # ---------------- B2 qSOFA / NEWS2 rule baseline (LR on the 2 scores)
        from sklearn.linear_model import LogisticRegression
        b2c = ["qsofa", "news2"]
        Xb = clean(df[b2c].to_numpy())
        mu, sd = np.nanmean(Xb[tr], 0), np.nanstd(Xb[tr], 0) + 1e-9
        Xb = np.nan_to_num((Xb - mu) / sd, nan=0.0)
        lr2 = LogisticRegression(C=1.0, max_iter=500).fit(Xb[tr], ytr)
        preds.loc[va, "p_B2"] = lr2.predict_proba(Xb[va])[:, 1]
        preds.loc[te, "p_B2"] = lr2.predict_proba(Xb[te])[:, 1]
        tic("B2 qSOFA/NEWS2 done")

        # ---------------- B3 LR on full new feature set
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, FunctionTransformer
        from sklearn.linear_model import LogisticRegression as LR
        F3 = [c for c in NEW_COLS + STATIC_COLS if c in df.columns]
        X3 = clean(df[F3].to_numpy())
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clip", FunctionTransformer(lambda X: np.clip(X, -8, 8))),
            ("lr", LR(C=0.1, max_iter=300, solver="lbfgs")),
        ])
        pipe.fit(X3[tr], ytr)
        preds.loc[va, "p_B3"] = pipe.predict_proba(X3[va])[:, 1]
        preds.loc[te, "p_B3"] = pipe.predict_proba(X3[te])[:, 1]
        tic(f"B3 LR done ({time.time()-t0:.0f}s)")

        # ---------------- B5 LightGBM full features
        F5 = [c for c in STATIC_COLS + NEW_COLS if c in df.columns]
        X5 = clean(df[F5].to_numpy())
        pv, pt, m5 = lgb_fit_predict(X5[tr], ytr, X5[va], X5[te], seed=42)
        preds.loc[va, "p_B5"], preds.loc[te, "p_B5"] = pv, pt
        models["B5"] = m5
        imp = pd.DataFrame({"feature": F5, "gain": m5.booster_.feature_importance("gain")})
        imp.sort_values("gain", ascending=False).to_csv(RES / "tables" / "table_feature_importance_B5.csv", index=False)
        tic(f"B5 done ({time.time()-t0:.0f}s)")

        # ---------------- B6 B5 + isotonic fitted on val
        from sklearn.isotonic import IsotonicRegression
        ir6 = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5)
        ir6.fit(pv, yva)
        preds.loc[va, "p_B6"] = ir6.predict(pv)
        preds.loc[te, "p_B6"] = ir6.predict(pt)
        tic("B6 isotonic done")

        # ---------------- B7 5-seed average
        vs, ts = [], []
        for s in SEEDS:
            pv_s, pt_s, _ = lgb_fit_predict(X5[tr], ytr, X5[va], X5[te], seed=s)
            vs.append(pv_s); ts.append(pt_s)
            tic(f"  B7 seed {s} done")
        preds.loc[va, "p_B7"] = np.mean(vs, 0)
        preds.loc[te, "p_B7"] = np.mean(ts, 0)
        tic(f"B7 done ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------ evaluation
    pcols = [c for c in preds.columns if c.startswith("p_B")]
    te_df = preds[te].reset_index(drop=True)
    va_df = preds[va].reset_index(drop=True)
    y_te = te_df["y"].values
    y_va = va_df["y"].values

    metrics = {
        "meta": {
            "suite": args.suite, "tag": tag,
            "n_total": int(len(preds)),
            "n_test": int(te_df.shape[0]), "n_test_stays": int(te_df["stay_id"].nunique()),
            "n_test_pos": int(y_te.sum()), "test_prevalence": float(y_te.mean()),
            "n_val": int(va_df.shape[0]), "n_val_stays": int(va_df["stay_id"].nunique()),
            "n_val_pos": int(y_va.sum()),
            "split": json.loads((OUT / "splits.json").read_text())["n_stays"],
            "lgb_params": LGB_PARAMS, "seeds": list(SEEDS),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "script": "experiments/decision/src/train_eval.py",
        },
        "models": {}, "comparisons": {},
    }

    for c in pcols:
        name = c[2:]
        pv_ = va_df[c].values.astype(float)
        pt_ = te_df[c].values.astype(float)
        m = E.point_metrics(y_te, pt_, budgets=BUDGETS)
        if len(np.unique(pt_)) <= 1:
            for b in m["budgets"]:
                b["sensitivity"] = None; b["ppv"] = None; b["nnr"] = None
            m["degenerate_ranking"] = True
        bb = E.BootBundle(y_te, pt_)
        m["ci"] = bb.bootstrap(te_df["stay_id"].values, n_boot=args.boot, seed=42)
        # operating point from val: cost-minimising single threshold (4:1)
        tau, k, cval = E.best_cost_threshold_on_val(y_va, pv_)
        m["cost"] = E.cost_at_threshold(y_te, pt_, tau)
        m["cost"]["threshold_from"] = "val_min_cost_4to1"
        m["cost"]["val_cost_at_tau"] = cval
        m["cost"]["val_alert_rate"] = k / len(y_va)
        m["val"] = {"auroc": float(E.roc_auc_score(y_va, pv_)) if len(np.unique(y_va)) > 1 else np.nan,
                    "auprc": float(E.average_precision_score(y_va, pv_)) if len(np.unique(y_va)) > 1 else np.nan,
                    "brier": float(E.brier_score_loss(y_va, np.clip(pv_, 0, 1)))}
        m["dca"] = E.dca_curve(y_te, pt_).to_dict(orient="records")
        metrics["models"][name] = m
        tic(f"eval {name}: AUPRC={m['auprc']:.4f} AUROC={m['auroc']:.4f} Brier={m['brier']:.4f}")

    # ---------------- comparisons vs B4
    def cmp(a, b):
        pa, pb = te_df[f"p_{a}"].values, te_df[f"p_{b}"].values
        sid = te_df["stay_id"].values
        bba, bbb = E.BootBundle(y_te, pa), E.BootBundle(y_te, pb)
        out = {
            "auprc_delta": bba.paired_delta(bbb, sid, "auprc", n_boot=args.boot),
            "auroc_delta": bba.paired_delta(bbb, sid, "auroc", n_boot=args.boot),
            "delong": E.delong_auc(y_te, pa, pb),
        }
        out["auprc_delta"]["abs"] = metrics["models"][a]["auprc"] - metrics["models"][b]["auprc"]
        out["auroc_delta"]["abs"] = metrics["models"][a]["auroc"] - metrics["models"][b]["auroc"]
        out["brier_delta_abs"] = metrics["models"][a]["brier"] - metrics["models"][b]["brier"]
        out["ece_delta_abs"] = metrics["models"][a]["ece15"] - metrics["models"][b]["ece15"]
        return out

    if te_df["p_B4"].notna().any():
        for a in pcols:
            nm = a[2:]
            if nm in ("B0", "B4"):
                continue
            metrics["comparisons"][f"{nm}_vs_B4"] = cmp(nm, "B4")
            tic(f"compare {nm} vs B4 done")
        if "B6" in [c[2:] for c in pcols] and "B5" in [c[2:] for c in pcols]:
            metrics["comparisons"]["B6_vs_B5"] = cmp("B6", "B5")
            metrics["comparisons"]["B7_vs_B5"] = cmp("B7", "B5")

    # ------------------------------------------------------------ write outputs
    (RES / "tables").mkdir(parents=True, exist_ok=True)
    keep = ["stay_id", "hour_index", "subject_id", "fold", "y", "y_dod"] + pcols
    preds_out = preds[keep].copy()
    preds_out.to_parquet(CACHE / f"preds_{tag}.parquet", index=False)
    preds_out.loc[te].to_parquet(CACHE / f"preds_test_{tag}.parquet", index=False)
    preds_out.loc[va].to_parquet(CACHE / f"preds_val_{tag}.parquet", index=False)

    # main table
    rows = []
    for c in pcols:
        n = c[2:]
        m = metrics["models"][n]
        rows.append({
            "model": n, "n": m["n"], "n_pos": m["n_pos"],
            "auroc": m["auroc"], "auroc_lo": m["ci"]["auroc"]["lo"], "auroc_hi": m["ci"]["auroc"]["hi"],
            "auprc": m["auprc"], "auprc_lo": m["ci"]["auprc"]["lo"], "auprc_hi": m["ci"]["auprc"]["hi"],
            "brier": m["brier"], "brier_lo": m["ci"]["brier"]["lo"], "brier_hi": m["ci"]["brier"]["hi"],
            "ece15": m["ece15"], "ece_lo": m["ci"]["ece15"]["lo"], "ece_hi": m["ci"]["ece15"]["hi"],
            "cal_slope": m["cal_slope"], "cal_intercept": m["cal_intercept"],
            "logloss": m["logloss"],
            "sens@50": m["budgets"][1]["sensitivity"],
            "ppv@50": m["budgets"][1]["ppv"], "nnr@50": m["budgets"][1]["nnr"],
            "cost_4to1": m["cost"]["cost_total"],
        })
    mt = pd.DataFrame(rows)
    mt.to_csv(RES / "tables" / "table_main.csv", index=False)

    # budget table
    brows = []
    for c in pcols:
        n = c[2:]
        for b in metrics["models"][n]["budgets"]:
            brows.append({"model": n, **b})
    pd.DataFrame(brows).to_csv(RES / "tables" / "table_budget.csv", index=False)

    # DCA table (long)
    drows = []
    for c in pcols:
        n = c[2:]
        for r in metrics["models"][n]["dca"]:
            drows.append({"model": n, **r})
    pd.DataFrame(drows).to_csv(RES / "tables" / "table_dca.csv", index=False)

    (RES / f"metrics_{tag}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    if args.suite == "full":
        json.dump(F5, open(CACHE / "cols_full.json", "w"), ensure_ascii=False)
    tic(f"wrote metrics_{tag}.json and tables ({time.time()-t0:.0f}s)")
    print(json.dumps({k: {"auprc": round(v["auprc"], 4), "auroc": round(v["auroc"], 4),
                          "brier": round(v["brier"], 4)} for k, v in metrics["models"].items()},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
