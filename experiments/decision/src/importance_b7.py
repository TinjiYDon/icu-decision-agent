"""Average LightGBM gain importance over the five B7 seeds (42-46).

The single-seed table `table_feature_importance_B5.csv` describes B5; this one describes
the delivered B7 ensemble, so the report never has to guess which model a ranking belongs to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
CACHE = OUT / "cache"
RES = OUT / "results"
sys.path.insert(0, str(HERE))

SEEDS = (42, 43, 44, 45, 46)
LGB_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
                  min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, deterministic=True, num_threads=8, verbose=-1)

NEW_PREF = ("v_", "l_", "d_", "tr_", "sofa_", "qsofa", "news2", "sirs", "shock", "pf_",
            "sf_", "vaso_", "vent_", "uo_", "miss_", "n_")


def main():
    import lightgbm as lgb

    preds = pd.read_parquet(CACHE / "preds_full.parquet")
    stays = pl.read_parquet(CACHE / "stays.parquet").to_pandas()
    stays = stays[["stay_id", "intime_year", "first_careunit", "admission_type", "race",
                   "gender", "anchor_age", "male", "race_white", "race_black", "race_other"]]
    df = preds.merge(stays, on="stay_id", how="left")
    feat = pl.read_parquet(CACHE / "feat_new.parquet").to_pandas()
    feat = feat.drop(columns=[c for c in ("anchor_age", "male", "race_white", "race_black",
                                          "race_other") if c in feat.columns], errors="ignore")
    df = df.merge(feat, on=["stay_id", "hour_index"], how="left")

    static_cols = ["anchor_age", "male", "race_white", "race_black", "race_other",
                   "pre_adm_hours", "careunit_micu", "careunit_sicu", "careunit_ccu",
                   "careunit_other", "admit_emergency", "admit_urgent", "admit_elective",
                   "ins_medicare", "ins_medicaid", "ins_private"] + \
                  [c for c in df.columns if c.startswith(("elix_", "hx_"))]
    new_cols = [c for c in df.columns if c.startswith(NEW_PREF)]
    cols = [c for c in static_cols + new_cols if c in df.columns]
    X = np.array(df[cols].to_numpy(), dtype=np.float32, copy=True)
    X[~np.isfinite(X)] = np.nan
    tr = (df["fold"] == "train").values
    y = df["y"].values.astype(int)

    gains = np.zeros(len(cols))
    for s in SEEDS:
        m = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": s}).fit(X[tr], y[tr])
        gains += m.booster_.feature_importance(importance_type="gain")
        print(f"  seed {s} fitted", flush=True)
    out = pd.DataFrame({"feature": cols, "gain_mean_5seeds": gains / len(SEEDS)})
    out["gain_share"] = out["gain_mean_5seeds"] / out["gain_mean_5seeds"].sum()
    out.sort_values("gain_mean_5seeds", ascending=False).to_csv(
        RES / "tables" / "table_feature_importance_B7.csv", index=False)
    print(out.sort_values("gain_mean_5seeds", ascending=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
