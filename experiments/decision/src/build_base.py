"""Build the decision-experiment base layer: stays meta, two label sets, old-method
(B4) feature matrix from `paper_feat.sample_matrix`, and subject-level splits.

Writes to experiments/decision/cache/:
  stays.parquet     one row per stay (DISTINCT ON stay_id)
  labels.parquet    stay_id, hour_index, y_death, y_dod   (472,290 rows)
  feat_old.parquet  stay_id, hour_index + the 19 `allowed` features (B4 input)
And experiments/decision/splits.json.

Run with the project venv (lightgbm 4.7 / sklearn 1.9 / psycopg3 / pyarrow).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import psycopg

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

DSN = __import__("os").environ["ICU_DECISION_DSN"]
DSN_MIMIC = __import__("os").environ["MIMIC_DSN"]

# exactly `allowed:` in icu-decision-agent/configs/features.yaml (19 features)
OLD_FEATURES = [
    "anchor_age", "careunit_other", "vital_temp", "vital_gcs_total", "vital_fio2",
    "vital_heart_rate", "vital_nbps", "vital_resp_rate", "lab_lactate", "lab_bun",
    "lab_ph", "lab_potassium", "lab_inr", "lab_albumin", "pre_icu_los_hours",
    "gcs_total", "vasopressor_1h", "shock_index", "spo2_fio2_ratio",
]
HOURS = [0, 1, 2, 4, 6]


def tic(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_stays():
    t0 = time.time()
    sql = """
    SELECT DISTINCT ON (stay_id)
           stay_id, subject_id, hadm_id, first_careunit, last_careunit,
           intime, outtime, EXTRACT(EPOCH FROM (outtime - intime))/3600.0 AS los_hours
    FROM mimic.mimiciv_icu.icustays
    ORDER BY stay_id, intime
    """
    with psycopg.connect(DSN_MIMIC) as c:
        with c.cursor() as cur:
            cur.execute(sql)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
    stays = pl.DataFrame(rows, schema=cols, orient="row")
    tic(f"icustays dedup: {stays.height} rows ({time.time()-t0:.1f}s)")

    with psycopg.connect(DSN_MIMIC) as c:
        with c.cursor() as cur:
            cur.execute("""
                SELECT hadm_id, admittime, dischtime, deathtime, admission_type,
                       admission_location, insurance, race, hospital_expire_flag,
                       marital_status, EXTRACT(EPOCH FROM (intime - admittime))/3600.0 AS pre_adm_hours
                FROM mimic.mimiciv_hosp.admissions a
                JOIN (SELECT DISTINCT hadm_id, intime FROM mimic.mimiciv_icu.icustays) s USING (hadm_id)
            """)
            cols = [d.name for d in cur.description]
            adm = pl.DataFrame([r for r in cur.fetchall()], schema=cols, orient="row")
        with c.cursor() as cur:
            cur.execute("SELECT subject_id, gender, anchor_age, anchor_year, dod FROM mimic.mimiciv_hosp.patients")
            cols = [d.name for d in cur.description]
            pat = pl.DataFrame([r for r in cur.fetchall()], schema=cols, orient="row")
    tic(f"admissions {adm.height} / patients {pat.height}")

    # DISTINCT ON (hadm_id) in case admissions is duplicated too
    adm = adm.unique(subset=["hadm_id"], keep="first")
    stays = stays.join(adm, on="hadm_id", how="left").join(pat, on="subject_id", how="left")
    stays = stays.with_columns(
        (pl.col("intime").dt.year()).alias("intime_year"),
        (pl.col("gender") == "M").cast(pl.Int8).alias("male"),
        (pl.col("race").str.contains("WHITE")).cast(pl.Int8).alias("race_white"),
        (pl.col("race").str.contains("BLACK")).cast(pl.Int8).alias("race_black"),
    )
    stays = stays.with_columns(
        pl.when(pl.col("race_white") + pl.col("race_black") == 0).then(1).otherwise(0)
          .cast(pl.Int8).alias("race_other")
    )
    stays.write_parquet(CACHE / "stays.parquet")
    tic(f"stays.parquet -> {stays.height}x{stays.width} ({time.time()-t0:.1f}s)")
    return stays


def build_labels():
    t0 = time.time()
    with psycopg.connect(DSN) as c:
        y = pl.read_database("SELECT stay_id, hour_index, label FROM paper_label.mortality_12h", c)
        yd = pl.read_database("SELECT stay_id, hour_index, label AS label_dod FROM paper_label.mortality_12h_dod", c)
    lab = y.rename({"label": "y_death"}).join(yd, on=["stay_id", "hour_index"], how="full", coalesce=True)
    lab = lab.with_columns(pl.col("y_death").cast(pl.Int8), pl.col("label_dod").cast(pl.Int8).alias("y_dod")).drop("label_dod")
    assert lab.height == 472290, lab.height
    # y_death must be a subset of y_dod (LABEL_AUDIT.md claim)
    bad = lab.filter((pl.col("y_death") == 1) & (pl.col("y_dod") == 0)).height
    tic(f"labels: {lab.height} rows; y_death pos={int(lab['y_death'].sum())} "
        f"y_dod pos={int(lab['y_dod'].sum())}; death_not_dod={bad}")
    assert bad == 0, "y_death must imply y_dod"
    lab.write_parquet(CACHE / "labels.parquet")
    tic(f"labels.parquet ({time.time()-t0:.1f}s)")
    return lab


def build_old_features():
    t0 = time.time()
    exprs = ", ".join(
        f"(feature_json->>'{f}')::double precision AS {f}" for f in OLD_FEATURES
    )
    sql = f"SELECT stay_id, hour_index, {exprs} FROM paper_feat.sample_matrix"
    with psycopg.connect(DSN) as c:
        old = pl.read_database(sql, c)
    assert old.height == 472290, old.height
    nulls = {f: old[f].null_count() for f in OLD_FEATURES}
    tic("old-feature null counts: " + json.dumps(nulls))
    old.write_parquet(CACHE / "feat_old.parquet")
    tic(f"feat_old.parquet {old.height}x{old.width} ({time.time()-t0:.1f}s)")
    return old


def build_splits(stays: pl.DataFrame, lab: pl.DataFrame):
    """Subject-level stratified 70/10/20, seed=42. Stratum = subject ever-positive
    under y_death (any hour). Every hour of a stay inherits its subject's fold."""
    t0 = time.time()
    subj = (
        lab.join(stays.select(["stay_id", "subject_id"]), on="stay_id", how="inner")
        .group_by("subject_id")
        .agg(pl.col("y_death").max().alias("pos"))
    )
    rng = np.random.default_rng(42)
    subj = subj.with_columns(pl.Series("rand", rng.random(subj.height)))
    def fold_expr(pos):
        # within stratum: sort by rand, assign 70/10/20
        return pl.when(pl.col("pos") == pos).then(pl.col("rand")).otherwise(None)
    subj = subj.sort("rand")
    parts = {}
    for pos in (0, 1):
        s = subj.filter(pl.col("pos") == pos)
        n = s.height
        n_tr, n_va = int(round(0.7 * n)), int(round(0.1 * n))
        f = np.array(["train"] * n, dtype=object)
        f[n_tr:n_tr + n_va] = "val"
        f[n_tr + n_va:] = "test"
        parts[pos] = s.with_columns(pl.Series("fold", f))
    subj = pl.concat([parts[0], parts[1]])

    st = stays.select(["stay_id", "subject_id"]).join(
        subj.select(["subject_id", "fold", "pos"]), on="subject_id", how="inner")
    splits = {
        "seed": 42,
        "unit": "subject_id",
        "ratios": {"train": 0.7, "val": 0.1, "test": 0.2},
        "stratify": "subject ever y_death=1 over {0,1,2,4,6}",
        "n_subjects": {k: int((subj["fold"] == k).sum()) for k in ("train", "val", "test")},
        "n_stays": {k: int((st["fold"] == k).sum()) for k in ("train", "val", "test")},
        "subject_id": {k: sorted(subj.filter(pl.col("fold") == k)["subject_id"].to_list())
                       for k in ("train", "val", "test")},
        "stay_id": {k: sorted(st.filter(pl.col("fold") == k)["stay_id"].to_list())
                    for k in ("train", "val", "test")},
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "builder": "experiments/decision/src/build_base.py",
    }
    (OUT / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    tic(f"splits.json: subjects={splits['n_subjects']} stays={splits['n_stays']} ({time.time()-t0:.1f}s)")

    # row-level split assignment (stay, hour) for convenience
    roww = lab.join(st.select(["stay_id", "fold"]), on="stay_id", how="inner")
    roww.write_parquet(CACHE / "rows_split.parquet")
    return splits, roww


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    stays = build_stays()
    lab = build_labels()
    build_old_features()
    build_splits(stays, lab)
    tic("DONE")
