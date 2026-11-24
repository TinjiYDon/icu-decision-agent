"""Build the new causal feature matrix from the shared data layer.

Contract: PROTOCOL §2 (`experiments/data/{vitals,labs,inputevents,procedureevents,
uroutput}.parquet`) + cache/stays.parquet + cache/static_extra.parquet.
Causality: a row for (stay_id, h) may only use rows with charttime < intime + h,
with a 24 h lookback floor (charttime >= intime - 24h).

Output: cache/feat_new.parquet  (stay_id, hour_index, ~330 features)

Everything is computed per hour_index and concatenated, so the five hour rows of a
stay never see each other's future.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path(__file__).resolve().parents[1]
CACHE = OUT / "cache"
DATA = OUT.parent / "data"
HOURS = [0, 1, 2, 4, 6]
LOOKBACK = 24.0

# ------------------------------------------------------------------ itemid maps
VIT = {
    220045: "hr",
    220210: "rr", 224690: "rr",
    220277: "spo2",
    223762: "temp", 223761: "temp",
    220050: "sbp", 220179: "sbp",
    220051: "dbp", 220180: "dbp",
    220052: "map", 220181: "map",
    220739: "gcs_eye", 223901: "gcs_motor", 223900: "gcs_verbal",
    223835: "fio2",
    220339: "peep",
    223849: "ventmode",
    226732: "o2dev",
    226512: "weight", 224639: "weight",
}
VIT_VARS = ["hr", "rr", "spo2", "temp", "sbp", "dbp", "map", "gcs", "fio2",
            "peep", "ventmode", "o2dev", "weight"]
VIT_STATS = ["last", "mean", "min", "max", "count", "hsince"]

LAB = {
    50813: "lactate", 52442: "lactate", 53154: "lactate",
    50912: "creatinine", 52546: "creatinine",
    51006: "bun", 52647: "bun",
    50971: "potassium", 52610: "potassium",
    50983: "sodium", 52623: "sodium",
    50882: "bicarbonate",
    50885: "bilirubin", 53089: "bilirubin",
    51265: "platelets", 53189: "platelets",
    51301: "wbc", 51755: "wbc",
    51222: "hemoglobin",
    51221: "hematocrit",
    50931: "glucose",
    50821: "po2",
    50818: "pco2",
    50820: "ph",
    50902: "chloride",
    50960: "magnesium",
    50893: "calcium",
    51237: "inr",
    51275: "pt",
    50862: "albumin",
    50878: "ast",
    50861: "alt",
    50863: "alp",
    51002: "troponin",
    50963: "bnp",
}
LAB_VARS = ["lactate", "creatinine", "bun", "potassium", "sodium", "bicarbonate",
            "bilirubin", "platelets", "wbc", "hemoglobin", "hematocrit", "glucose",
            "po2", "pco2", "ph", "chloride", "magnesium", "calcium", "inr", "pt",
            "albumin", "ast", "alt", "alp", "troponin", "bnp"]
LAB_STATS = ["last", "mean", "min", "max", "count", "hsince"]

TREND_VARS = ["hr", "sbp", "map", "lactate", "creatinine", "rr", "spo2"]

VASO = {221906: "norepi", 221289: "epi", 221662: "dopamine", 221653: "dobutamine",
        222315: "vasopressin", 221749: "phenyl"}
VENT_IDS = {225792, 225794}


def tic(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_table(name: str) -> pl.DataFrame | None:
    for ext in (".parquet", ".csv"):
        p = DATA / f"{name}{ext}"
        if p.exists():
            t0 = time.time()
            df = pl.read_parquet(p) if ext == ".parquet" else pl.read_csv(p, infer_schema_length=10000)
            tic(f"read {p.name}: {df.shape} ({time.time()-t0:.1f}s)")
            return df
    return None


def pick_col(df: pl.DataFrame, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


def prep_vitals(vit: pl.DataFrame, stays: pl.DataFrame) -> pl.DataFrame:
    vcol = pick_col(vit, "valuenum", "value")
    vit = vit.select(["stay_id", "charttime", vcol]).rename({vcol: "val"})
    vit = vit.with_columns(pl.col("val").cast(pl.Float64))
    if "itemid" in vit.columns:
        pass
    vit = vit.with_columns(pl.col("stay_id").cast(pl.Int64),
                           pl.col("charttime").cast(pl.Datetime("us")))
    return vit


def windowed(long: pl.DataFrame, stays: pl.DataFrame, h: float) -> pl.DataFrame:
    """Attach dt (hours since intime) and keep the causal window for hour h."""
    d = long.join(stays.select(["stay_id", "intime"]), on="stay_id", how="inner")
    d = d.with_columns(
        ((pl.col("charttime") - pl.col("intime")).dt.total_seconds() / 3600.0).alias("dt"))
    d = d.filter((pl.col("dt") >= -LOOKBACK) & (pl.col("dt") < float(h)))
    return d


def stats_long(d: pl.DataFrame, varmap: dict, prefix: str, h: float, stats) -> pl.DataFrame:
    """Per (stay, var) aggregates -> wide frame with prefix."""
    g = (d.group_by(["stay_id", "var"])
           .agg([
               pl.col("val").sort_by("dt").last().alias("last"),
               pl.col("val").mean().alias("mean"),
               pl.col("val").min().alias("min"),
               pl.col("val").max().alias("max"),
               pl.col("val").len().alias("count"),
               pl.col("dt").max().alias("dtmax"),
           ]))
    out = None
    for s in stats:
        if s == "hsince":
            pv = (g.select(["stay_id", "var", "dtmax"])
                    .with_columns((float(h) - pl.col("dtmax")).alias("hsince")))
            w = pv.pivot(on="var", index="stay_id", values="hsince")
            w = w.rename({c: f"{prefix}{c}_hsince" for c in w.columns if c != "stay_id"})
        else:
            w = g.pivot(on="var", index="stay_id", values=s)
            w = w.rename({c: f"{prefix}{c}_{s}" for c in w.columns if c != "stay_id"})
        out = w if out is None else out.join(w, on="stay_id", how="full", coalesce=True)
    return out


def trend_long(d: pl.DataFrame, h: float) -> pl.DataFrame:
    """Least-squares slope over (t-6h, t] requiring >= 3 points."""
    e = d.filter(pl.col("dt") >= float(h) - 6.0)
    e = e.filter(pl.col("var").is_in(TREND_VARS))
    g = (e.group_by(["stay_id", "var"])
          .agg([
              pl.col("val").len().alias("n"),
              pl.col("dt").sum().alias("sx"),
              pl.col("val").sum().alias("sy"),
              (pl.col("dt") * pl.col("val")).sum().alias("sxy"),
              (pl.col("dt") * pl.col("dt")).sum().alias("sxx"),
          ]))
    g = g.with_columns(
        ((pl.col("n") * pl.col("sxy") - pl.col("sx") * pl.col("sy")) /
         (pl.col("n") * pl.col("sxx") - pl.col("sx") ** 2)).alias("slope"))
    g = g.with_columns(
        pl.when((pl.col("n") >= 3) & (pl.col("n") * pl.col("sxx") - pl.col("sx") ** 2 > 1e-9))
          .then(pl.col("slope")).otherwise(None).alias("slope"))
    w = g.pivot(on="var", index="stay_id", values="slope")
    w = w.rename({c: f"tr_{c}_slope" for c in w.columns if c != "stay_id"})
    return w


# ------------------------------------------------------------------ score helpers
def _pts(cond_list):
    """Sum of indicator vectors."""
    out = np.zeros_like(cond_list[0], dtype=np.float64)
    for c in cond_list:
        out += np.nan_to_num(c, nan=0.0)
    return out


def sofa_parts(df: pl.DataFrame) -> pl.DataFrame:
    np.seterr(divide="ignore", invalid="ignore")
    d = df.to_pandas()
    g = lambda c: d[c].to_numpy(dtype=float) if c in d.columns else np.full(len(d), np.nan)

    fio2 = g("v_fio2_last")
    fio2 = np.where(fio2 > 1.5, fio2 / 100.0, fio2)          # % -> fraction
    fio2 = np.where((fio2 < 0.21) | (fio2 > 1.0), np.nan, fio2)
    po2 = g("l_po2_min")
    spo2 = g("v_spo2_min")
    vent = np.nan_to_num(g("vent_active"), nan=0.0)

    pf = np.where(np.isfinite(po2) & np.isfinite(fio2) & (fio2 > 0), po2 / fio2, np.nan)
    sf = np.where(np.isfinite(spo2) & np.isfinite(fio2) & (fio2 > 0), spo2 / fio2, np.nan)

    # respiratory: prefer P/F, fall back to S/F only when P/F missing
    resp_pf = np.select(
        [~np.isfinite(pf), pf >= 400, pf >= 300, pf >= 200, pf >= 100],
        [np.nan, 0, 1, 2, 3], default=4)
    resp_sf = np.select(
        [~np.isfinite(sf), sf >= 512, sf >= 357, sf >= 214, sf >= 89],
        [np.nan, 0, 1, 2, 3], default=4)
    resp = np.where(np.isfinite(resp_pf), resp_pf, resp_sf)
    resp = np.where((resp >= 3) & (vent == 0), np.minimum(resp, 2), resp)   # vent-dependent
    resp = np.nan_to_num(resp, nan=0.0)

    plt_ = g("l_platelets_min")
    coag = np.select([~np.isfinite(plt_), plt_ >= 150, plt_ >= 100, plt_ >= 50, plt_ >= 20],
                     [0, 0, 1, 2, 3], default=4)
    bili = g("l_bilirubin_max")
    liver = np.select([~np.isfinite(bili), bili < 1.2, bili < 2.0, bili < 6.0, bili < 12.0],
                      [0, 0, 1, 2, 3], default=4)
    mp = g("v_map_min")
    vaso = np.nan_to_num(g("vaso_active"), nan=0.0)
    rate = np.nan_to_num(g("vaso_rate_max"), nan=0.0)
    cardio = np.select([~np.isfinite(mp), mp >= 70, mp < 70], [0, 0, 1], default=1)
    cardio = np.where(vaso > 0, np.where(rate > 0.2, 4, 3), cardio)
    gcs = g("v_gcs_min")
    cns = np.select([~np.isfinite(gcs), gcs >= 15, gcs >= 13, gcs >= 10, gcs >= 6],
                    [0, 0, 1, 2, 3], default=4)
    cr = g("l_creatinine_max")
    uo = g("uo_24h")
    renal = np.select([~np.isfinite(cr), cr < 1.2, cr < 2.0, cr < 3.5, cr < 5.0],
                      [0, 0, 1, 2, 3], default=4)
    renal = np.where(np.isfinite(uo) & (uo < 200), 4, np.where(np.isfinite(uo) & (uo < 500), 3, renal))

    out = pl.DataFrame({
        "stay_id": d["stay_id"].values,
        "sofa_resp": resp, "sofa_coag": coag, "sofa_liver": liver,
        "sofa_cardio": cardio, "sofa_cns": cns, "sofa_renal": renal,
    })
    out = out.with_columns((pl.col("sofa_resp") + pl.col("sofa_coag") + pl.col("sofa_liver")
                            + pl.col("sofa_cardio") + pl.col("sofa_cns") + pl.col("sofa_renal")
                            ).alias("sofa_total"))
    out = out.with_columns(pl.Series("pf_ratio", pf), pl.Series("sf_ratio", sf))
    return out


def rule_scores(df: pl.DataFrame) -> pl.DataFrame:
    np.seterr(divide="ignore", invalid="ignore")
    d = df.to_pandas()
    g = lambda c: d[c].to_numpy(dtype=float) if c in d.columns else np.full(len(d), np.nan)
    rr, spo2, sbp, hr = g("v_rr_last"), g("v_spo2_last"), g("v_sbp_last"), g("v_hr_last")
    temp, gcs, wbc, pco2 = g("v_temp_last"), g("v_gcs_last"), g("l_wbc_max"), g("l_pco2_max")
    fio2 = g("v_fio2_last")
    supp_o2 = ((np.nan_to_num(fio2, nan=21) > 21) | (np.nan_to_num(g("v_o2dev_count"), nan=0) > 0)
               | (np.nan_to_num(g("vent_active"), nan=0) > 0)).astype(float)

    qsofa = _pts([rr >= 22, sbp <= 100, gcs < 15])
    sirs = _pts([(temp > 38) | (temp < 36), hr > 90, (rr > 20) | (pco2 < 32),
                 (wbc > 12) | (wbc < 4)])
    news_rr = np.select([~np.isfinite(rr), rr <= 8, rr <= 11, rr <= 20, rr <= 24],
                        [0, 3, 1, 0, 2], default=3)
    news_spo2 = np.select([~np.isfinite(spo2), spo2 <= 91, spo2 <= 93, spo2 <= 95],
                          [0, 3, 2, 1], default=0)
    news_sbp = np.select([~np.isfinite(sbp), sbp <= 90, sbp <= 100, sbp <= 110, sbp <= 219],
                         [0, 3, 2, 1, 0], default=3)
    news_hr = np.select([~np.isfinite(hr), hr <= 40, hr <= 50, hr <= 90, hr <= 110, hr <= 130],
                        [0, 3, 1, 0, 1, 2], default=3)
    news_temp = np.select([~np.isfinite(temp), temp <= 35, temp <= 36, temp <= 38, temp <= 39],
                          [0, 3, 1, 0, 1], default=2)
    news_cns = np.where(np.isfinite(gcs) & (gcs < 15), 3.0, 0.0)
    news2 = news_rr + news_spo2 + news_sbp + news_hr + news_temp + news_cns + 2 * supp_o2
    den = np.where(np.isfinite(sbp) & (sbp > 0), sbp, np.nan)
    shi = hr / den
    return pl.DataFrame({
        "stay_id": d["stay_id"].values,
        "qsofa": qsofa, "sirs": sirs, "news2": news2, "shock_index": shi,
        "supp_o2": supp_o2,
    })


# ------------------------------------------------------------------ main build
def build(hours=HOURS, allow_missing=False):
    t0 = time.time()
    stays = pl.read_parquet(CACHE / "stays.parquet").select(
        ["stay_id", "subject_id", "intime", "anchor_age", "male", "race_white", "race_black",
         "race_other", "pre_adm_hours", "first_careunit", "admission_type", "insurance"])
    stat = pl.read_parquet(CACHE / "static_extra.parquet")

    vit = read_table("vitals")
    lab = read_table("labs")
    inp = read_table("inputevents")
    prc = read_table("procedureevents")
    uro = read_table("uroutput")

    def empty(cols):
        return pl.DataFrame({c: pl.Series(c, [], dtype=d) for c, d in cols})

    EMPTY = {
        "vitals": [("stay_id", pl.Int64), ("charttime", pl.Datetime("us")),
                   ("itemid", pl.Int64), ("valuenum", pl.Float64)],
        "labs": [("stay_id", pl.Int64), ("charttime", pl.Datetime("us")),
                 ("itemid", pl.Int64), ("valuenum", pl.Float64)],
        "inputevents": [("stay_id", pl.Int64), ("starttime", pl.Datetime("us")),
                        ("endtime", pl.Datetime("us")), ("itemid", pl.Int64), ("rate", pl.Float64)],
        "procedureevents": [("stay_id", pl.Int64), ("starttime", pl.Datetime("us")),
                            ("endtime", pl.Datetime("us")), ("itemid", pl.Int64)],
        "uroutput": [("stay_id", pl.Int64), ("charttime", pl.Datetime("us")),
                     ("itemid", pl.Int64), ("value", pl.Float64)],
    }
    for nm, df in (("vitals", vit), ("labs", lab), ("inputevents", inp),
                   ("procedureevents", prc), ("uroutput", uro)):
        if df is None:
            if not allow_missing:
                raise SystemExit(f"missing experiments/data/{nm}.parquet — data layer not ready "
                                 f"(reported by data-etl per PROTOCOL §2)")
            tic(f"WARNING: {nm}.parquet missing, using empty frame (smoke mode, NOT results)")
            EMPTY[nm] = empty(EMPTY[nm])
    vit = vit if vit is not None else EMPTY["vitals"]
    lab = lab if lab is not None else EMPTY["labs"]
    inp = inp if inp is not None else EMPTY["inputevents"]
    prc = prc if prc is not None else EMPTY["procedureevents"]
    uro = uro if uro is not None else EMPTY["uroutput"]

    # ---- vitals: map itemid -> var, GCS sum, Fahrenheit -> Celsius
    vcol = pick_col(vit, "valuenum", "value")
    vit = vit.select(["stay_id", "charttime", "itemid", vcol]).rename({vcol: "val"})
    vit = vit.with_columns(pl.col("val").cast(pl.Float64), pl.col("itemid").cast(pl.Int64),
                           pl.col("charttime").cast(pl.Datetime("us")))
    n0 = vit.height
    vit = vit.filter(pl.col("itemid").is_in(list(VIT.keys())))
    vit = vit.with_columns(pl.col("itemid").replace_strict(VIT, return_dtype=pl.Utf8).alias("var"))
    # device/vent-mode items are categorical (valuenum is null): keep them as presence rows
    vit = vit.with_columns(
        pl.when(pl.col("var").is_in(["ventmode", "o2dev"])).then(pl.lit(1.0))
          .otherwise(pl.col("val")).alias("val"))
    vit = vit.filter(pl.col("val").is_not_null())
    vit = vit.with_columns(
        pl.when((pl.col("var") == "temp") & (pl.col("itemid") == 223761))
          .then((pl.col("val") - 32.0) * 5.0 / 9.0).otherwise(pl.col("val")).alias("val"))
    # GCS components -> total per timestamp
    gcs = (vit.filter(pl.col("var").is_in(["gcs_eye", "gcs_motor", "gcs_verbal"]))
              .group_by(["stay_id", "charttime"])
              .agg(pl.col("val").sum().alias("val"))
              .with_columns(pl.lit("gcs").alias("var"), pl.lit(0, dtype=pl.Int64).alias("itemid")))
    vit = pl.concat([vit.filter(~pl.col("var").is_in(["gcs_eye", "gcs_motor", "gcs_verbal"])), gcs],
                    how="diagonal_relaxed")
    tic(f"vitals mapped: {n0} -> {vit.height} rows, vars={sorted(vit['var'].unique().to_list())}")

    # ---- labs
    lcol = pick_col(lab, "valuenum", "value")
    lab = lab.select(["stay_id", "charttime", "itemid", lcol]).rename({lcol: "val"})
    lab = lab.with_columns(pl.col("val").cast(pl.Float64), pl.col("itemid").cast(pl.Int64),
                           pl.col("charttime").cast(pl.Datetime("us")))
    n0 = lab.height
    lab = lab.filter(pl.col("itemid").is_in(list(LAB.keys())) & pl.col("val").is_not_null())
    lab = lab.with_columns(pl.col("itemid").replace_strict(LAB, return_dtype=pl.Utf8).alias("var"))
    tic(f"labs mapped: {n0} -> {lab.height} rows, vars={len(lab['var'].unique())}")

    # ---- pre-window global filter (dt in [-24, 6)) once, to keep memory sane
    vit = windowed(vit, stays, 6.0)
    lab = windowed(lab, stays, 6.0)
    tic(f"windowed: vitals {vit.height}, labs {lab.height}")

    # ---- vasopressors (active at t) and ventilation
    iv = inp.select(["stay_id", "starttime", "endtime", "itemid", "rate"]).with_columns(
        pl.col("itemid").cast(pl.Int64), pl.col("rate").cast(pl.Float64),
        pl.col("starttime").cast(pl.Datetime("us")), pl.col("endtime").cast(pl.Datetime("us")))
    iv = iv.filter(pl.col("itemid").is_in(list(VASO.keys())))
    pv_ = prc.select(["stay_id", "starttime", "endtime", "itemid"]).with_columns(
        pl.col("itemid").cast(pl.Int64),
        pl.col("starttime").cast(pl.Datetime("us")), pl.col("endtime").cast(pl.Datetime("us")))
    pv_ = pv_.filter(pl.col("itemid").is_in(list(VENT_IDS)))
    uro = uro.select(["stay_id", "charttime", "itemid", "value"]).with_columns(
        pl.col("value").cast(pl.Float64), pl.col("charttime").cast(pl.Datetime("us")))

    frames = []
    for h in hours:
        t = time.time()
        dv = vit.filter(pl.col("dt") < float(h))
        dl = lab.filter(pl.col("dt") < float(h))

        fv = stats_long(dv, VIT, "v_", h, VIT_STATS)
        fl = stats_long(dl, LAB, "l_", h, LAB_STATS)
        ftr = trend_long(dv.select(["stay_id", "dt", "var", "val"]).vstack(
            dl.filter(pl.col("var").is_in(TREND_VARS)).select(["stay_id", "dt", "var", "val"])), h)
        base = pl.DataFrame({"stay_id": stays["stay_id"]})
        f = base.join(fv, on="stay_id", how="left").join(fl, on="stay_id", how="left") \
                .join(ftr, on="stay_id", how="left")

        # vaso / vent / urine at this hour
        ivh = iv.join(stays.select(["stay_id", "intime"]), on="stay_id", how="inner")
        ivh = ivh.with_columns(((pl.col("starttime") - pl.col("intime")).dt.total_seconds() / 3600).alias("st"))
        ivh = ivh.filter(pl.col("st") < float(h))
        vaso = ivh.group_by("stay_id").agg([
            pl.col("starttime").len().alias("vaso_any"),
            pl.col("rate").max().alias("vaso_rate_max"),
            pl.col("rate").sort_by("starttime").last().alias("vaso_rate_last"),
            (pl.col("endtime") > pl.col("intime") + pl.duration(hours=h)).sum().alias("vaso_active"),
        ])
        vaso = vaso.with_columns((pl.col("vaso_active") > 0).cast(pl.Int8).alias("vaso_active"))

        pvh = pv_.join(stays.select(["stay_id", "intime"]), on="stay_id", how="inner")
        pvh = pvh.with_columns(((pl.col("starttime") - pl.col("intime")).dt.total_seconds() / 3600).alias("st"))
        pvh = pvh.filter(pl.col("st") < float(h))
        vent = pvh.group_by("stay_id").agg(
            (pl.col("endtime") > pl.col("intime") + pl.duration(hours=h)).sum().alias("vent_active"))
        vent = vent.with_columns((pl.col("vent_active") > 0).cast(pl.Int8).alias("vent_active"))

        uh = windowed(uro, stays, h)
        uo = uh.group_by("stay_id").agg(pl.col("value").sum().alias("uo_24h"),
                                        pl.col("value").len().alias("uo_n"))

        f = f.join(vaso, on="stay_id", how="left").join(vent, on="stay_id", how="left") \
             .join(uo, on="stay_id", how="left")
        f = f.with_columns(pl.lit(h, dtype=pl.Int32).alias("hour_index"))

        sc = sofa_parts(f)
        rs = rule_scores(f)
        f = f.join(sc, on="stay_id", how="left").join(rs, on="stay_id", how="left")

        # urine rate needs weight (fallback 80 kg when no weight charted)
        if "v_weight_last" in f.columns:
            f = f.with_columns(pl.col("v_weight_last").fill_null(80.0).clip(30, 250).alias("_wt"))
        else:
            f = f.with_columns(pl.lit(80.0).alias("_wt"))
        f = f.with_columns((pl.col("uo_24h") / (pl.col("_wt") * 24.0)).alias("uo_rate")).drop("_wt")

        frames.append(f)
        tic(f"hour {h}: {f.height} x {f.width} ({time.time()-t:.1f}s)")

    feat = pl.concat(frames, how="diagonal_relaxed")

    # ---- derived ratios (only when both legs exist)
    def safe_ratio(df, num, den, out):
        if num in df.columns and den in df.columns:
            return df.with_columns((pl.col(num) / pl.col(den)).alias(out))
        return df

    feat = safe_ratio(feat, "v_hr_last", "v_sbp_last", "shock_index")
    feat = safe_ratio(feat, "v_hr_max", "v_hr_min", "hr_ratio_max_min")
    for a, b, out in (("v_hr_max", "v_hr_min", "hr_range"),
                      ("v_sbp_max", "v_sbp_min", "sbp_range"),
                      ("v_temp_max", "v_temp_min", "temp_range")):
        if a in feat.columns and b in feat.columns:
            feat = feat.with_columns((pl.col(a) - pl.col(b)).alias(out))
    gcs_parts = ["v_gcs_eye_last", "v_gcs_motor_last", "v_gcs_verbal_last"]
    if all(c in feat.columns for c in gcs_parts):
        feat = feat.with_columns(
            sum([pl.col(c) for c in gcs_parts]).alias("gcs_parts_sum"))

    # observation density
    cnt_cols = [c for c in feat.columns if c.endswith("_count") and (c.startswith("v_") or c.startswith("l_"))]
    if cnt_cols:
        feat = feat.with_columns(
            pl.sum_horizontal([pl.col(c).fill_null(0) for c in cnt_cols]).alias("n_obs_total"))

    # ---- static
    st = stays.select(["stay_id", "anchor_age", "male", "race_white", "race_black", "race_other",
                       "pre_adm_hours", "first_careunit", "admission_type", "insurance"])
    st = st.with_columns([
        (pl.col("first_careunit").str.contains("MICU")).cast(pl.Int8).alias("careunit_micu"),
        (pl.col("first_careunit").str.contains("SICU")).cast(pl.Int8).alias("careunit_sicu"),
        (pl.col("first_careunit").str.contains("CCU") | pl.col("first_careunit").str.contains("CVICU")
         ).cast(pl.Int8).alias("careunit_ccu"),
        (pl.col("admission_type").is_in(["EW EMER.", "DIRECT EMER."])).cast(pl.Int8).alias("admit_emergency"),
        (pl.col("admission_type") == "URGENT").cast(pl.Int8).alias("admit_urgent"),
        (pl.col("admission_type").is_in(["ELECTIVE", "SURGICAL SAME DAY ADMISSION"])
         ).cast(pl.Int8).alias("admit_elective"),
        (pl.col("insurance") == "Medicare").cast(pl.Int8).alias("ins_medicare"),
        (pl.col("insurance") == "Medicaid").cast(pl.Int8).alias("ins_medicaid"),
        (pl.col("insurance") == "Private").cast(pl.Int8).alias("ins_private"),
    ]).drop(["first_careunit", "admission_type", "insurance"])
    feat = feat.join(st, on="stay_id", how="left").join(stat, on="stay_id", how="left")

    # ---- final schema hygiene: drop constant/all-null columns, cast float32
    drop = []
    for c in feat.columns:
        if c in ("stay_id", "hour_index"):
            continue
        if feat[c].null_count() == feat.height or feat[c].n_unique() <= 1:
            drop.append(c)
    if drop:
        tic(f"dropping {len(drop)} constant/all-null cols: {drop[:12]}{'...' if len(drop)>12 else ''}")
    feat = feat.drop(drop)
    numcols = [c for c in feat.columns if c not in ("stay_id", "hour_index")]
    feat = feat.with_columns([pl.col(c).cast(pl.Float32) for c in numcols])
    # inf (from guarded divisions) -> null, so models see "missing", not +-inf
    feat = feat.with_columns([
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
        for c in numcols])
    feat = feat.with_columns(pl.col("stay_id").cast(pl.Int64), pl.col("hour_index").cast(pl.Int32))
    feat = feat.sort(["stay_id", "hour_index"])
    feat.write_parquet(CACHE / "feat_new.parquet")
    meta = {"rows": feat.height, "cols": feat.width,
            "features": [c for c in feat.columns if c not in ("stay_id", "hour_index")],
            "hours": hours, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_dropped": len(drop)}
    (CACHE / "feat_new_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    tic(f"feat_new.parquet {feat.shape} ({time.time()-t0:.1f}s)")
    print("feature groups:",
          {p: sum(1 for c in meta["features"] if c.startswith(p)) for p in ("v_", "l_", "tr_", "sofa_", "hx_")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", default="0,1,2,4,6")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--out", default=None, help="override output filename (smoke runs)")
    a = ap.parse_args()
    build([int(x) for x in a.hours.split(",")], allow_missing=a.allow_missing)
