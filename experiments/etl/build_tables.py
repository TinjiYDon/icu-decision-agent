#!/usr/bin/env python
"""共享实验数据层构建（PROTOCOL.md 第2节 = SSOT）。

用法
----
    # 分步构建（推荐）
    python build_tables.py --only stays,cohort
    python build_tables.py --only vitals,labs
    python build_tables.py --only inputevents,procedureevents,uroutput,micro,transfers
    python build_tables.py --only sofa
    python build_tables.py --only qc

    # 一次性全量
    python build_tables.py --only all

    # 小样本链路验证（5,000 stay）
    python build_tables.py --only all --limit-stays 5000

产物：experiments/data/{name}.parquet + {name}.csv，以及 qc.json / logs/*.log

口径提醒
--------
* 时间窗：charttime in [intime-24h, intime+72h] 且 charttime < outtime（PROTOCOL 第2节）。
* stays 按 stay_id 去重至 94,458 行；cohort 直接复用 icu_decision.paper_label.*，不重算标签。
* 体征只能从 CSV chartevents 读（DB 里 chartevents 为 0 行）；化验从 DB 读（158M 行）。
* 所有随机性种子固定 SEED=42。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------- 路径与常量
WORK = Path(__file__).resolve().parents[2]
CSVROOT = Path(__import__("os").environ["MIMIC_CSV_ROOT"])
OUT = WORK / "experiments" / "data"
LOGD = WORK / "experiments" / "etl" / "logs"

DSN_MIMIC = __import__("os").environ["MIMIC_DSN"]
DSN_DECISION = __import__("os").environ["ICU_DECISION_DSN"]

SEED = 42
H_PRE = 24     # charttime >= intime - 24h
H_POST = 72    # charttime <= intime + 72h
HMAX = 71      # sofa hour_index 0..71
GRID_H = (-H_PRE, HMAX)

# 需要同时产出 csv 的小表（调度侧解释器无 pyarrow，只能读 csv）
CSV_TABLES = {"stays", "cohort", "micro", "transfers", "sofa_hourly"}
OPTS = {"csv_big": False}

# ---------------------------------------------------------------- itemid 白名单
VITAL_ITEMS = {
    220045: "Heart Rate",
    220210: "Respiratory Rate",
    224690: "Respiratory Rate (Total)",
    220277: "O2 saturation pulseoxymetry",
    223762: "Temperature Celsius",
    220050: "Arterial Blood Pressure systolic",
    220051: "Arterial Blood Pressure diastolic",
    220052: "Arterial Blood Pressure mean",
    220179: "Non Invasive Blood Pressure systolic",
    220180: "Non Invasive Blood Pressure diastolic",
    220181: "Non Invasive Blood Pressure mean",
    220739: "GCS - Eye Opening",
    223900: "GCS - Verbal Response",
    223901: "GCS - Motor Response",
    223835: "Inspired O2 Fraction",
    220339: "PEEP set",
    223849: "Ventilator Mode",
    226732: "O2 Delivery Device(s)",
    226512: "Admission Weight (Kg)",
    224639: "Daily Weight",
}

LAB_ITEMS = {
    50813: "Lactate", 52442: "Lactate", 53154: "Lactate",
    50912: "Creatinine", 52546: "Creatinine",
    51006: "Urea Nitrogen", 52647: "Urea Nitrogen",
    50971: "Potassium", 52610: "Potassium",
    50983: "Sodium", 52623: "Sodium",
    50882: "Bicarbonate",
    50885: "Bilirubin, Total", 53089: "Bilirubin, Total",
    51265: "Platelet Count", 53189: "Platelet Count",
    51301: "White Blood Cells", 51755: "White Blood Cells",
    51222: "Hemoglobin", 51221: "Hematocrit", 50931: "Glucose",
    50821: "pO2", 50818: "pCO2", 50820: "pH",
    50902: "Chloride", 50960: "Magnesium", 50893: "Calcium, Total",
    51237: "INR", 51275: "PT", 50862: "Albumin",
    50878: "AST", 50861: "ALT", 50863: "Alkaline Phosphatase",
    51002: "Troponin T", 50963: "NTproBNP",
}

VASO_ITEMS = {
    221906: "Norepinephrine",
    221289: "Epinephrine",
    221662: "Dopamine",
    221653: "Dobutamine",
    222315: "Vasopressin",
    221749: "Phenylephrine",
}

PROC_ITEMS = {
    225792: "Invasive Ventilation",
    225794: "Non-invasive Ventilation",
    225802: "Dialysis - CRRT",
    225803: "Dialysis - CVVHD",
    225805: "Dialysis - Peritoneal",
    225809: "Dialysis - CVVHDF",
    225441: "Hemodialysis",
    225203: "CVC Placement",
    225204: "Arterial Line Placement",
}

UO_ITEMS = {
    226559: "Foley", 226560: "Void", 226561: "Condom Cath",
    226584: "Ileoconduit", 226563: "Suprapubic", 226564: "R Nephrostomy",
    226565: "L Nephrostomy", 226567: "Straight Cath", 226557: "R Ureteral Stent",
    226558: "L Ureteral Stent", 227488: "GU Irrigant Volume In",
    227489: "GU Irrigant/Urine Volume Out",
}
UO_NEGATIVE = {227488}   # 冲洗液入量：计算尿量时剔除

# ---------------------------------------------------------------- 日志
LOGGER = logging.getLogger("etl")


def setup_logging(tag: str = "build") -> Path:
    LOGD.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    p = LOGD / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(p, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOGGER.addHandler(fh)
    LOGGER.addHandler(sh)
    return p


def log(msg: str, *a) -> None:
    LOGGER.info(msg, *a)


# ---------------------------------------------------------------- QC 容器
class QC:
    def __init__(self, path: Path, meta: dict):
        self.path = path
        self.data = {"meta": meta, "tables": {}, "audit": [], "itemid_hits": {},
                     "sofa": {}, "notes": [], "join_audit": []}
        if path.exists():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                for k in ("tables", "itemid_hits", "sofa", "audit", "notes",
                          "join_audit", "cohort"):
                    if k in old:
                        self.data[k] = old[k]
            except Exception:  # noqa: BLE001
                pass

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2),
                             encoding="utf-8")


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_table(df: pl.DataFrame, name: str, qc: QC, csv_cols: list[str] | None = None,
                time_col: str | None = None, nonnull_cols: list[str] | None = None,
                sort_by: list[str] | None = None, max_sort_rows: int = 30_000_000) -> pl.DataFrame:
    """写 parquet + csv，登记行数/时间范围/非空率/sha256。

    csv 策略（lead 2026-11-13 决定）：只给下游必须用 csv 的解释器（调度侧，无 pyarrow）导出
    小表 —— stays/cohort/micro/transfers/sofa_hourly；vitals/labs 等大表只留 parquet。
    `--csv-big` 可强制给所有表导出 csv（按需重建）。
    """
    t0 = time.time()
    if sort_by and df.height > max_sort_rows:
        log("  %s 行数 %d > %d，跳过排序（内存保护）", name, df.height, max_sort_rows)
        sort_by = None
    if sort_by:
        df = df.sort(sort_by)
    pq = OUT / f"{name}.parquet"
    df.write_parquet(pq, compression="zstd")
    t_pq = time.time() - t0

    want_csv = OPTS["csv_big"] or name in CSV_TABLES
    entry: dict = {}
    t_csv = 0.0
    cs = OUT / f"{name}.csv"
    if want_csv:
        csv_df = df.select(csv_cols) if csv_cols else df
        t1 = time.time()
        csv_df.write_csv(cs)
        t_csv = time.time() - t1
        entry["csv"] = {"path": str(cs.relative_to(WORK)), "bytes": cs.stat().st_size,
                        "sha256": sha256(cs), "cols": csv_df.columns}
    else:
        if cs.exists():
            cs.unlink()
        entry["csv"] = None
        entry["csv_note"] = "大表不导出 csv（lead 决定）：下游读 parquet；--csv-big 可按需导出"

    entry.update({
        "rows": df.height,
        "cols": df.columns,
        "elapsed_s": round(time.time() - t0, 2),
        "elapsed_parquet_s": round(t_pq, 2),
        "elapsed_csv_s": round(t_csv, 2),
        "parquet": {"path": str(pq.relative_to(WORK)), "bytes": pq.stat().st_size,
                    "sha256": sha256(pq)},
    })
    if time_col and time_col in df.columns:
        entry["time_min"] = str(df[time_col].min())
        entry["time_max"] = str(df[time_col].max())
    if nonnull_cols:
        n = df.height or 1
        entry["nonnull_rate"] = {
            c: round(float(df[c].is_not_null().sum()) / n, 6)
            for c in nonnull_cols if c in df.columns
        }
    qc.data["tables"][name] = entry
    log("写出 %-16s rows=%d  parquet=%.1fs/%.1fMB  csv=%.1fs/%s",
        name, df.height, t_pq, entry["parquet"]["bytes"] / 1e6, t_csv,
        (f"{entry['csv']['bytes'] / 1e6:.1f}MB" if entry["csv"] else "略"))
    return df


def audit(qc: QC, table: str, stage: str, rows: int, **kw) -> None:
    rec = {"table": table, "stage": stage, "rows": int(rows)}
    rec.update(kw)
    qc.data["audit"].append(rec)
    log("  [audit] %-14s %-22s rows=%d %s", table, stage, rows,
        " ".join(f"{k}={v}" for k, v in kw.items()))


def audit_window(qc: QC, table: str, before: int, after_join: int, after_window: int) -> None:
    qc.data["join_audit"].append(
        {"table": table, "input_rows": before, "after_join": after_join,
         "after_window": after_window, "amplified": after_join > before})
    log("  [join] %-14s input=%d  join后=%d  window后=%d %s", table, before,
        after_join, after_window, "<-- 放大!" if after_join > before else "")


# ---------------------------------------------------------------- PG 工具
def pg_copy(dsn: str, sql: str) -> pl.DataFrame:
    import psycopg
    buf = io.BytesIO()
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SET work_mem='512MB'")
        with cur.copy(sql) as cp:
            for chunk in cp:
                buf.write(chunk)
    raw = buf.getvalue()
    log("  pg_copy 取回 %.1f MB", len(raw) / 1e6)
    return pl.read_csv(io.BytesIO(raw), try_parse_dates=True, infer_schema_length=10000)


# ---- 小表 CSV 读取：MIMIC 的数值列存在混合类型（如 totalamount 里混浮点字符串），
# 统一按字符串读入再显式 cast，避免 schema 推断踩坑。
def read_csv_str(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=0, encoding="utf8-lossy",
                       try_parse_dates=False)


def cast_cols(df: pl.DataFrame, int_cols=(), dbl_cols=(), dt_cols=()) -> pl.DataFrame:
    exprs = [pl.col(c).cast(pl.Int64, strict=False) for c in int_cols]
    exprs += [pl.col(c).cast(pl.Float64, strict=False) for c in dbl_cols]
    exprs += [pl.col(c).str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False) for c in dt_cols]
    return df.with_columns(exprs) if exprs else df


# ---------------------------------------------------------------- stays
def _dt_from_us(expr: pl.Expr) -> pl.Expr:
    """微秒整数 -> Datetime（polars 2.0 里 Duration 关键字受限，用 epoch 换算）。"""
    return expr.cast(pl.Datetime("us"))


STAY_COLS = ["stay_id", "subject_id", "hadm_id", "intime", "outtime", "los_hours",             "first_careunit", "last_careunit", "admission_type", "insurance", "race",
             "anchor_age", "gender", "hospital_expire_flag", "dod", "deathtime",
             "dischtime", "admission_location", "discharge_location"]


def load_stays(limit: int | None = None) -> tuple[pl.DataFrame, dict]:
    t0 = time.time()
    ic = pl.read_csv(CSVROOT / "icu" / "icustays.csv.gz", try_parse_dates=True)
    raw_rows = ic.height
    ic = ic.unique(subset=["stay_id"], keep="first")
    log("icustays: 原始 %d 行 -> 去重 %d 行 (%.1fs)", raw_rows, ic.height, time.time() - t0)

    ad = pl.read_csv(CSVROOT / "hosp" / "admissions.csv.gz", try_parse_dates=True) \
        .unique(subset=["hadm_id"], keep="first")
    pt = pl.read_csv(CSVROOT / "hosp" / "patients.csv.gz", try_parse_dates=True) \
        .unique(subset=["subject_id"], keep="first")

    df = (ic.select(["stay_id", "subject_id", "hadm_id", "first_careunit", "last_careunit",
                     "intime", "outtime", "los"])
            .join(ad.select(["hadm_id", "admission_type", "insurance", "race",
                             "hospital_expire_flag", "dischtime", "deathtime",
                             "admission_location", "discharge_location"]),
                  on="hadm_id", how="left")
            .join(pt.select(["subject_id", "gender", "anchor_age", "dod"]),
                  on="subject_id", how="left"))
    # outtime 缺失（14 行，los 也为空）用 intime + los 兜底，仍空则用住院 dischtime，
    # 目的是保证 stays 保持 94,458 行（PROTOCOL 验收口径）
    n_null_out = df["outtime"].null_count()
    fallback_ids: list[int] = []
    if n_null_out:
        fallback_ids = sorted(df.filter(pl.col("outtime").is_null())["stay_id"].to_list())
        log("outtime 缺失 %d 行 -> intime+los / dischtime 兜底（qc 中单独标注）", n_null_out)
        df = df.with_columns(
            pl.coalesce([pl.col("outtime"),
                         _dt_from_us(pl.col("intime").dt.epoch("us")
                                     + (pl.col("los") * 86400e6).cast(pl.Int64)),
                         pl.col("dischtime")])
            .alias("outtime"))
    df = df.with_columns(
        ((pl.col("outtime") - pl.col("intime")).dt.total_seconds() / 3600.0)
        .alias("los_hours")).drop("los")

    n_bad = df.filter(pl.col("outtime").is_null() | pl.col("intime").is_null()).height
    if n_bad:
        log("WARN intime/outtime 仍缺失 %d 行，丢弃", n_bad)
        df = df.drop_nulls(["intime", "outtime"])

    df = df.select(STAY_COLS).drop_nulls("intime").sort("stay_id")
    df = df.with_columns([pl.col("stay_id").cast(pl.Int64), pl.col("subject_id").cast(pl.Int64),
                          pl.col("hadm_id").cast(pl.Int64),
                          pl.col("anchor_age").cast(pl.Int32),
                          pl.col("hospital_expire_flag").cast(pl.Int32)])
    if limit:
        rng = np.random.default_rng(SEED)
        sid = df["stay_id"].to_numpy()
        keep = np.sort(rng.choice(sid, size=min(limit, len(sid)), replace=False))
        df = df.filter(pl.col("stay_id").is_in(keep.tolist()))
        log("LIMIT: 抽样 %d stay（seed=%d）", df.height, SEED)
    log("stays 就绪 rows=%d  los_hours 中位数=%.1f  总计 %.1fs",
        df.height, float(df["los_hours"].median() or 0), time.time() - t0)
    return df, {"icustays_raw": raw_rows, "outtime_fallback_ids": fallback_ids}


def build_stays(ctx: dict, qc: QC) -> None:
    df = ctx["stays"]
    write_table(df, "stays", qc, time_col="intime",
                nonnull_cols=["stay_id", "subject_id", "hadm_id", "intime", "outtime",
                              "los_hours", "first_careunit", "admission_type", "race",
                              "anchor_age", "gender", "hospital_expire_flag", "dod"],
                sort_by=["stay_id"])
    audit(qc, "stays", "dedup_and_join", df.height,
          icustays_raw=ctx["stays_meta"]["icustays_raw"])
    qc.data["tables"]["stays"]["los_hours_quantiles"] = {
        f"p{q}": round(float(df["los_hours"].quantile(q / 100) or 0), 2)
        for q in (5, 25, 50, 75, 95)}
    qc.data["tables"]["stays"]["n_subject"] = int(df["subject_id"].n_unique())
    qc.data["tables"]["stays"]["n_hadm"] = int(df["hadm_id"].n_unique())
    qc.data["tables"]["stays"]["careunit_counts"] = {
        str(k): int(v) for k, v in
        df["first_careunit"].value_counts().sort("count", descending=True).iter_rows()}
    fb = ctx["stays_meta"].get("outtime_fallback_ids") or []
    qc.data["tables"]["stays"]["outtime_fallback"] = {
        "n": len(fb),
        "stay_ids": fb,
        "note": "MIMIC-IV 3.1 中这 14 行 icustays 的 outtime 与 los 均为空；为保持 "
                "94,458 行验收口径，outtime 用同一 hadm 的 admissions.dischtime 兜底，"
                "los_hours 相应按 dischtime-intime 计算；报告限制里需单独提一句。",
    }
    qc.save()


# ---------------------------------------------------------------- cohort
def build_cohort(ctx: dict, qc: QC) -> None:
    t0 = time.time()
    qc.data["notes"].append("cohort 标签直接复用 icu_decision.paper_label.*，未重算（§2.7）")
    d1 = pg_copy(DSN_DECISION,
                 "COPY (SELECT stay_id, hour_index, label AS y_death "
                 "FROM paper_label.mortality_12h) TO STDOUT WITH (FORMAT csv, HEADER true)")
    d2 = pg_copy(DSN_DECISION,
                 "COPY (SELECT stay_id, hour_index, label AS y_dod "
                 "FROM paper_label.mortality_12h_dod) TO STDOUT WITH (FORMAT csv, HEADER true)")
    audit(qc, "cohort", "pg_export_y_death", d1.height)
    audit(qc, "cohort", "pg_export_y_dod", d2.height)
    d1 = d1.with_columns([pl.col("stay_id").cast(pl.Int64),
                          pl.col("hour_index").cast(pl.Int32)])
    d2 = d2.with_columns([pl.col("stay_id").cast(pl.Int64),
                          pl.col("hour_index").cast(pl.Int32)])
    df = d1.join(d2, on=["stay_id", "hour_index"], how="inner")
    audit(qc, "cohort", "join_two_labels", df.height, join="inner(stay_id,hour_index)")
    df = df.join(ctx["stays"].select(["stay_id", "subject_id"]), on="stay_id", how="inner")
    audit(qc, "cohort", "filter_to_stays", df.height, n_stays=ctx["stays"].height)
    df = df.select(["stay_id", "subject_id", "hour_index", "y_death", "y_dod"]) \
           .sort(["stay_id", "hour_index"])
    write_table(df, "cohort", qc, nonnull_cols=["stay_id", "subject_id", "hour_index",
                                                "y_death", "y_dod"],
                sort_by=["stay_id", "hour_index"])
    n = df.height
    pos_death = int(df["y_death"].sum())
    pos_dod = int(df["y_dod"].sum())
    qc.data["cohort"] = {
        "rows": n,
        "pos_y_death": pos_death, "rate_y_death": round(pos_death / max(n, 1), 6),
        "pos_y_dod": pos_dod, "rate_y_dod": round(pos_dod / max(n, 1), 6),
        "hour_index_counts": {str(k): int(v) for k, v in
                              df["hour_index"].value_counts().sort("hour_index").iter_rows()},
        "elapsed_s": round(time.time() - t0, 2),
    }
    qc.save()
    log("cohort 阳性率 y_death=%.4f%%  y_dod=%.4f%%", 100 * pos_death / max(n, 1),
        100 * pos_dod / max(n, 1))


# ---------------------------------------------------------------- 时间窗工具
def rel_hour_expr(time_col: str = "charttime") -> pl.Expr:
    return (((pl.col(time_col) - pl.col("intime")).dt.total_seconds() / 3600.0)
            .floor().cast(pl.Int32).alias("rel_hour"))


def window_filter(df: pl.DataFrame, stays: pl.DataFrame, time_col: str = "charttime",
                  stay_col: str = "stay_id", qc: QC | None = None,
                  table: str | None = None) -> pl.DataFrame:
    """join stays 时间窗并过滤；逐表核对 join 前后行数（不得放大）。"""
    before = df.height
    times = stays.select(["stay_id", "intime", "outtime"]).rename({"stay_id": stay_col})
    j = df.join(times, on=stay_col, how="inner")
    after_join = j.height
    j = j.filter(
        (pl.col(time_col) >= pl.col("intime") - pl.duration(hours=H_PRE))
        & (pl.col(time_col) <= pl.col("intime") + pl.duration(hours=H_POST))
        & (pl.col(time_col) < pl.col("outtime"))
    )
    after = j.height
    if qc is not None:
        audit_window(qc, table or "?", before, after_join, after)
    return j.drop(["intime", "outtime"])


# ---------------------------------------------------------------- vitals
def build_vitals(ctx: dict, qc: QC) -> None:
    """chartevents 单遍扫描：先落「全库白名单」中间 parquet（供全库 itemid 命中行数核对，
    满足「每项 >1e7 行级别（全库）」验收口径），再 join stays 做时间窗过滤。"""
    t0 = time.time()
    ids = sorted(VITAL_ITEMS)
    stays = ctx["stays"].select(["stay_id", "intime", "outtime"])
    ce = CSVROOT / "icu" / "chartevents.csv.gz"
    log("chartevents 单遍扫描开始（白名单 %d itemid, low_memory=True）", len(ids))
    lf = pl.scan_csv(ce, low_memory=True, try_parse_dates=True)
    lf = (lf.select(["stay_id", "charttime", "itemid", "valuenum", "valueuom"])
            .with_columns([pl.col("stay_id").cast(pl.Int64),
                           pl.col("itemid").cast(pl.Int64),
                           pl.col("valuenum").cast(pl.Float64)])
            .filter(pl.col("itemid").is_in(ids)))
    tmp_all = OUT / "_ce_whitelist.parquet"
    t1 = time.time()
    lf.sink_parquet(tmp_all, compression="zstd", engine="streaming")
    log("chartevents 白名单全库中间件写出 %.1fs（%.1fMB）", time.time() - t1,
        tmp_all.stat().st_size / 1e6)

    # 全库（未做时间窗）各 itemid 命中行数：验收要求的「>1e7 行级别（全库）」
    full_hits = (pl.scan_parquet(tmp_all).group_by("itemid").len()
                 .collect(engine="streaming").sort("itemid"))
    full_map = {int(i): int(c) for i, c in full_hits.iter_rows()}
    qc.data["itemid_hits"]["vitals_full_library"] = {
        str(i): {"label": VITAL_ITEMS.get(i), "rows": full_map.get(i, 0)} for i in ids}
    log("全库白名单命中 %d 行；%s", sum(full_map.values()),
        ", ".join(f"{VITAL_ITEMS[i]}={full_map.get(i, 0)}" for i in [220045, 220210,
                                                                    220277, 220050]))

    # 时间窗过滤（保留 intime-24h 起的入科前记录）
    lf2 = (pl.scan_parquet(tmp_all)
             .join(stays.lazy(), on="stay_id", how="inner")
             .filter((pl.col("charttime") >= pl.col("intime") - pl.duration(hours=H_PRE))
                     & (pl.col("charttime") <= pl.col("intime") + pl.duration(hours=H_POST))
                     & (pl.col("charttime") < pl.col("outtime")))
             .drop(["intime", "outtime"])
             .select(["stay_id", "charttime", "itemid", "valuenum", "valueuom"]))
    tmp = OUT / "_vitals_tmp.parquet"
    t2 = time.time()
    lf2.sink_parquet(tmp, compression="zstd", engine="streaming")
    log("vitals 时间窗过滤写出 %.1fs", time.time() - t2)
    tmp_all.unlink()

    df = pl.read_parquet(tmp)
    tmp.unlink()
    d_items = pl.read_csv(CSVROOT / "icu" / "d_items.csv.gz", infer_schema_length=0) \
        .select([pl.col("itemid").cast(pl.Int64), "label"]).unique("itemid", keep="first")
    df = df.join(d_items, on="itemid", how="left") \
           .select(["stay_id", "charttime", "itemid", "label", "valuenum", "valueuom"])
    audit(qc, "vitals", "after_window_and_label", df.height)

    # 单位不做就地换算（℃/℉、lbs/kg 由特征层统一处理），仅保留 valueuom 标记
    write_table(df, "vitals", qc, csv_cols=["stay_id", "charttime", "itemid", "label",
                                            "valuenum", "valueuom"],
                time_col="charttime", nonnull_cols=["valuenum", "valueuom"],
                sort_by=["stay_id", "charttime", "itemid"])
    # 注意：write_table 会整体覆盖 tables[name]，因此派生统计必须放在其后
    qc.data["tables"]["vitals"]["uom_counts"] = {
        str(u): int(n) for u, n in
        df.group_by("valueuom").agg(pl.len().alias("n")).sort("n", descending=True).iter_rows()
        if u is not None}
    qc.data["tables"]["vitals"]["unit_note"] = (
        "valueuom 原样保留（含 °F / lbs 等），未做单位换算；温度换算由特征层统一处理")

    hits = df.group_by("itemid").agg(pl.len().alias("rows")).sort("rows", descending=True)
    lab = dict(df.group_by("itemid").agg(pl.col("label").first()).iter_rows())
    qc.data["itemid_hits"]["vitals"] = {
        str(i): {"label": lab.get(i), "rows": int(r)} for i, r in hits.iter_rows()}
    cov = df.group_by("stay_id").agg(pl.col("itemid").n_unique().alias("n_items"))
    qc.data["tables"]["vitals"]["per_stay"] = {
        "stays_with_any": int(cov.height),
        "stay_coverage": round(cov.height / max(ctx["stays"].height, 1), 4),
        "median_items_per_stay": float(cov["n_items"].median() or 0)}
    qc.data["tables"]["vitals"]["key_items"] = {}
    for iid, name in [(220045, "HR"), (220210, "RR"), (220277, "SpO2"), (220050, "SBP"),
                      (220052, "MAP_invasive"), (220181, "MAP_cuff"),
                      (220739, "GCS_eye"), (223900, "GCS_verbal"), (223901, "GCS_motor"),
                      (223835, "FiO2"), (226512, "weight")]:
        sub = df.filter(pl.col("itemid") == iid)
        n = int(sub.height)
        n_stay = int(sub["stay_id"].n_unique())
        qc.data["tables"]["vitals"]["key_items"][name] = {
            "itemid": iid, "rows": n,
            "stay_coverage": round(n_stay / max(ctx["stays"].height, 1), 4)}
        log("  vitals key %-12s itemid=%-6d rows=%-10d stay_coverage=%.3f", name, iid, n,
            n_stay / max(ctx["stays"].height, 1))
    qc.data["tables"]["vitals"]["elapsed_s"] = round(time.time() - t0, 2)
    qc.save()


# ---------------------------------------------------------------- labs
def build_labs(ctx: dict, qc: QC) -> None:
    """labevents 无 stay_id、DB 无 itemid 索引（158M 行）：把 stays 灌进 temp table，
    在 PG 侧做 hadm+时间窗 hash join，用 DISTINCT ON (labevent_id) 保证不放大，
    COPY 落临时 CSV 再流式写 parquet（避免把 40M 行全塞进内存）。"""
    import psycopg

    t0 = time.time()
    ids = sorted(LAB_ITEMS)
    idlist = ",".join(str(i) for i in ids)
    stays = ctx["stays"].select(["stay_id", "hadm_id", "intime", "outtime"])
    raw_csv = OUT / "_labs_raw.csv"

    sql = f"""
    COPY (
      SELECT DISTINCT ON (l.labevent_id)
             s.stay_id, l.charttime, l.itemid, l.valuenum, l.valueuom,
             count(*) OVER (PARTITION BY l.labevent_id) AS n_match
      FROM mimiciv_hosp.labevents l
      JOIN _etl_stays s
        ON s.hadm_id = l.hadm_id
       AND l.charttime >= s.intime - interval '{H_PRE} hours'
       AND l.charttime <= s.intime + interval '{H_POST} hours'
       AND l.charttime <  s.outtime
      WHERE l.itemid IN ({idlist})
      ORDER BY l.labevent_id
    ) TO STDOUT WITH (FORMAT csv, HEADER true)
    """
    with psycopg.connect(DSN_MIMIC) as conn, conn.cursor() as cur:
        cur.execute("SET work_mem='768MB'")
        cur.execute("CREATE TEMP TABLE _etl_stays (stay_id bigint, hadm_id bigint, "
                    "intime timestamp, outtime timestamp)")
        with cur.copy("COPY _etl_stays FROM STDIN") as cp:   # text 格式，write_row 用 tab 分隔
            for row in stays.iter_rows():
                cp.write_row(row)
        cur.execute("CREATE INDEX ON _etl_stays(hadm_id)")
        cur.execute("ANALYZE _etl_stays")
        log("labs: temp stays %d 行就绪，开始 PG 侧 join+window（DISTINCT ON labevent_id）",
            stays.height)
        t1 = time.time()
        with cur.copy(sql) as cp, raw_csv.open("wb") as f:
            while True:
                chunk = cp.read()
                if not chunk:
                    break
                f.write(chunk)
        log("labs: PG join/window/COPY 完成 %.1fs，临时 CSV %.1fMB", time.time() - t1,
            raw_csv.stat().st_size / 1e6)
        conn.rollback()

    lf = (pl.scan_csv(raw_csv, try_parse_dates=True)
            .select(["stay_id", "charttime", "itemid", "valuenum", "valueuom", "n_match"])
            .with_columns([pl.col("stay_id").cast(pl.Int64),
                           pl.col("itemid").cast(pl.Int64),
                           pl.col("valuenum").cast(pl.Float64)]))
    tmp = OUT / "_labs_tmp.parquet"
    t2 = time.time()
    lf.sink_parquet(tmp, compression="zstd", engine="streaming")
    log("labs: 中间 parquet 写出 %.1fs", time.time() - t2)
    raw_csv.unlink()

    # join 审计：n_match 是「同一 labevent 命中几个 stay 窗口」
    nm = pl.read_parquet(tmp, columns=["n_match"])
    after_join = int(nm["n_match"].sum())
    after_win = int(nm.height)
    del nm
    audit_window(qc, "labs", after_join, after_join, after_win)
    qc.data["join_audit"][-1]["note"] = (
        "input=命中白名单且 hadm 在 stays 内的 (lab,stay) 对；after=按 labevent_id "
        "去重后行数（DISTINCT ON），放大倍数=%.4f" % (after_join / max(after_win, 1)))
    qc.data["join_audit"][-1]["dup_factor"] = round(after_join / max(after_win, 1), 6)

    df = pl.read_parquet(tmp, columns=["stay_id", "charttime", "itemid", "valuenum",
                                       "valueuom"])
    tmp.unlink()
    d_lab = pl.read_csv(CSVROOT / "hosp" / "d_labitems.csv.gz", infer_schema_length=0) \
        .select([pl.col("itemid").cast(pl.Int64), "label"]).unique("itemid", keep="first")
    df = df.join(d_lab, on="itemid", how="left") \
           .select(["stay_id", "charttime", "itemid", "label", "valuenum", "valueuom"])
    audit(qc, "labs", "after_window_and_label", df.height)

    write_table(df, "labs", qc, csv_cols=["stay_id", "charttime", "itemid", "label",
                                          "valuenum", "valueuom"],
                time_col="charttime", nonnull_cols=["valuenum", "valueuom"],
                sort_by=["stay_id", "charttime", "itemid"])
    hits = df.group_by("itemid").agg(pl.len().alias("rows")).sort("rows", descending=True)
    lab = dict(df.group_by("itemid").agg(pl.col("label").first()).iter_rows())
    qc.data["itemid_hits"]["labs"] = {
        str(i): {"label": lab.get(i), "rows": int(r)} for i, r in hits.iter_rows()}
    qc.data["itemid_hits"]["labs_missing_itemids"] = [
        i for i in ids if i not in set(hits["itemid"].to_list())]
    cov = df.group_by("stay_id").agg(pl.col("itemid").n_unique().alias("n_items"))
    qc.data["tables"]["labs"]["per_stay"] = {
        "stays_with_any": int(cov.height),
        "stay_coverage": round(cov.height / max(ctx["stays"].height, 1), 4),
        "median_items_per_stay": float(cov["n_items"].median() or 0)}
    qc.data["tables"]["labs"]["elapsed_s"] = round(time.time() - t0, 2)
    log("labs 完成 rows=%d 用时 %.1fs", df.height, time.time() - t0)
    qc.save()


# ---------------------------------------------------------------- inputevents
def build_inputevents(ctx: dict, qc: QC) -> None:
    t0 = time.time()
    df = read_csv_str(CSVROOT / "icu" / "inputevents.csv.gz")
    df = cast_cols(df, int_cols=["stay_id", "itemid"],
                   dbl_cols=["amount", "rate", "patientweight"],
                   dt_cols=["starttime", "endtime"])
    raw = df.filter(pl.col("itemid").is_in(sorted(VASO_ITEMS)))
    audit(qc, "inputevents", "itemid_whitelist", raw.height)
    raw = raw.select(["stay_id", "starttime", "endtime", "itemid", "amount", "amountuom",
                      "rate", "rateuom", "patientweight"])
    w = window_filter(raw, ctx["stays"], time_col="starttime", qc=qc, table="inputevents")
    audit(qc, "inputevents", "after_window", w.height)
    d_items = pl.read_csv(CSVROOT / "icu" / "d_items.csv.gz", infer_schema_length=0) \
        .select([pl.col("itemid").cast(pl.Int64), "label"]).unique("itemid", keep="first")
    out = w.join(d_items, on="itemid", how="left") \
           .select(["stay_id", "starttime", "endtime", "itemid", "label", "rate", "rateuom",
                    "amount", "amountuom"])
    before = out.height
    out = out.sort(["stay_id", "starttime", "itemid"]).unique(
        subset=["stay_id", "starttime", "itemid", "rate", "amount"], keep="first",
        maintain_order=True)
    audit(qc, "inputevents", "dedup", out.height, dropped=before - out.height)
    write_table(out, "inputevents", qc, time_col="starttime",
                nonnull_cols=["rate", "rateuom", "amount", "amountuom"],
                sort_by=["stay_id", "starttime"])
    u = out.group_by("rateuom").agg(pl.len().alias("rows")).sort("rows", descending=True)
    qc.data["itemid_hits"]["inputevents"] = {
        str(i): int(n) for i, n in out.group_by("itemid").len().sort("itemid").iter_rows()}
    qc.data["tables"]["inputevents"]["rateuom"] = {str(a): int(b) for a, b in u.iter_rows()}
    qc.data["tables"]["inputevents"]["elapsed_s"] = round(time.time() - t0, 2)
    qc.save()


# ---------------------------------------------------------------- procedureevents
def build_procedureevents(ctx: dict, qc: QC) -> None:
    t0 = time.time()
    df = read_csv_str(CSVROOT / "icu" / "procedureevents.csv.gz")
    df = cast_cols(df, int_cols=["stay_id", "itemid"], dbl_cols=["value"],
                   dt_cols=["starttime", "endtime"])
    raw = df.filter(pl.col("itemid").is_in(sorted(PROC_ITEMS)))
    audit(qc, "procedureevents", "itemid_whitelist", raw.height)
    raw = raw.select(["stay_id", "starttime", "endtime", "itemid", "value", "valueuom"])
    w = window_filter(raw, ctx["stays"], time_col="starttime", qc=qc,
                      table="procedureevents")
    d_items = pl.read_csv(CSVROOT / "icu" / "d_items.csv.gz", infer_schema_length=0) \
        .select([pl.col("itemid").cast(pl.Int64), "label"]).unique("itemid", keep="first")
    out = w.join(d_items, on="itemid", how="left") \
           .select(["stay_id", "starttime", "endtime", "itemid", "label", "value", "valueuom"])
    before = out.height
    out = out.sort(["stay_id", "starttime", "itemid"]).unique(
        subset=["stay_id", "starttime", "endtime", "itemid"], keep="first",
        maintain_order=True)
    audit(qc, "procedureevents", "dedup", out.height, dropped=before - out.height)
    write_table(out, "procedureevents", qc, time_col="starttime",
                nonnull_cols=["value", "valueuom"], sort_by=["stay_id", "starttime"])
    qc.data["itemid_hits"]["procedureevents"] = {
        str(i): int(n) for i, n in out.group_by("itemid").len().sort("itemid").iter_rows()}
    qc.data["tables"]["procedureevents"]["elapsed_s"] = round(time.time() - t0, 2)
    qc.save()


# ---------------------------------------------------------------- uroutput
def build_uroutput(ctx: dict, qc: QC) -> None:
    t0 = time.time()
    df = read_csv_str(CSVROOT / "icu" / "outputevents.csv.gz")
    df = cast_cols(df, int_cols=["stay_id", "itemid"], dbl_cols=["value"],
                   dt_cols=["charttime"])
    raw = df.filter(pl.col("itemid").is_in(sorted(UO_ITEMS)))
    audit(qc, "uroutput", "itemid_whitelist", raw.height)
    raw = raw.select(["stay_id", "charttime", "itemid", "value"])
    w = window_filter(raw, ctx["stays"], time_col="charttime", qc=qc, table="uroutput")
    d_items = pl.read_csv(CSVROOT / "icu" / "d_items.csv.gz", infer_schema_length=0) \
        .select([pl.col("itemid").cast(pl.Int64), "label"]).unique("itemid", keep="first")
    out = w.join(d_items, on="itemid", how="left") \
           .select(["stay_id", "charttime", "itemid", "label", "value"])
    before = out.height
    out = out.sort(["stay_id", "charttime", "itemid"]).unique(
        subset=["stay_id", "charttime", "itemid", "value"], keep="first", maintain_order=True)
    audit(qc, "uroutput", "dedup", out.height, dropped=before - out.height)
    write_table(out, "uroutput", qc, time_col="charttime", nonnull_cols=["value"],
                sort_by=["stay_id", "charttime"])
    qc.data["itemid_hits"]["uroutput"] = {
        str(i): int(n) for i, n in out.group_by("itemid").len().sort("itemid").iter_rows()}
    qc.data["tables"]["uroutput"]["elapsed_s"] = round(time.time() - t0, 2)
    qc.save()


# ---------------------------------------------------------------- micro
def build_micro(ctx: dict, qc: QC) -> None:
    t0 = time.time()
    df = read_csv_str(CSVROOT / "hosp" / "microbiologyevents.csv.gz")
    df = cast_cols(df, int_cols=["subject_id", "hadm_id", "ab_itemid"],
                   dt_cols=["charttime", "chartdate"])
    hadm = ctx["stays"].select("hadm_id").unique().with_columns(pl.col("hadm_id").cast(pl.Int64))
    raw = df.filter(pl.col("hadm_id").is_not_null()) \
            .join(hadm, on="hadm_id", how="inner")
    audit(qc, "micro", "hadm_in_stays", raw.height)
    org_level = raw.filter(pl.col("ab_itemid").is_null())
    audit(qc, "micro", "ab_itemid_is_null", org_level.height)
    out = org_level.select([
        pl.col("subject_id").cast(pl.Int64), pl.col("hadm_id"),
        pl.col("charttime").fill_null(pl.col("chartdate")).alias("charttime"),
        pl.col("spec_type_desc"), pl.col("org_name"),
        pl.col("org_name").is_not_null().alias("positive"),
    ])
    before = out.height
    out = out.unique(subset=["subject_id", "hadm_id", "charttime", "spec_type_desc",
                             "org_name"], keep="first", maintain_order=True)
    audit(qc, "micro", "dedup", out.height, dropped=before - out.height)
    write_table(out.sort(["subject_id", "charttime"]), "micro", qc, time_col="charttime",
                nonnull_cols=["org_name", "positive", "spec_type_desc"])
    qc.data["tables"]["micro"]["positive_rate"] = round(float(out["positive"].mean() or 0), 4)
    qc.data["tables"]["micro"]["n_hadm"] = int(out["hadm_id"].n_unique())
    qc.data["tables"]["micro"]["elapsed_s"] = round(time.time() - t0, 2)
    qc.save()


# ---------------------------------------------------------------- transfers
def build_transfers(ctx: dict, qc: QC) -> None:
    """hosp/transfers 无 stay_id：用 (hadm_id + careunit 属于该 stay 的 first/last) +
    区间重叠把 transfer 段归到 stay；无法归属的行 stay_id 置空（保留医院级转运段）。"""
    t0 = time.time()
    tr = read_csv_str(CSVROOT / "hosp" / "transfers.csv.gz")
    tr = cast_cols(tr, int_cols=["subject_id", "hadm_id", "transfer_id"],
                   dt_cols=["intime", "outtime"])
    st = ctx["stays"].select(["stay_id", "hadm_id", "first_careunit", "last_careunit",
                              "intime", "outtime"])
    raw = tr.join(st, on="hadm_id", how="inner", suffix="_stay")
    audit(qc, "transfers", "hadm_join", raw.height, input_rows=tr.height)
    m = raw.filter(
        pl.col("careunit").is_not_null()
        & ((pl.col("careunit") == pl.col("first_careunit"))
           | (pl.col("careunit") == pl.col("last_careunit")))
        & (pl.col("intime") < pl.col("outtime_stay"))
        & (pl.col("outtime") > pl.col("intime_stay"))
    ).with_columns(
        (pl.col("intime") - pl.col("intime_stay")).dt.total_seconds().abs().alias("_d"))
    key = (m.sort("_d").unique(subset=["transfer_id"], keep="first")
             .select(["transfer_id", "stay_id"]))
    audit(qc, "transfers", "icu_segment_matched", key.height)
    out = (tr.filter(pl.col("hadm_id").is_in(ctx["stays"]["hadm_id"].unique().to_list()))
             .join(key, on="transfer_id", how="left")
             .select(["subject_id", "hadm_id", "stay_id", "careunit", "intime", "outtime",
                      "eventtype"])
             .unique(keep="first", maintain_order=True)
             .sort(["subject_id", "intime"]))
    audit(qc, "transfers", "final", out.height,
          stay_linked=int(out["stay_id"].is_not_null().sum()))
    write_table(out, "transfers", qc, time_col="intime",
                nonnull_cols=["stay_id", "careunit", "eventtype"])
    qc.data["tables"]["transfers"]["stay_id_coverage"] = round(
        float(out["stay_id"].is_not_null().mean() or 0), 4)
    qc.data["tables"]["transfers"]["elapsed_s"] = round(time.time() - t0, 2)
    qc.save()


# ---------------------------------------------------------------- SOFA
def _score_series(x: np.ndarray, cuts: list[tuple[float, int]], ge: bool = True) -> np.ndarray:
    """按阈值把连续值映射为分值（缺失 -> 0）。cuts 为 [(阈值, 分), ...] 升序。"""
    out = np.zeros(x.shape, dtype=np.int8)
    valid = ~np.isnan(x)
    for thr, sc in cuts:
        m = (x >= thr) if ge else (x > thr)
        out = np.where(valid & m, np.int8(sc), out)
    return out.astype(np.int8)


def build_sofa(ctx: dict, qc: QC) -> None:
    """24h 滚动最差口径 SOFA 小时序列（阈值参考 icu-scheduling-agent/domain/scoring/sofa.py，
    见 BUILD.md 对照表；缺失分项按 0 计，覆盖率写入 qc）。"""
    t0 = time.time()
    stays = ctx["stays"]
    n = stays.height
    stay_ids = stays["stay_id"].to_numpy()
    rowmap = pl.DataFrame({"stay_id": stay_ids, "row_i": np.arange(n, dtype=np.int32)})
    hmin, hmax = GRID_H
    ncol = hmax - hmin + 1
    nwin = hmax + 1

    def rel(df: pl.DataFrame, time_col: str, keep: list[str]) -> pl.DataFrame:
        j = df.join(stays.select(["stay_id", "intime"]), on="stay_id", how="inner") \
              .with_columns(rel_hour_expr(time_col))
        j = j.filter((pl.col("rel_hour") >= hmin) & (pl.col("rel_hour") <= hmax))
        return j.select(["stay_id", "rel_hour"] + keep)

    def mat(df: pl.DataFrame, value: str, agg: str) -> np.ndarray:
        g = df.join(rowmap, on="stay_id", how="inner") \
              .group_by(["row_i", "rel_hour"]) \
              .agg(getattr(pl.col(value), agg)().alias("v"))
        m = np.full((n, ncol), np.nan, dtype=np.float32)
        if g.height:
            ri = g["row_i"].to_numpy()
            rh = g["rel_hour"].to_numpy()
            vv = g["v"].to_numpy().astype(np.float32)
            ok = ~np.isnan(vv)
            m[ri[ok], rh[ok] - hmin] = vv[ok]
        return m

    def win(m: np.ndarray) -> np.ndarray:
        v = np.lib.stride_tricks.sliding_window_view(m, H_PRE, axis=1)
        return v[:, 1:1 + nwin, :]

    def worst(m: np.ndarray, how: str) -> tuple[np.ndarray, np.ndarray]:
        w = win(m)                                  # (n, 72, 24)
        cnt = (~np.isnan(w)).sum(axis=2)            # (n, 72)
        ww = np.where((cnt > 0)[:, :, None], w, np.nan)
        if how == "max":
            out = np.nanmax(ww, axis=2)
        elif how == "min":
            out = np.nanmin(ww, axis=2)
        else:
            out = np.nansum(ww, axis=2)
        return np.where(cnt > 0, out, np.nan), cnt

    log("sofa: 读入 labs/vitals/uroutput/inputevents/procedureevents")
    lb = pl.read_parquet(OUT / "labs.parquet",
                         columns=["stay_id", "charttime", "itemid", "valuenum"])
    vt = pl.read_parquet(OUT / "vitals.parquet",
                         columns=["stay_id", "charttime", "itemid", "valuenum"])
    uo = pl.read_parquet(OUT / "uroutput.parquet",
                         columns=["stay_id", "charttime", "itemid", "value"])
    ie = pl.read_parquet(OUT / "inputevents.parquet",
                         columns=["stay_id", "starttime", "itemid", "rate", "rateuom"])
    pe = pl.read_parquet(OUT / "procedureevents.parquet",
                         columns=["stay_id", "starttime", "endtime", "itemid"])

    # ---- 呼吸：PaO2/FiO2 + 通气标志
    fio2 = vt.filter(pl.col("itemid") == 223835).select(["stay_id", "charttime", "valuenum"]) \
             .with_columns(pl.when(pl.col("valuenum") <= 1.0).then(pl.col("valuenum") * 100.0)
                           .otherwise(pl.col("valuenum")).alias("fio2")) \
             .filter(pl.col("fio2") > 20).select(["stay_id", "charttime", "fio2"])
    fio2_hi, fio2_cnt = worst(mat(rel(fio2, "charttime", ["fio2"]), "fio2", "max"), "max")

    pao2 = lb.filter(pl.col("itemid") == 50821).select(["stay_id", "charttime", "valuenum"]) \
             .rename({"valuenum": "pao2"})
    pao2_lo, pao2_cnt = worst(mat(rel(pao2, "charttime", ["pao2"]), "pao2", "min"), "min")
    pf = np.where((pao2_cnt > 0) & (fio2_cnt > 0), pao2_lo / (fio2_hi / 100.0), np.nan)

    vent = np.zeros((n, ncol), dtype=bool)      # 列轴 = rel_hour -24..71
    pv = pe.filter(pl.col("itemid") == 225792)
    if pv.height:
        pv = pv.join(stays.select(["stay_id", "intime"]), on="stay_id", how="inner") \
               .with_columns([
                   ((pl.col("starttime") - pl.col("intime")).dt.total_seconds() / 3600.0)
                   .floor().cast(pl.Int32).alias("h0"),
                   ((pl.col("endtime") - pl.col("intime")).dt.total_seconds() / 3600.0)
                   .ceil().cast(pl.Int32).alias("h1")]) \
               .with_columns([pl.col("h0").clip(0, hmax).alias("h0"),
                              pl.col("h1").clip(0, hmax).alias("h1")]) \
               .join(rowmap, on="stay_id", how="inner")
        for r, a, b in pv.select(["row_i", "h0", "h1"]).unique().iter_rows():
            vent[int(r), int(a) - hmin:min(int(b), hmax) - hmin + 1] = True
    vm = vt.filter(pl.col("itemid") == 223849).select(["stay_id", "charttime"]) \
           .join(stays.select(["stay_id", "intime"]), on="stay_id", how="inner") \
           .with_columns(rel_hour_expr("charttime")) \
           .filter((pl.col("rel_hour") >= 0) & (pl.col("rel_hour") <= hmax)) \
           .join(rowmap, on="stay_id", how="inner")
    for r, h in vm.select(["row_i", "rel_hour"]).unique().iter_rows():
        lo = int(h) - hmin
        vent[int(r), lo:min(int(h) + H_PRE - 1, hmax) - hmin + 1] = True
    vent_w = win(vent.astype(np.uint8)).max(axis=2).astype(bool)

    def resp_score(pfv: np.ndarray, ventv: np.ndarray) -> np.ndarray:
        s = np.zeros(pfv.shape, dtype=np.int8)
        valid = ~np.isnan(pfv)
        s = np.where(valid & (pfv < 400), np.int8(1), s)
        s = np.where(valid & (pfv < 300), np.int8(2), s)
        s = np.where(valid & (pfv < 200), np.where(ventv, np.int8(3), np.int8(2)), s)
        s = np.where(valid & (pfv < 100), np.where(ventv, np.int8(4), np.int8(2)), s)
        return s.astype(np.int8)

    sofa_resp = resp_score(pf, vent_w)

    # ---- 凝血 PLT（最差=最低；<150→1, <100→2, <50→3, <20→4）
    plt_df = lb.filter(pl.col("itemid") == 51265).select(["stay_id", "charttime", "valuenum"]) \
               .rename({"valuenum": "plt"})
    plt_lo, plt_cnt = worst(mat(rel(plt_df, "charttime", ["plt"]), "plt", "min"), "min")
    sofa_coag = _score_series(plt_lo, [(0, 4), (20, 3), (50, 2), (100, 1)])

    # ---- 肝 总胆红素
    tb = lb.filter(pl.col("itemid") == 50885).select(["stay_id", "charttime", "valuenum"]) \
            .rename({"valuenum": "tbil"})
    tb_hi, tb_cnt = worst(mat(rel(tb, "charttime", ["tbil"]), "tbil", "max"), "max")
    sofa_liver = _score_series(tb_hi, [(1.2, 1), (2.0, 2), (6.0, 3), (12.0, 4)])

    # ---- 循环 MAP（最差=最低；MAP<70 记 1 分）+ 升压药
    map_df = vt.filter(pl.col("itemid").is_in([220052, 220181])) \
               .select(["stay_id", "charttime", "valuenum"]).rename({"valuenum": "map"})
    map_lo, map_cnt = worst(mat(rel(map_df, "charttime", ["map"]), "map", "min"), "min")
    sofa_cardio = np.where(~np.isnan(map_lo) & (map_lo < 70.0), 1, 0).astype(np.int8)

    wt = vt.filter(pl.col("itemid").is_in([226512, 224639])) \
           .select(["stay_id", "valuenum"]).rename({"valuenum": "wt"}) \
           .group_by("stay_id").agg(pl.col("wt").median()).rename({"wt": "weight_kg"})

    if ie.height:
        ivs = ie.join(wt, on="stay_id", how="left") \
                .with_columns([pl.col("rateuom").fill_null("").str.to_lowercase().alias("u"),
                               pl.col("rate").cast(pl.Float64)])
        ivs = ivs.with_columns(pl.when(pl.col("u").str.contains("mcg/kg/min"))
                               .then(pl.col("rate"))
                               .when(pl.col("u").str.contains("mg/kg/min"))
                               .then(pl.col("rate") * 1000.0)
                               .when(pl.col("u").str.contains("mcg/min"))
                               .then(pl.col("rate") / pl.col("weight_kg").fill_null(80.0))
                               .when(pl.col("u").str.contains("units/min"))
                               .then(pl.col("rate"))
                               .when(pl.col("u").str.contains("units/hour"))
                               .then(pl.col("rate") / 60.0)
                               .when(pl.col("u").str.contains("mg/hour"))
                               .then(pl.col("rate") * 1000.0 / 60.0
                                     / pl.col("weight_kg").fill_null(80.0))
                               .otherwise(None).alias("rate_n"))
        units = ivs.group_by(["itemid", "u"]).agg(pl.len().alias("n")) \
                   .sort("n", descending=True)
        qc.data["sofa"]["vaso_rateuom"] = [{"itemid": int(i), "uom": str(u), "n": int(c)}
                                           for i, u, c in units.iter_rows()]
        ivs = (ivs.join(stays.select(["stay_id", "intime"]), on="stay_id", how="left")
                  .with_columns(rel_hour_expr("starttime"))
                  .filter((pl.col("rel_hour") >= hmin) & (pl.col("rel_hour") <= hmax)))
        drug_max = {}
        for iid, name in [(221906, "norepi"), (221289, "epi"), (221662, "dopamine"),
                          (221653, "dobutamine"), (222315, "vasopressin"),
                          (221749, "phenyl")]:
            d = ivs.filter((pl.col("itemid") == iid) & pl.col("rate_n").is_not_null()) \
                   .select(["stay_id", "rel_hour", "rate_n"])
            if d.height:
                drug_max[name] = worst(mat(d, "rate_n", "max"), "max")
                log("  vaso %-11s rows=%-7d 有值 stay-hour=%d", name, d.height,
                    int((drug_max[name][1] > 0).sum()))
        zero = np.zeros_like(pf)
        dopa = np.nan_to_num(drug_max.get("dopamine", (zero, zero))[0])
        nor = np.nan_to_num(drug_max.get("norepi", (zero, zero))[0])
        epi = np.nan_to_num(drug_max.get("epi", (zero, zero))[0])
        dob = np.nan_to_num(drug_max.get("dobutamine", (zero, zero))[0])
        vas = np.nan_to_num(drug_max.get("vasopressin", (zero, zero))[0])
        phn = np.nan_to_num(drug_max.get("phenyl", (zero, zero))[0])
        s4 = (dopa > 15) | (nor > 0.1) | (epi > 0.1) | (vas > 0.03) | (phn > 1.0)
        s3 = ((dopa > 5) & (dopa <= 15)) | ((nor > 0) & (nor <= 0.1)) \
            | ((epi > 0) & (epi <= 0.1)) | ((vas > 0) & (vas <= 0.03)) \
            | ((phn > 0) & (phn <= 1.0))
        s2 = ((dopa > 0) & (dopa <= 5)) | (dob > 0)
        vasc = np.where(s4, 4, np.where(s3, 3, np.where(s2, 2, 0))).astype(np.int8)
        sofa_cardio = np.maximum(sofa_cardio, vasc)
        qc.data["sofa"]["vaso_present"] = {k: bool(v[1].max() > 0)
                                           for k, v in drug_max.items()}

    # ---- 神经 GCS（同一 charttime 三项求和，至少 2 项；15→0, 13-14→1, 10-12→2, 6-9→3, <6→4）
    gcs = vt.filter(pl.col("itemid").is_in([220739, 223900, 223901])) \
            .group_by(["stay_id", "charttime"]) \
            .agg([pl.col("valuenum").sum().alias("gcs"), pl.len().alias("ncomp")]) \
            .filter(pl.col("ncomp") >= 2).select(["stay_id", "charttime", "gcs"])
    gcs_lo, gcs_cnt = worst(mat(rel(gcs, "charttime", ["gcs"]), "gcs", "min"), "min")
    sofa_cns = _score_series(gcs_lo, [(0, 4), (6, 3), (10, 2), (13, 1), (15, 0)])

    # ---- 肾 肌酐 + 尿量
    cr = lb.filter(pl.col("itemid") == 50912).select(["stay_id", "charttime", "valuenum"]) \
           .rename({"valuenum": "cr"})
    cr_hi, cr_cnt = worst(mat(rel(cr, "charttime", ["cr"]), "cr", "max"), "max")
    sofa_renal = _score_series(cr_hi, [(1.2, 1), (2.0, 2), (3.5, 3), (5.0, 4)])

    uod = uo.filter(~pl.col("itemid").is_in(sorted(UO_NEGATIVE))) \
           .select(["stay_id", "charttime", "value"])
    if uod.height:
        uod = rel(uod, "charttime", ["value"])
        uo_sum, uo_bins = worst(mat(uod, "value", "sum"), "sum")
        uo_ok = uo_bins >= 6                      # 窗口内至少 6 个小时有记录才采用尿量口径
        uo_sc = np.where(uo_ok & (uo_sum < 200), 4, np.where(uo_ok & (uo_sum < 500), 3, 0))
        sofa_renal = np.maximum(sofa_renal, uo_sc.astype(np.int8))
        qc.data["sofa"]["uo_usable_share"] = round(float(uo_ok.mean()), 4)

    subs = {"sofa_resp": sofa_resp, "sofa_coag": sofa_coag, "sofa_liver": sofa_liver,
            "sofa_cardio": sofa_cardio, "sofa_cns": sofa_cns, "sofa_renal": sofa_renal}
    total = np.clip(sum(v.astype(np.int16) for v in subs.values()), 0, 24).astype(np.int16)

    los_h = stays["los_hours"].to_numpy()
    los_floor = np.floor(np.nan_to_num(los_h, nan=0.0)).astype(np.int32)
    hours = np.arange(nwin)
    valid = hours[None, :] <= los_floor[:, None]
    valid[:, 0] = True

    rr, hh = np.where(valid)
    out = pl.DataFrame({
        "stay_id": stay_ids[rr].astype(np.int64),
        "hour_index": hh.astype(np.int32),
        "sofa_total": total[rr, hh].astype(np.int16),
        "sofa_resp": subs["sofa_resp"][rr, hh].astype(np.int16),
        "sofa_coag": subs["sofa_coag"][rr, hh].astype(np.int16),
        "sofa_liver": subs["sofa_liver"][rr, hh].astype(np.int16),
        "sofa_cardio": subs["sofa_cardio"][rr, hh].astype(np.int16),
        "sofa_cns": subs["sofa_cns"][rr, hh].astype(np.int16),
        "sofa_renal": subs["sofa_renal"][rr, hh].astype(np.int16),
    })
    write_table(out, "sofa_hourly", qc, nonnull_cols=["sofa_total"],
                sort_by=["stay_id", "hour_index"])

    h0 = out.filter(pl.col("hour_index") == 0)
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]

    def cov0(arr: np.ndarray) -> float:
        return round(float((~np.isnan(arr[:, 0])).sum()) / max(n, 1), 4)

    qc.data["sofa"] = dict(qc.data.get("sofa") or {})
    qc.data["sofa"].update({
        "hours_total": out.height,
        "stays_in_sofa": int(out["stay_id"].n_unique()),
        "hour0_stay_coverage": round(h0.height / max(n, 1), 4),
        "mean_total_hour0": round(float(h0["sofa_total"].mean() or 0), 3),
        "mean_total_all_hours": round(float(out["sofa_total"].mean() or 0), 3),
        "total_min": int(out["sofa_total"].min() or 0),
        "total_max": int(out["sofa_total"].max() or 0),
        "component_present_h0": {
            "resp_PF": cov0(pf), "coag_PLT": cov0(plt_lo), "liver_TBIL": cov0(tb_hi),
            "cardio_MAP": cov0(map_lo), "cns_GCS": cov0(gcs_lo), "renal_Cr": cov0(cr_hi)},
        "vent_window_share_h0": round(float(vent_w[:, 0].mean()), 4),
        "quantiles_hour0": {},
        "subscore_means_hour0": {},
        "sf_ratio_note": "SpO2/FiO2 仅记录在 qc，不进入 resp 主口径（§2.6）",
    })
    qc.data["sofa"]["quantiles_hour0"]["sofa_total"] = {
        f"p{int(q * 100)}": float(np.quantile(h0["sofa_total"].to_numpy(), q)) for q in qs}
    byh = (out.group_by("hour_index").agg([pl.col("sofa_total").mean().alias("mean_total"),
                                           pl.len().alias("n")]).sort("hour_index"))
    qc.data["sofa"]["mean_total_by_hour"] = {
        str(h): {"mean_total": round(float(m), 3), "n": int(nn)}
        for h, m, nn in [(r[0], r[1], r[2]) for r in byh.iter_rows()]}
    for hh in (0, 1, 2, 4, 6, 12, 24, 48, 71):
        r = out.filter(pl.col("hour_index") == hh)
        qc.data["sofa"][f"mean_total_h{hh}"] = round(float(r["sofa_total"].mean() or 0), 3)
        qc.data["sofa"][f"n_h{hh}"] = int(r.height)
    for c in ["sofa_resp", "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_cns",
              "sofa_renal"]:
        qc.data["sofa"]["subscore_means_hour0"][c] = round(float(h0[c].mean() or 0), 3)
        qc.data["sofa"]["quantiles_hour0"][c] = {
            f"p{int(q * 100)}": float(np.quantile(h0[c].to_numpy(), q)) for q in qs}

    spo2 = vt.filter(pl.col("itemid") == 220277).select(["stay_id", "charttime", "valuenum"]) \
             .rename({"valuenum": "spo2"})
    sf, _ = worst(mat(rel(spo2, "charttime", ["spo2"]), "spo2", "min"), "min")
    sf_ratio = np.where(fio2_cnt > 0, sf / (fio2_hi / 100.0), np.nan)
    qc.data["sofa"]["sf_ratio_available_share_h0"] = round(
        float((~np.isnan(sf_ratio[:, 0])).mean()), 4)
    qc.data["sofa"]["elapsed_s"] = round(time.time() - t0, 2)
    log("sofa 完成: hours=%d hour0覆盖=%.3f 均值(全部)=%.2f 均值(h0)=%.2f 用时 %.1fs",
        out.height, qc.data["sofa"]["hour0_stay_coverage"],
        qc.data["sofa"]["mean_total_all_hours"], qc.data["sofa"]["mean_total_hour0"],
        time.time() - t0)
    qc.save()


# ---------------------------------------------------------------- main
MODULES = {
    "stays": build_stays, "cohort": build_cohort, "vitals": build_vitals, "labs": build_labs,
    "inputevents": build_inputevents, "procedureevents": build_procedureevents,
    "uroutput": build_uroutput, "micro": build_micro, "transfers": build_transfers,
    "sofa": build_sofa,
}
ORDER = list(MODULES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all",
                    help="逗号分隔：" + ",".join(ORDER) + " 或 all")
    ap.add_argument("--limit-stays", type=int, default=0, help="随机抽样 stay 数（联调）")
    ap.add_argument("--csv-big", action="store_true",
                    help="强制所有表导出 csv（默认只导 stays/cohort/micro/transfers/sofa_hourly）")
    args = ap.parse_args()
    OPTS["csv_big"] = bool(args.csv_big)

    only = ORDER if args.only == "all" else [s.strip() for s in args.only.split(",")]
    bad = [s for s in only if s not in MODULES]
    if bad:
        raise SystemExit(f"未知模块 {bad}，可选 {ORDER}")

    tag = "build" if not args.limit_stays else f"build_sub{args.limit_stays}"
    logp = setup_logging(tag)
    t_all = time.time()
    log("=" * 78)
    log("共享数据层构建 | only=%s limit_stays=%s seed=%d", only, args.limit_stays or "None", SEED)
    log("工作区 %s", WORK)
    log("日志 %s", logp)

    stays, meta = load_stays(args.limit_stays or None)
    qc = QC(OUT / "qc.json", {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": "experiments/etl/build_tables.py",
        "protocol": "experiments/PROTOCOL.md §2",
        "seed": SEED,
        "limit_stays": args.limit_stays or None,
        "window": f"charttime in [intime-{H_PRE}h, intime+{H_POST}h] and charttime < outtime",
        "python": sys.version.split()[0],
        "polars": pl.__version__,
    })
    ctx = {"stays": stays, "stays_meta": meta, "qc": qc}
    qc.data["meta"]["n_stays"] = stays.height

    for m in only:
        if m == "sofa" and not (OUT / "vitals.parquet").exists():
            raise SystemExit("sofa 需要先构建 vitals/labs/uroutput/inputevents/procedureevents")
        log("-" * 78)
        log(">>> %s", m)
        MODULES[m](ctx, qc)

    qc.data["meta"]["total_elapsed_s"] = round(time.time() - t_all, 1)
    qc.save()
    log("=" * 78)
    log("完成，总耗时 %.1f s（%.1f min）", time.time() - t_all, (time.time() - t_all) / 60)
    log("qc -> %s", OUT / "qc.json")


if __name__ == "__main__":
    main()
