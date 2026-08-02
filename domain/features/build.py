"""Build admission-time features → feat.sample_matrix (no outcome leak)."""

from __future__ import annotations

import json
import math
import re

from sqlalchemy import text

from data_access.mimic_repo import (
    ELIXHAUSER_CATEGORIES,
    LAB_ITEMIDS,
    VITAL_ITEMIDS,
    fetch_abg_first,
    fetch_admission_info,
    fetch_cohort,
    fetch_elixhauser,
    fetch_first_icu_labs,
    fetch_first_icu_vitals,
    fetch_gcs_subscores,
    fetch_gcs_total,
    fetch_genetic_flags,
    fetch_pre_icu_labs,
    fetch_pre_icu_los,
    fetch_sofa_components,
    fetch_vasopressor_1h,
    fetch_vent_flag,
)
from infra.config import load_yaml
from infra.db import get_engine


def prediction_hour_index() -> int:
    """S1: hour_index=1 (intime+1h). S2: vary per sample row."""
    return int(load_yaml("features.yaml").get("hour_index", 1))

# ── 基线 ──
BASE_COLS = ["anchor_age", "gender_m", "careunit_micu", "careunit_sicu", "careunit_ccu", "careunit_other"]

# ── 化验 ──
LAB_COLS = [f"lab_{k}" for k in LAB_ITEMIDS]

# ── 生命体征 ──
VITAL_COLS = [f"vital_{k}" for k in VITAL_ITEMIDS]

# ── 入院信息 ──
ADMIT_COLS = [
    "admit_type_emergency", "admit_type_urgent", "admit_type_elective",
    "admit_from_ed", "admit_from_transfer",
    "insurance_medicare", "insurance_medicaid", "insurance_private",
]

# ── 遗传病标志 ──
GENETIC_COLS = ["has_q_code", "has_congenital_heart", "has_family_ihd"]

# ── Elixhauser 合并症（30 类） ──
ELIX_COLS = [f"elix_{cat}" for cat in ELIXHAUSER_CATEGORIES]

# ── SOFA 分量（6 器官系统 + 总分） ──
SOFA_COLS = [
    "sofa_respiration", "sofa_coagulation", "sofa_liver",
    "sofa_cardiovascular", "sofa_neurological", "sofa_renal",
    "sofa_total",
]

# ── 动脉血气 ──
ABG_COLS = ["pao2", "paco2", "abg_fio2", "pf_ratio", "intubation_flag"]

# ── GCS 分项 ──
GCS_SUB_COLS = ["gcs_eye", "gcs_verbal", "gcs_motor"]

# ── 机械通气标志 ──
VENT_COLS = ["vent_flag"]

# ── Wave A 全量特征列（所有合法特征） ──
WAVE_A_FULL_COLS = (
    BASE_COLS + LAB_COLS + VITAL_COLS + ADMIT_COLS + GENETIC_COLS
    + ELIX_COLS + SOFA_COLS + ABG_COLS + GCS_SUB_COLS + VENT_COLS
    + [
        "pre_icu_los_hours", "gcs_total", "vasopressor_1h",
        "shock_index", "spo2_fio2_ratio",
    ]
)

# ── Wave2.6 新特征 ──
NEW_COLS = [
    "pre_icu_los_hours",       # 入院→入ICU时长
    "gcs_total",               # GCS总分(E+V+M)，入ICU 1h内
    "vasopressor_1h",          # 1h内是否用血管活性药
    "shock_index",             # 心率/NIBP收缩压（衍生）
    "spo2_fio2_ratio",         # SpO2/FiO2（衍生，反映氧合）
]

# ── 两个特征集 ──

# 模型A：优化特征集（去掉31个无用特征，保留14个有用+新增5个）
FEATURE_COLS = [
    # 基线（保留anchor_age, careunit_other）
    "anchor_age", "careunit_other",
    # 体征（保留temp, gcs_motor, fio2, heart_rate, nbps, resp_rate）
    "vital_temp", "vital_gcs_total", "vital_fio2",
    "vital_heart_rate", "vital_nbps", "vital_resp_rate",
    # 化验（保留lactate, bun, ph, potassium, inr, albumin）
    "lab_lactate", "lab_bun", "lab_ph", "lab_potassium", "lab_inr", "lab_albumin",
    # 新特征
    "pre_icu_los_hours", "gcs_total", "vasopressor_1h",
    "shock_index", "spo2_fio2_ratio",
]

# 模型B：纯预ICU特征（不含任何入ICU后数据）
FEATURE_COLS_PRE_ICU = [
    "anchor_age", "gender_m", "careunit_micu", "careunit_sicu",
    "careunit_ccu", "careunit_other",
    "pre_icu_los_hours",
] + [f"lab_{k}" for k in LAB_ITEMIDS] + ADMIT_COLS + GENETIC_COLS


def _gender_m(g: str | None) -> int:
    return 1 if (g or "").upper() == "M" else 0


def _careunit_flags(name: str | None) -> dict[str, int]:
    u = (name or "").upper()
    micu = 1 if "MICU" in u or re.search(r"\bMICU\b", u) else 0
    sicu = 1 if "SICU" in u else 0
    ccu = 1 if re.search(r"\bCCU\b", u) or "CORONARY" in u else 0
    return {
        "careunit_micu": micu, "careunit_sicu": sicu,
        "careunit_ccu": ccu, "careunit_other": 1 if micu == 0 and sicu == 0 and ccu == 0 else 0,
    }


def _admit_flags(info: dict | None) -> dict[str, int]:
    if info is None:
        return {c: 0 for c in ADMIT_COLS}
    t = (info.get("admission_type") or "").upper()
    loc = (info.get("admission_location") or "").upper()
    ins = (info.get("insurance") or "").upper()
    return {
        "admit_type_emergency": 1 if "EMER" in t else 0,
        "admit_type_urgent": 1 if "URGENT" in t else 0,
        "admit_type_elective": 1 if "ELECTIVE" in t else 0,
        "admit_from_ed": 1 if "EMERGENCY ROOM" in loc else 0,
        "admit_from_transfer": 1 if "TRANSFER" in loc else 0,
        "insurance_medicare": 1 if "MEDICARE" in ins else 0,
        "insurance_medicaid": 1 if "MEDICAID" in ins else 0,
        "insurance_private": 1 if "PRIVATE" in ins else 0,
    }


def _json_safe(obj):
    """Convert NaN/None to None for JSON serialization."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def row_to_features(
    row: dict,
    pre_icu_labs: dict[str, float] | None = None,
    first_icu_labs: dict[str, float] | None = None,
    first_vitals: dict[str, float] | None = None,
    admit_info: dict | None = None,
    genetic_flags: dict[str, int] | None = None,
    pre_icu_los: float | None = None,
    gcs_total: int | None = None,
    vasopressor_1h: int | None = None,
    elixhauser: dict[str, int] | None = None,
    sofa_components: dict[str, int] | None = None,
    abg: dict[str, float] | None = None,
    gcs_subscores: dict[str, int] | None = None,
    vent_flag: int | None = None,
) -> dict:
    feat = {
        "anchor_age": int(row.get("anchor_age") or 0),
        "gender_m": _gender_m(row.get("gender")),
    }
    feat.update(_careunit_flags(row.get("first_careunit")))

    # 化验：优先预ICU，否则入ICU后1h内
    for lab_name in LAB_ITEMIDS:
        val = None
        if pre_icu_labs and lab_name in pre_icu_labs:
            val = pre_icu_labs[lab_name]
        elif first_icu_labs and lab_name in first_icu_labs:
            val = first_icu_labs[lab_name]
        feat[f"lab_{lab_name}"] = val

    # 生命体征：入ICU后第一个值
    for vital_name in VITAL_ITEMIDS:
        val = None
        if first_vitals and vital_name in first_vitals:
            val = first_vitals[vital_name]
        feat[f"vital_{vital_name}"] = val

    # 入院信息
    feat.update(_admit_flags(admit_info))

    # 遗传病标志
    if genetic_flags:
        for col in GENETIC_COLS:
            feat[col] = genetic_flags.get(col, 0)
    else:
        for col in GENETIC_COLS:
            feat[col] = 0

    # ── Wave2.6 新特征 ──
    feat["pre_icu_los_hours"] = pre_icu_los
    feat["gcs_total"] = gcs_total
    feat["vasopressor_1h"] = vasopressor_1h if vasopressor_1h is not None else 0

    # 衍生特征：休克指数 = HR / NIBP收缩压
    hr = feat.get("vital_heart_rate")
    nbps = feat.get("vital_nbps")
    if hr is not None and nbps is not None and nbps > 0:
        feat["shock_index"] = round(hr / nbps, 4)
    else:
        feat["shock_index"] = None

    # 衍生特征：SpO2/FiO2 氧合比
    spo2 = feat.get("vital_spo2")
    fio2 = feat.get("vital_fio2")
    if spo2 is not None and fio2 is not None and fio2 > 0:
        # SpO2 0-100, FiO2 0.21-1.0, 归一化到同一量纲
        feat["spo2_fio2_ratio"] = round(spo2 / fio2, 2)
    else:
        feat["spo2_fio2_ratio"] = None

    # ── Wave A 扩展特征 ──

    # Elixhauser 合并症（30 类，0/1 标志）
    for col in ELIX_COLS:
        feat[col] = elixhauser.get(col, 0) if elixhauser else 0

    # SOFA 分量（6 个器官系统评分 + 总分）
    for col in SOFA_COLS:
        feat[col] = sofa_components.get(col, 0) if sofa_components else 0

    # 动脉血气
    for col in ABG_COLS:
        if abg:
            val = abg.get(col)
            # Ensure numeric types for LightGBM compatibility (no object dtype)
            if col == "intubation_flag":
                feat[col] = int(val) if val is not None else 0
            else:
                feat[col] = float(val) if val is not None else float("nan")
        else:
            feat[col] = 0 if col == "intubation_flag" else float("nan")

    # GCS 分项（Eye / Verbal / Motor）
    for col in GCS_SUB_COLS:
        feat[col] = gcs_subscores.get(col, 0) if gcs_subscores else 0

    # 机械通气标志
    feat["vent_flag"] = vent_flag if vent_flag is not None else 0

    return feat


def build_features() -> dict:
    rows = fetch_cohort()
    # S1 FEATURE_COLS 所需查询（跳过 admit/genetic/elix/sofa/abg 以控时；完整池仍可由 row_to_features 填 0）
    pre_icu = fetch_pre_icu_labs()
    icu_labs = fetch_first_icu_labs()
    vitals = fetch_first_icu_vitals()
    admits: dict = {}
    genetics: dict = {}
    los = fetch_pre_icu_los()
    gcs = fetch_gcs_total()
    vaso = fetch_vasopressor_1h()
    elix: dict = {}
    sofa: dict = {}
    abg: dict = {}
    gcs_sub: dict = {}
    vent: dict = {}

    hour_index = prediction_hour_index()
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE feat.sample_matrix"))
        for row in rows:
            sid = row["stay_id"]
            feat = row_to_features(
                row,
                pre_icu_labs=pre_icu.get(sid),
                first_icu_labs=icu_labs.get(sid),
                first_vitals=vitals.get(sid),
                admit_info=admits.get(sid),
                genetic_flags=genetics.get(sid),
                pre_icu_los=los.get(sid),
                gcs_total=gcs.get(sid),
                vasopressor_1h=vaso.get(sid),
                elixhauser=elix.get(sid),
                sofa_components=sofa.get(sid),
                abg=abg.get(sid),
                gcs_subscores=gcs_sub.get(sid),
                vent_flag=vent.get(sid),
            )
            conn.execute(
                text(
                    """
                    INSERT INTO feat.sample_matrix (stay_id, hour_index, feature_json)
                    VALUES (:stay_id, :hour_index, CAST(:feature_json AS jsonb))
                    """
                ),
                {
                    "stay_id": sid,
                    "hour_index": hour_index,
                    "feature_json": json.dumps(
                        {k: _json_safe(v) for k, v in feat.items()}, ensure_ascii=False
                    ),
                },
            )
    return {
        "feat_rows": len(rows),
        "feature_cols": FEATURE_COLS,
        "hour_index": hour_index,
        "stored_feature_cols": WAVE_A_FULL_COLS,
    }