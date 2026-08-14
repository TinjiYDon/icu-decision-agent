"""Read ICU stays from mock or Layer0 MIMIC."""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from infra.config import get_data_source, get_layer0_dsn, get_settings


def _window_hours() -> int:
    from infra.config import load_yaml
    return max(int(load_yaml("features.yaml").get("prediction_offset_hours", 1)), 0)


def _wh(window_hours: int | None) -> int:
    h = _window_hours() if window_hours is None else int(window_hours)
    return max(h, 0)



def _mock_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def _layer0_engine() -> Engine | None:
    dsn = get_layer0_dsn()
    if not dsn:
        return None
    return create_engine(dsn, pool_pre_ping=True)


def _read_engine() -> Engine:
    source = get_data_source()
    if source == "mock":
        return _mock_engine()
    engine = _layer0_engine()
    if engine is None:
        raise RuntimeError("layer0 DSN not configured; set configs/database.yaml")
    return engine


def count_icustays() -> int:
    source = get_data_source()
    if source == "mock":
        with _mock_engine().connect() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM mock.icustays")).scalar_one()
    with _read_engine().connect() as conn:
        return conn.execute(
            text("SELECT COUNT(DISTINCT stay_id) FROM mimiciv_icu.icustays")
        ).scalar_one()


def fetch_icustays() -> list[dict[str, Any]]:
    """Fetch icustays rows from Layer0 (mimic) or mock schema."""
    source = get_data_source()
    if source == "mock":
        sql = """
            SELECT stay_id, subject_id,
                   stay_id AS hadm_id,
                   NULL::text AS first_careunit, NULL::text AS last_careunit,
                   intime, outtime,
                   EXTRACT(EPOCH FROM (COALESCE(outtime, intime) - intime)) / 3600.0 AS los_hours
            FROM mock.icustays
            ORDER BY stay_id
        """
    else:
        sql = """
            SELECT DISTINCT ON (stay_id)
                   stay_id, subject_id, hadm_id,
                   first_careunit, last_careunit,
                   intime, outtime, los AS los_hours
            FROM mimiciv_icu.icustays
            ORDER BY stay_id, intime
        """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return [dict(r) for r in rows]


def fetch_cohort() -> list[dict[str, Any]]:
    """ICU stays joined with patients + admissions for features/labels."""
    source = get_data_source()
    if source == "mock":
        sql = """
            SELECT i.stay_id, i.subject_id, i.stay_id AS hadm_id,
                   i.intime, i.outtime,
                   EXTRACT(EPOCH FROM (COALESCE(i.outtime, i.intime) - i.intime)) / 3600.0 AS los_hours,
                   65 AS anchor_age, 'M'::text AS gender, NULL::date AS dod,
                   NULL::timestamp AS deathtime,
                   0 AS hospital_expire_flag,
                   'MICU'::text AS first_careunit
            FROM mock.icustays i
            ORDER BY i.stay_id
        """
    else:
        sql = """
            SELECT DISTINCT ON (i.stay_id)
                   i.stay_id, i.subject_id, i.hadm_id, i.intime, i.outtime, i.los AS los_hours,
                   i.first_careunit,
                   p.anchor_age, p.gender, p.dod,
                   a.deathtime,
                   a.hospital_expire_flag
            FROM mimiciv_icu.icustays i
            JOIN mimiciv_hosp.patients p ON i.subject_id = p.subject_id
            JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
            ORDER BY i.stay_id, i.intime
        """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return [dict(r) for r in rows]


# ── 新增特征提取函数 ────────────────────────────────────────────────

# 关键化验 itemid（Blood 来源）
LAB_ITEMIDS = {
    "lactate": 50813,
    "creatinine": 50912,
    "bun": 51006,
    "hemoglobin": 50811,
    "hematocrit": 51221,
    "wbc": 51300,
    "platelets": 51265,
    "sodium": 50983,
    "potassium": 50971,
    "chloride": 50902,
    "bicarbonate": 50882,
    "glucose": 50931,
    "calcium": 50893,
    "magnesium": 50960,
    "albumin": 50862,
    "bilirubin": 50885,
    "inr": 51237,
    "ph": 50820,
}

# 关键生命体征 itemid
VITAL_ITEMIDS = {
    "heart_rate": 220045,
    "nbps": 220179,  # NIBP systolic
    "nbpd": 220180,  # NIBP diastolic
    "nbpm": 220181,  # NIBP mean
    "resp_rate": 220210,
    "spo2": 220277,
    "temp": 223761,  # Fahrenheit — convert to Celsius
    "gcs_total": 223901,  # GCS Motor (can approximate total)
    "fio2": 223835,
    "peep": 220339,
}


def _itemid_list() -> str:
    """Comma-separated list of lab itemids for SQL IN clause."""
    return ",".join(str(v) for v in LAB_ITEMIDS.values())


def _vital_itemid_list() -> str:
    return ",".join(str(v) for v in VITAL_ITEMIDS.values())


def fetch_pre_icu_labs() -> dict[int, dict[str, float]]:
    """Last lab value BEFORE intime for each stay → {stay_id: {lab_name: value}}."""
    source = get_data_source()
    if source == "mock":
        return {}
    sql = f"""
        WITH ranked AS (
            SELECT
                i.stay_id,
                l.itemid,
                l.valuenum,
                ROW_NUMBER() OVER (PARTITION BY i.stay_id, l.itemid ORDER BY l.charttime DESC) AS rn
            FROM mimiciv_icu.icustays i
            JOIN mimiciv_hosp.labevents l ON i.hadm_id = l.hadm_id
            WHERE l.charttime < i.intime
              AND l.valuenum IS NOT NULL
              AND l.itemid IN ({_itemid_list()})
        )
        SELECT stay_id, itemid, valuenum FROM ranked WHERE rn = 1
    """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    result: dict[int, dict[str, float]] = {}
    name_map = {v: k for k, v in LAB_ITEMIDS.items()}
    for r in rows:
        sid = r["stay_id"]
        name = name_map.get(r["itemid"])
        if name is None:
            continue
        result.setdefault(sid, {})[name] = float(r["valuenum"])
    return result


def fetch_first_icu_vitals(window_hours: int | None = None) -> dict[int, dict[str, float]]:
    """Latest vital sign value up to intime+window_hours for each stay.

    Uses last-observation-carried-forward: for prediction time t=intime+h,
    we take the most recent chartevents value with charttime <= t.
    This makes features differ across h (unlike taking the first value).
    """
    source = get_data_source()
    if source == "mock":
        return {}
    wh = _wh(window_hours)
    sql = f"""
        WITH ranked AS (
            SELECT
                c.stay_id,
                c.itemid,
                c.valuenum,
                ROW_NUMBER() OVER (PARTITION BY c.stay_id, c.itemid ORDER BY c.charttime DESC) AS rn
            FROM mimiciv_icu.chartevents c
            JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
            WHERE c.charttime <= i.intime + INTERVAL '{wh} hours'
              AND c.valuenum IS NOT NULL
              AND c.itemid IN ({_vital_itemid_list()})
        )
        SELECT stay_id, itemid, valuenum FROM ranked WHERE rn = 1
    """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    result: dict[int, dict[str, float]] = {}
    name_map = {v: k for k, v in VITAL_ITEMIDS.items()}
    for r in rows:
        sid = r["stay_id"]
        name = name_map.get(r["itemid"])
        if name is None:
            continue
        val = float(r["valuenum"])
        # Convert Fahrenheit to Celsius for temp
        if name == "temp":
            val = (val - 32.0) * 5.0 / 9.0
        result.setdefault(sid, {})[name] = val
    return result


def fetch_first_icu_labs(window_hours: int | None = None) -> dict[int, dict[str, float]]:
    """Latest lab value up to intime+window_hours per stay.

    Uses last-observation-carried-forward so features differ across h.
    Used as fallback when pre-ICU labs are not available.
    """
    source = get_data_source()
    if source == "mock":
        return {}
    wh = _wh(window_hours)
    sql = f"""
        WITH ranked AS (
            SELECT
                i.stay_id,
                l.itemid,
                l.valuenum,
                ROW_NUMBER() OVER (PARTITION BY i.stay_id, l.itemid ORDER BY l.charttime DESC) AS rn
            FROM mimiciv_icu.icustays i
            JOIN mimiciv_hosp.labevents l ON i.hadm_id = l.hadm_id
            WHERE l.charttime <= i.intime + INTERVAL '{wh} hours'
              AND l.valuenum IS NOT NULL
              AND l.itemid IN ({_itemid_list()})
        )
        SELECT stay_id, itemid, valuenum FROM ranked WHERE rn = 1
    """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    result: dict[int, dict[str, float]] = {}
    name_map = {v: k for k, v in LAB_ITEMIDS.items()}
    for r in rows:
        sid = r["stay_id"]
        name = name_map.get(r["itemid"])
        if name is None:
            continue
        result.setdefault(sid, {})[name] = float(r["valuenum"])
    return result


def fetch_admission_info() -> dict[int, dict[str, Any]]:
    """Admission-level info per stay."""
    source = get_data_source()
    if source == "mock":
        return {}
    sql = """
        SELECT DISTINCT ON (i.stay_id)
               i.stay_id,
               a.admission_type,
               a.admission_location,
               a.insurance
        FROM mimiciv_icu.icustays i
        JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
        ORDER BY i.stay_id, i.intime
    """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return {r["stay_id"]: dict(r) for r in rows}


def fetch_genetic_flags() -> dict[int, dict[str, int]]:
    """Genetic / congenital / family-history flags from diagnoses_icd per stay."""
    source = get_data_source()
    if source == "mock":
        return {}
    sql = """
        SELECT
            i.stay_id,
            MAX(CASE WHEN d.icd_code LIKE 'Q%' AND d.icd_version = 10 THEN 1 ELSE 0 END) AS has_q_code,
            MAX(CASE WHEN d.icd_code LIKE 'Q2%' AND d.icd_version = 10 THEN 1 ELSE 0 END) AS has_congenital_heart,
            MAX(CASE WHEN d.icd_code = 'Z8249' AND d.icd_version = 10 THEN 1 ELSE 0 END) AS has_family_ihd
        FROM mimiciv_icu.icustays i
        LEFT JOIN mimiciv_hosp.diagnoses_icd d ON i.hadm_id = d.hadm_id
        GROUP BY i.stay_id
    """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return {r["stay_id"]: dict(r) for r in rows}


# ── Wave2.6 新特征 ─────────────────────────────────────────────────

def fetch_pre_icu_los() -> dict[int, float]:
    """Hours from hospital admission to ICU admission per stay."""
    source = get_data_source()
    if source == "mock":
        return {}
    sql = """
        SELECT DISTINCT ON (i.stay_id)
               i.stay_id,
               EXTRACT(EPOCH FROM (i.intime - a.admittime)) / 3600.0 AS pre_icu_los
        FROM mimiciv_icu.icustays i
        JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
        ORDER BY i.stay_id, i.intime
    """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return {r["stay_id"]: float(r["pre_icu_los"]) for r in rows}


def fetch_gcs_total(window_hours: int | None = None) -> dict[int, int]:
    """GCS total score (E+V+M) — first recorded values within 1h of intime."""
    source = get_data_source()
    if source == "mock":
        return {}
    sql = f"""
        WITH eye AS (
            SELECT DISTINCT ON (c.stay_id) c.stay_id, c.valuenum
            FROM mimiciv_icu.chartevents c
            JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
            WHERE c.itemid = 220739 AND c.valuenum IS NOT NULL
              AND c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
            ORDER BY c.stay_id, c.charttime
        ),
        verbal AS (
            SELECT DISTINCT ON (c.stay_id) c.stay_id, c.valuenum
            FROM mimiciv_icu.chartevents c
            JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
            WHERE c.itemid = 223900 AND c.valuenum IS NOT NULL
              AND c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
            ORDER BY c.stay_id, c.charttime
        ),
        motor AS (
            SELECT DISTINCT ON (c.stay_id) c.stay_id, c.valuenum
            FROM mimiciv_icu.chartevents c
            JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
            WHERE c.itemid = 223901 AND c.valuenum IS NOT NULL
              AND c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
            ORDER BY c.stay_id, c.charttime
        )
        SELECT
            i.stay_id,
            (COALESCE(e.valuenum, 0) + COALESCE(v.valuenum, 0) + COALESCE(m.valuenum, 0)) AS gcs_total
        FROM mimiciv_icu.icustays i
        LEFT JOIN eye e ON i.stay_id = e.stay_id
        LEFT JOIN verbal v ON i.stay_id = v.stay_id
        LEFT JOIN motor m ON i.stay_id = m.stay_id
    """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return {r["stay_id"]: int(r["gcs_total"]) for r in rows}


def fetch_vasopressor_1h(window_hours: int | None = None) -> dict[int, int]:
    """Whether any vasopressor was administered within 1h of ICU admission."""
    source = get_data_source()
    if source == "mock":
        return {}
    sql = f"""
        SELECT DISTINCT ON (i.stay_id)
               i.stay_id,
               CASE WHEN ie.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vasopressor_1h
        FROM mimiciv_icu.icustays i
        LEFT JOIN mimiciv_icu.inputevents ie
            ON i.stay_id = ie.stay_id
            AND ie.itemid IN (221906, 221289, 229617, 221662, 221653, 222315, 221749)
            AND ie.starttime >= i.intime AND ie.starttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
            AND ie.amount > 0
        ORDER BY i.stay_id, ie.starttime
    """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return {r["stay_id"]: int(r["vasopressor_1h"]) for r in rows}


# ── Wave A 扩展特征 ───────────────────────────────────────────────

# Elixhauser 合并症类别名称（30 类）
ELIXHAUSER_CATEGORIES = [
    "congestive_heart_failure",
    "cardiac_arrhythmias",
    "valvular_disease",
    "pulmonary_circulation",
    "peripheral_vascular",
    "hypertension_uncomplicated",
    "hypertension_complicated",
    "paralysis",
    "neurological_disorders",
    "chronic_pulmonary",
    "diabetes_uncomplicated",
    "diabetes_complicated",
    "hypothyroidism",
    "renal_failure",
    "liver_disease",
    "peptic_ulcer",
    "lymphoma",
    "metastatic_cancer",
    "solid_tumor",
    "rheumatoid_arthritis",
    "coagulopathy",
    "obesity",
    "weight_loss",
    "fluid_electrolyte",
    "blood_loss_anemia",
    "deficiency_anemia",
    "alcohol_abuse",
    "drug_abuse",
    "psychoses",
    "depression",
]

# ICD-10 → Elixhauser 映射（Quan 2005 算法，AHRQ 2024 版本）
# 每个 key 是类别名，value 是 ICD-10 前缀列表
ELIXHAUSER_ICD10_MAP: dict[str, list[str]] = {
    "congestive_heart_failure": [
        "I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427",
        "I428", "I429", "I43", "I50", "P290",
    ],
    "cardiac_arrhythmias": [
        "I441", "I442", "I443", "I456", "I459", "I47", "I48", "I49",
        "R000", "R001", "R008", "T821", "Z450", "Z950",
    ],
    "valvular_disease": [
        "A520", "I05", "I06", "I07", "I08", "I091", "I098",
        "I34", "I35", "I36", "I37", "I38", "I39",
        "Q230", "Q231", "Q232", "Q233", "Z952", "Z953", "Z954",
    ],
    "pulmonary_circulation": [
        "I26", "I27", "I280", "I288", "I289",
    ],
    "peripheral_vascular": [
        "I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792",
        "K551", "K558", "K559", "Z958", "Z959",
    ],
    "hypertension_uncomplicated": ["I10"],
    "hypertension_complicated": ["I11", "I12", "I13", "I15"],
    "paralysis": [
        "G041", "G114", "G801", "G802", "G81", "G82", "G830", "G831",
        "G832", "G833", "G834", "G839",
    ],
    "neurological_disorders": [
        "G10", "G11", "G12", "G13", "G20", "G21", "G22", "G254", "G255",
        "G312", "G318", "G319", "G32", "G35", "G36", "G37",
        "G40", "G41", "G931", "G934", "R470", "R56",
    ],
    "chronic_pulmonary": [
        "I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47",
        "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67",
        "J684", "J701", "J703",
    ],
    "diabetes_uncomplicated": [
        "E100", "E101", "E109", "E110", "E111", "E119",
        "E120", "E121", "E129", "E130", "E131", "E139",
        "E140", "E141", "E149",
    ],
    "diabetes_complicated": [
        "E102", "E103", "E104", "E105", "E106", "E107", "E108",
        "E112", "E113", "E114", "E115", "E116", "E117", "E118",
        "E122", "E123", "E124", "E125", "E126", "E127", "E128",
        "E132", "E133", "E134", "E135", "E136", "E137", "E138",
        "E142", "E143", "E144", "E145", "E146", "E147", "E148",
    ],
    "hypothyroidism": ["E00", "E01", "E02", "E03", "E890"],
    "renal_failure": [
        "I120", "I131", "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940", "Z992",
    ],
    "liver_disease": [
        "B18", "I85", "I864", "I982",
        "K70", "K711", "K713", "K714", "K715", "K717",
        "K72", "K73", "K74", "K760", "K762", "K763", "K764", "K765",
        "K766", "K767", "K768", "K769", "Z944",
    ],
    "peptic_ulcer": ["K257", "K259", "K267", "K269", "K277", "K279", "K287", "K289"],
    "lymphoma": ["C81", "C82", "C83", "C84", "C85", "C88", "C96", "C900", "C902"],
    "metastatic_cancer": ["C77", "C78", "C79", "C80"],
    "solid_tumor": [
        "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08",
        "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17",
        "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
        "C30", "C31", "C32", "C33", "C34",
        "C37", "C38", "C39", "C40", "C41",
        "C43", "C45", "C46", "C47", "C48", "C49", "C50",
        "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58",
        "C60", "C61", "C62", "C63", "C64", "C65", "C66", "C67",
        "C68", "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76",
        "C97",
    ],
    "rheumatoid_arthritis": [
        "L940", "L941", "L943", "M05", "M06", "M08", "M120", "M123",
        "M30", "M310", "M311", "M312", "M313",
        "M32", "M33", "M34", "M35", "M45", "M461", "M468", "M469",
    ],
    "coagulopathy": ["D65", "D66", "D67", "D68", "D691", "D693", "D694", "D695", "D696"],
    "obesity": ["E66"],
    "weight_loss": ["E40", "E41", "E42", "E43", "E44", "E45", "E46", "R634", "R64"],
    "fluid_electrolyte": ["E222", "E86", "E87"],
    "blood_loss_anemia": ["D500"],
    "deficiency_anemia": ["D508", "D509", "D51", "D52", "D53"],
    "alcohol_abuse": [
        "F10", "E52", "G621", "I426", "K292", "K700", "K703", "K709",
        "T51", "Z502", "Z714", "Z721",
    ],
    "drug_abuse": ["F11", "F12", "F13", "F14", "F15", "F16", "F18", "F19", "Z715", "Z722"],
    "psychoses": ["F20", "F22", "F23", "F24", "F25", "F28", "F29", "F302", "F312", "F315"],
    "depression": ["F204", "F313", "F314", "F315", "F32", "F33", "F341", "F412", "F432"],
}


def _elixhauser_sql() -> str:
    """Generate SQL for Elixhauser comorbidity flags from diagnoses_icd."""
    cases = []
    for cat, prefixes in ELIXHAUSER_ICD10_MAP.items():
        conditions = " OR ".join(f"d.icd_code LIKE '{p}%'" for p in prefixes)
        cases.append(f"MAX(CASE WHEN ({conditions}) AND d.icd_version = 10 THEN 1 ELSE 0 END) AS elix_{cat}")
    return ",\n        ".join(cases)


def fetch_elixhauser() -> dict[int, dict[str, int]]:
    """Elixhauser comorbidity flags (30 categories) from diagnoses_icd.
    All ICD codes are present-on-admission → no leakage risk.
    """
    source = get_data_source()
    if source == "mock":
        return {}
    case_clauses = _elixhauser_sql()
    sql = f"""
        SELECT
            i.stay_id,
            {case_clauses}
        FROM mimiciv_icu.icustays i
        LEFT JOIN mimiciv_hosp.diagnoses_icd d ON i.hadm_id = d.hadm_id
        GROUP BY i.stay_id
    """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    result: dict[int, dict[str, int]] = {}
    for r in rows:
        sid = r["stay_id"]
        result[sid] = {f"elix_{cat}": int(r[f"elix_{cat}"]) for cat in ELIXHAUSER_CATEGORIES}
    return result


def fetch_sofa_components(window_hours: int | None = None) -> dict[int, dict[str, int]]:
    """SOFA component scores (0-4 each) + total, calculated in Python from raw values.

    Instead of a monster 8-CTE SQL, we fetch raw component values via
    lightweight queries and calculate SOFA scores in Python.
    This avoids scanning chartevents 8 times in a single query.
    """
    source = get_data_source()
    if source == "mock":
        return {}

    # Step 1: Fetch raw values in separate lightweight queries
    # PaO2 (itemid 50821) - first value within 1h
    pao2_map: dict[int, float] = {}
    # FiO2 (itemid 223835) - first value within 1h
    fio2_map: dict[int, float] = {}
    # MAP (itemid 220181) - first value within 1h
    map_map: dict[int, float] = {}
    # Platelets (itemid 51265) - first value within 1h
    plt_map: dict[int, float] = {}
    # Bilirubin (itemid 50885) - first value within 1h
    bili_map: dict[int, float] = {}
    # Creatinine (itemid 50912) - first value within 1h
    cr_map: dict[int, float] = {}
    # GCS components
    gcs_eye_map: dict[int, float] = {}
    gcs_verbal_map: dict[int, float] = {}
    gcs_motor_map: dict[int, float] = {}
    # Ventilation flag
    vent_map: dict[int, int] = {}
    # Vasopressor rates
    norepi_rate_map: dict[int, float] = {}
    epi_rate_map: dict[int, float] = {}
    dopa_rate_map: dict[int, float] = {}
    dobu_rate_map: dict[int, float] = {}

    with _read_engine().connect() as conn:
        # Chartevents: PaO2, FiO2, MAP, GCS
        sql_vitals = f"""
            SELECT c.stay_id, c.itemid, c.valuenum
            FROM mimiciv_icu.chartevents c
            JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
            WHERE c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
              AND c.valuenum IS NOT NULL
              AND c.itemid IN (50821, 223835, 220181, 220739, 223900, 223901)
        """
        rows = conn.execute(text(sql_vitals)).mappings().all()
        for r in rows:
            sid, itemid, val = r["stay_id"], r["itemid"], float(r["valuenum"])
            if itemid == 50821:
                pao2_map.setdefault(sid, val)
            elif itemid == 223835:
                # Normalize FiO2: if > 1.0, assume percentage → divide by 100
                fio2_val = val / 100.0 if val > 1.0 else val
                fio2_map.setdefault(sid, fio2_val)
            elif itemid == 220181:
                map_map.setdefault(sid, val)
            elif itemid == 220739:
                gcs_eye_map.setdefault(sid, val)
            elif itemid == 223900:
                gcs_verbal_map.setdefault(sid, val)
            elif itemid == 223901:
                gcs_motor_map.setdefault(sid, val)

        # Ventilation flag (any vent settings within 1h)
        sql_vent = f"""
            SELECT DISTINCT c.stay_id
            FROM mimiciv_icu.chartevents c
            JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
            WHERE c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
              AND c.itemid IN (223848, 223849, 224684, 224685, 224686, 224687)
        """
        rows = conn.execute(text(sql_vent)).mappings().all()
        for r in rows:
            vent_map[r["stay_id"]] = 1

        # Labevents: Platelets, Bilirubin, Creatinine
        sql_labs = f"""
            SELECT i.stay_id, l.itemid, l.valuenum
            FROM mimiciv_icu.icustays i
            JOIN mimiciv_hosp.labevents l ON i.hadm_id = l.hadm_id
            WHERE l.charttime >= i.intime AND l.charttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
              AND l.valuenum IS NOT NULL
              AND l.itemid IN (51265, 50885, 50912)
        """
        rows = conn.execute(text(sql_labs)).mappings().all()
        for r in rows:
            sid, itemid, val = r["stay_id"], r["itemid"], float(r["valuenum"])
            if itemid == 51265:
                plt_map.setdefault(sid, val)
            elif itemid == 50885:
                bili_map.setdefault(sid, val)
            elif itemid == 50912:
                cr_map.setdefault(sid, val)

        # Vasopressor rates within 1h
        sql_vaso = f"""
            SELECT ie.stay_id, ie.itemid, COALESCE(ie.rate, 0) AS rate
            FROM mimiciv_icu.inputevents ie
            JOIN mimiciv_icu.icustays i ON ie.stay_id = i.stay_id
            WHERE ie.starttime >= i.intime AND ie.starttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
              AND ie.amount > 0
              AND ie.itemid IN (221906, 221289, 221662, 221653)
        """
        rows = conn.execute(text(sql_vaso)).mappings().all()
        for r in rows:
            sid, itemid, rate = r["stay_id"], r["itemid"], float(r["rate"])
            if itemid == 221906:
                norepi_rate_map.setdefault(sid, rate)
            elif itemid == 221289:
                epi_rate_map.setdefault(sid, rate)
            elif itemid == 221662:
                dopa_rate_map.setdefault(sid, rate)
            elif itemid == 221653:
                dobu_rate_map.setdefault(sid, rate)

    # Step 2: Get all stay_ids
    with _read_engine().connect() as conn:
        stay_ids = [r["stay_id"] for r in conn.execute(
            text("SELECT stay_id FROM mimiciv_icu.icustays")
        ).mappings().all()]

    # Step 3: Calculate SOFA scores in Python
    result: dict[int, dict[str, int]] = {}
    for sid in stay_ids:
        # Respiration SOFA
        pao2 = pao2_map.get(sid)
        fio2 = fio2_map.get(sid)
        is_vent = vent_map.get(sid, 0)
        if pao2 is not None and fio2 is not None and fio2 > 0:
            pf = pao2 / fio2
            if pf < 100 and is_vent:
                sofa_resp = 4
            elif pf < 200 and is_vent:
                sofa_resp = 3
            elif pf < 300:
                sofa_resp = 2
            elif pf < 400:
                sofa_resp = 1
            else:
                sofa_resp = 0
        else:
            sofa_resp = 0

        # Coagulation SOFA
        plt = plt_map.get(sid)
        if plt is not None:
            if plt < 20:
                sofa_coag = 4
            elif plt < 50:
                sofa_coag = 3
            elif plt < 100:
                sofa_coag = 2
            elif plt < 150:
                sofa_coag = 1
            else:
                sofa_coag = 0
        else:
            sofa_coag = 0

        # Liver SOFA
        bili = bili_map.get(sid)
        if bili is not None:
            if bili >= 12.0:
                sofa_liver = 4
            elif bili >= 6.0:
                sofa_liver = 3
            elif bili >= 2.0:
                sofa_liver = 2
            elif bili >= 1.2:
                sofa_liver = 1
            else:
                sofa_liver = 0
        else:
            sofa_liver = 0

        # Cardiovascular SOFA
        norepi = norepi_rate_map.get(sid, 0)
        epi = epi_rate_map.get(sid, 0)
        dopa = dopa_rate_map.get(sid, 0)
        dobu = dobu_rate_map.get(sid, 0)
        map_v = map_map.get(sid)
        if norepi > 0.1 or epi > 0.1:
            sofa_cardio = 4
        elif dopa > 5 or norepi > 0 or epi > 0:
            sofa_cardio = 3
        elif dopa > 0 or dobu > 0:
            sofa_cardio = 2
        elif map_v is not None and map_v < 70:
            sofa_cardio = 1
        else:
            sofa_cardio = 0

        # Neurological SOFA
        gcs = (gcs_eye_map.get(sid, 0) + gcs_verbal_map.get(sid, 0)
               + gcs_motor_map.get(sid, 0))
        if gcs < 6:
            sofa_neuro = 4
        elif gcs < 10:
            sofa_neuro = 3
        elif gcs < 13:
            sofa_neuro = 2
        elif gcs < 15:
            sofa_neuro = 1
        else:
            sofa_neuro = 0

        # Renal SOFA (simplified: creatinine only)
        cr = cr_map.get(sid)
        if cr is not None:
            if cr >= 5.0:
                sofa_renal = 4
            elif cr >= 3.5:
                sofa_renal = 3
            elif cr >= 2.0:
                sofa_renal = 2
            elif cr >= 1.2:
                sofa_renal = 1
            else:
                sofa_renal = 0
        else:
            sofa_renal = 0

        sofa_total = sofa_resp + sofa_coag + sofa_liver + sofa_cardio + sofa_neuro + sofa_renal
        result[sid] = {
            "sofa_respiration": sofa_resp,
            "sofa_coagulation": sofa_coag,
            "sofa_liver": sofa_liver,
            "sofa_cardiovascular": sofa_cardio,
            "sofa_neurological": sofa_neuro,
            "sofa_renal": sofa_renal,
            "sofa_total": sofa_total,
        }

    return result


def fetch_abg_first(window_hours: int | None = None) -> dict[int, dict[str, float]]:
    """Arterial blood gas: PaO2, PaCO2, FiO2, PaO2/FiO2 ratio, intubation flag.
    Only arterial specimens, first value within 1h of intime.
    """
    source = get_data_source()
    if source == "mock":
        return {}
    sql = f"""
    WITH arterial_bg AS (
        SELECT DISTINCT ON (c.stay_id) c.stay_id, c.charttime
        FROM mimiciv_icu.chartevents c
        JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
        WHERE c.itemid = 52033
          AND c.value = 'ART.'
          AND c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
        ORDER BY c.stay_id, c.charttime
    ),
    pao2_first AS (
        SELECT DISTINCT ON (c.stay_id) c.stay_id, c.valuenum AS pao2
        FROM mimiciv_icu.chartevents c
        JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
        WHERE c.itemid = 50821 AND c.valuenum IS NOT NULL
          AND c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
        ORDER BY c.stay_id, c.charttime
    ),
    paco2_first AS (
        SELECT DISTINCT ON (c.stay_id) c.stay_id, c.valuenum AS paco2
        FROM mimiciv_icu.chartevents c
        JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
        WHERE c.itemid = 50818 AND c.valuenum IS NOT NULL
          AND c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
        ORDER BY c.stay_id, c.charttime
    ),
    fio2_lab AS (
        SELECT DISTINCT ON (l.hadm_id) l.hadm_id, l.valuenum AS abg_fio2
        FROM mimiciv_hosp.labevents l
        WHERE l.itemid = 50816 AND l.valuenum IS NOT NULL
        ORDER BY l.hadm_id, l.charttime
    ),
    intub_flag AS (
        SELECT DISTINCT ON (c.stay_id) c.stay_id,
               CASE WHEN c.valuenum IS NOT NULL AND c.valuenum > 0 THEN 1 ELSE 0 END AS intubation_flag
        FROM mimiciv_icu.chartevents c
        JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
        WHERE c.itemid = 50812
          AND c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
        ORDER BY c.stay_id, c.charttime
    )
    SELECT
        i.stay_id,
        p.pao2,
        pc.paco2,
        fl.abg_fio2,
        CASE WHEN p.pao2 IS NOT NULL AND fl.abg_fio2 IS NOT NULL AND fl.abg_fio2 > 0
             THEN p.pao2 / fl.abg_fio2 ELSE NULL END AS pf_ratio,
        COALESCE(itf.intubation_flag, 0) AS intubation_flag
    FROM mimiciv_icu.icustays i
    LEFT JOIN pao2_first p ON i.stay_id = p.stay_id
    LEFT JOIN paco2_first pc ON i.stay_id = pc.stay_id
    LEFT JOIN fio2_lab fl ON i.hadm_id = fl.hadm_id
    LEFT JOIN intub_flag itf ON i.stay_id = itf.stay_id
    """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    result: dict[int, dict[str, float]] = {}
    for r in rows:
        sid = r["stay_id"]
        d: dict[str, float] = {}
        for key in ("pao2", "paco2", "abg_fio2", "pf_ratio"):
            d[key] = float(r[key]) if r[key] is not None else None  # type: ignore[assignment]
        d["intubation_flag"] = int(r["intubation_flag"])
        result[sid] = d
    return result


def fetch_gcs_subscores(window_hours: int | None = None) -> dict[int, dict[str, int]]:
    """GCS sub-scores (Eye/Verbal/Motor) separately, first value within 1h of intime."""
    source = get_data_source()
    if source == "mock":
        return {}
    sql = f"""
    WITH eye AS (
        SELECT DISTINCT ON (c.stay_id) c.stay_id, c.valuenum
        FROM mimiciv_icu.chartevents c
        JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
        WHERE c.itemid = 220739 AND c.valuenum IS NOT NULL
          AND c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
        ORDER BY c.stay_id, c.charttime
    ),
    verbal AS (
        SELECT DISTINCT ON (c.stay_id) c.stay_id, c.valuenum
        FROM mimiciv_icu.chartevents c
        JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
        WHERE c.itemid = 223900 AND c.valuenum IS NOT NULL
          AND c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
        ORDER BY c.stay_id, c.charttime
    ),
    motor AS (
        SELECT DISTINCT ON (c.stay_id) c.stay_id, c.valuenum
        FROM mimiciv_icu.chartevents c
        JOIN mimiciv_icu.icustays i ON c.stay_id = i.stay_id
        WHERE c.itemid = 223901 AND c.valuenum IS NOT NULL
          AND c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
        ORDER BY c.stay_id, c.charttime
    )
    SELECT
        i.stay_id,
        COALESCE(e.valuenum, 0) AS gcs_eye,
        COALESCE(v.valuenum, 0) AS gcs_verbal,
        COALESCE(m.valuenum, 0) AS gcs_motor
    FROM mimiciv_icu.icustays i
    LEFT JOIN eye e ON i.stay_id = e.stay_id
    LEFT JOIN verbal v ON i.stay_id = v.stay_id
    LEFT JOIN motor m ON i.stay_id = m.stay_id
    """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return {r["stay_id"]: {
        "gcs_eye": int(r["gcs_eye"]),
        "gcs_verbal": int(r["gcs_verbal"]),
        "gcs_motor": int(r["gcs_motor"]),
    } for r in rows}


def fetch_vent_flag(window_hours: int | None = None) -> dict[int, int]:
    """Mechanical ventilation flag within 1h of intime."""
    source = get_data_source()
    if source == "mock":
        return {}
    sql = f"""
    SELECT DISTINCT ON (i.stay_id)
           i.stay_id,
           CASE WHEN v.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vent_flag
    FROM mimiciv_icu.icustays i
    LEFT JOIN mimiciv_icu.chartevents v
        ON i.stay_id = v.stay_id
        AND v.itemid IN (223848, 223849, 224684, 224685, 224686, 224687, 224688, 224689, 224690, 224691)
        AND v.charttime >= i.intime AND v.charttime < i.intime + INTERVAL '{_wh(window_hours)} hours'
    ORDER BY i.stay_id
    """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return {r["stay_id"]: int(r["vent_flag"]) for r in rows}