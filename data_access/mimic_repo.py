"""Read ICU stays from mock or Layer0 MIMIC."""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from infra.config import get_data_source, get_layer0_dsn, get_settings


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
        return conn.execute(text("SELECT COUNT(*) FROM mimiciv_icu.icustays")).scalar_one()


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
            SELECT stay_id, subject_id, hadm_id,
                   first_careunit, last_careunit,
                   intime, outtime, los AS los_hours
            FROM mimiciv_icu.icustays
            ORDER BY stay_id
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
                   0 AS hospital_expire_flag
            FROM mock.icustays i
            ORDER BY i.stay_id
        """
    else:
        sql = """
            SELECT i.stay_id, i.subject_id, i.hadm_id, i.intime, i.outtime, i.los AS los_hours,
                   p.anchor_age, p.gender, p.dod, a.hospital_expire_flag
            FROM mimiciv_icu.icustays i
            JOIN mimiciv_hosp.patients p ON i.subject_id = p.subject_id
            JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
            ORDER BY i.stay_id
        """
    with _read_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()
    return [dict(r) for r in rows]