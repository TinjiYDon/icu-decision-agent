"""L4 API: UI-specific queries for Streamlit pages."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import text

from infra.db import get_engine


@lru_cache(maxsize=1)
def list_demo_stays(limit: int = 2500) -> list[dict[str, int | float]]:
    """Stays with age + ≥2 labs at any hour (usable for demo)."""
    engine = get_engine()
    sql = """
        SELECT f.stay_id,
               MAX(COALESCE(s.los_hours, 0)) AS los_hours,
               MIN(f.hour_index) AS first_hour
        FROM feat.sample_matrix f
        LEFT JOIN staging.icustays s ON s.stay_id = f.stay_id
        WHERE f.hour_index = 1
          AND COALESCE((f.feature_json->>'anchor_age')::float, 0) > 0
          AND (
            (CASE WHEN COALESCE((f.feature_json->>'lab_creatinine')::float, 0) <> 0 THEN 1 ELSE 0 END)
          + (CASE WHEN COALESCE((f.feature_json->>'lab_hematocrit')::float, 0) <> 0 THEN 1 ELSE 0 END)
          + (CASE WHEN COALESCE((f.feature_json->>'lab_bun')::float, 0) <> 0 THEN 1 ELSE 0 END)
          + (CASE WHEN COALESCE((f.feature_json->>'lab_lactate')::float, 0) <> 0 THEN 1 ELSE 0 END)
          + (CASE WHEN COALESCE((f.feature_json->>'lab_sodium')::float, 0) <> 0 THEN 1 ELSE 0 END)
          ) >= 2
        GROUP BY f.stay_id
        ORDER BY f.stay_id
        LIMIT :lim
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"lim": int(limit)}).mappings().all()
    return [
        {
            "stay_id": int(r["stay_id"]),
            "los_hours": float(r["los_hours"] or 0),
            "first_hour": int(r["first_hour"] or 0),
        }
        for r in rows
    ]


@lru_cache(maxsize=128)
def list_hours_for_stay(stay_id: int) -> list[int]:
    """Available hour indices for a given stay_id."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT hour_index FROM feat.sample_matrix "
                "WHERE stay_id = :sid ORDER BY hour_index"
            ),
            {"sid": int(stay_id)},
        ).fetchall()
    return [int(r[0]) for r in rows]


@lru_cache(maxsize=1)
def list_all_stay_ids(limit: int = 500) -> list[int]:
    """All distinct stay_ids in sample_matrix (for explain page dropdown)."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT f.stay_id "
                "FROM feat.sample_matrix f "
                "ORDER BY f.stay_id LIMIT :lim"
            ),
            {"lim": int(limit)},
        ).mappings().all()
    return [int(r["stay_id"]) for r in rows]
