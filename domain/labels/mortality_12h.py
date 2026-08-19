"""12-hour mortality label from prediction time t (S1/S2: t=intime+h).

Prefer admissions.deathtime (timestamp) over date-level patients.dod.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import text

from data_access.mimic_repo import fetch_cohort
from domain.features.build import prediction_hour_index, prediction_hours
from infra.config import load_yaml
from infra.db import get_engine


def resolve_death_time(row: dict[str, Any]) -> tuple[Any, str]:
    """Return (death_instant_or_date, source_tag).

    Preference: deathtime (precise) → dod (date-level fallback) → None.
    """
    dt = row.get("deathtime")
    if dt is not None:
        return dt, "deathtime"
    dod = row.get("dod")
    if dod is not None:
        return dod, "dod"
    return None, "none"


def death_within_window(window_start: datetime, window_end: datetime, death) -> int:
    """1 if death overlaps [window_start, window_end], else 0."""
    if death is None or window_start is None:
        return 0
    # date-only (no hour) → treat as ambiguous full calendar day
    if isinstance(death, date) and not isinstance(death, datetime):
        death_end = datetime.combine(death, datetime.max.time().replace(microsecond=0))
        death_start = datetime.combine(death, datetime.min.time())
    elif hasattr(death, "year") and not hasattr(death, "hour"):
        death_end = datetime.combine(death, datetime.max.time().replace(microsecond=0))
        death_start = datetime.combine(death, datetime.min.time())
    else:
        death_start = death_end = death
    if death_end < window_start or death_start > window_end:
        return 0
    return 1


# Back-compat alias used by older imports/tests
_death_within_window = death_within_window


def build_labels() -> dict:
    cfg = load_yaml("labels.yaml").get("primary", {})
    horizon = int(cfg.get("horizon_hours", 12))
    prefer = str(cfg.get("death_time_prefer", "deathtime")).lower()
    hours = prediction_hours()
    rows = fetch_cohort()
    pos = 0
    n = 0
    src_counts = {"deathtime": 0, "dod": 0, "none": 0}
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE label.mortality_12h"))
        for h in hours:
            for row in rows:
                intime = row["intime"]
                death, src = resolve_death_time(row)
                if prefer == "dod":
                    death, src = row.get("dod"), ("dod" if row.get("dod") is not None else "none")
                src_counts[src] = src_counts.get(src, 0) + 1
                if intime is None:
                    label = 0
                else:
                    t = intime + timedelta(hours=h)
                    label = death_within_window(t, t + timedelta(hours=horizon), death)
                pos += label
                n += 1
                conn.execute(
                    text(
                        """
                        INSERT INTO label.mortality_12h (stay_id, hour_index, label)
                        VALUES (:stay_id, :hour_index, :label)
                        """
                    ),
                    {"stay_id": row["stay_id"], "hour_index": h, "label": label},
                )
    return {
        "label_rows": n,
        "n_stays": len(rows),
        "positive": pos,
        "horizon_hours": horizon,
        "prediction_hours": hours,
        "hour_index": prediction_hour_index(),
        "death_time_prefer": prefer,
        "death_time_source_counts": src_counts,
        "label_version": str(cfg.get("label_version", "mortality_12h_v2_deathtime")),
    }
