"""12-hour mortality label at ICU admission (hour_index=0)."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import text

from data_access.mimic_repo import fetch_cohort
from infra.config import load_yaml
from infra.db import get_engine


def _death_within_horizon(intime: datetime, dod, horizon_hours: int) -> int:
    if dod is None or intime is None:
        return 0
    if hasattr(dod, "year") and not hasattr(dod, "hour"):
        death_end = datetime.combine(dod, datetime.max.time().replace(microsecond=0))
        death_start = datetime.combine(dod, datetime.min.time())
    else:
        death_start = death_end = dod
    window_end = intime + timedelta(hours=horizon_hours)
    if death_end < intime or death_start > window_end:
        return 0
    return 1


def build_labels() -> dict:
    cfg = load_yaml("labels.yaml").get("primary", {})
    horizon = int(cfg.get("horizon_hours", 12))
    rows = fetch_cohort()
    pos = 0
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE label.mortality_12h"))
        for row in rows:
            label = _death_within_horizon(row["intime"], row.get("dod"), horizon)
            pos += label
            conn.execute(
                text(
                    """
                    INSERT INTO label.mortality_12h (stay_id, hour_index, label)
                    VALUES (:stay_id, 0, :label)
                    """
                ),
                {"stay_id": row["stay_id"], "label": label},
            )
    return {"label_rows": len(rows), "positive": pos, "horizon_hours": horizon}
