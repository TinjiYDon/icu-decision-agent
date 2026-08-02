"""12-hour mortality label from prediction time t (S1: t=intime+1h)."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import text

from data_access.mimic_repo import fetch_cohort
from infra.config import load_yaml
from infra.db import get_engine


def _death_within_window(window_start: datetime, window_end: datetime, dod) -> int:
    if dod is None or window_start is None:
        return 0
    if hasattr(dod, "year") and not hasattr(dod, "hour"):
        death_end = datetime.combine(dod, datetime.max.time().replace(microsecond=0))
        death_start = datetime.combine(dod, datetime.min.time())
    else:
        death_start = death_end = dod
    if death_end < window_start or death_start > window_end:
        return 0
    return 1


def build_labels() -> dict:
    cfg = load_yaml("labels.yaml").get("primary", {})
    feat_cfg = load_yaml("features.yaml")
    horizon = int(cfg.get("horizon_hours", 12))
    offset = int(feat_cfg.get("prediction_offset_hours", 1))
    hour_index = int(feat_cfg.get("hour_index", offset))
    rows = fetch_cohort()
    pos = 0
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE label.mortality_12h"))
        for row in rows:
            intime = row["intime"]
            if intime is None:
                label = 0
            else:
                t = intime + timedelta(hours=offset)
                label = _death_within_window(t, t + timedelta(hours=horizon), row.get("dod"))
            pos += label
            conn.execute(
                text(
                    """
                    INSERT INTO label.mortality_12h (stay_id, hour_index, label)
                    VALUES (:stay_id, :hour_index, :label)
                    """
                ),
                {"stay_id": row["stay_id"], "hour_index": hour_index, "label": label},
            )
    return {
        "label_rows": len(rows),
        "positive": pos,
        "horizon_hours": horizon,
        "prediction_offset_hours": offset,
        "hour_index": hour_index,
    }
