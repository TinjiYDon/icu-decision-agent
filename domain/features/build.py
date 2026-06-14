"""Build tabular features from Layer0 cohort → feat.sample_matrix."""

from __future__ import annotations

import json

from sqlalchemy import text

from data_access.mimic_repo import fetch_cohort
from infra.db import get_engine


def _gender_m(g: str | None) -> int:
    return 1 if (g or "").upper() == "M" else 0


def build_features() -> dict:
    rows = fetch_cohort()
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE feat.sample_matrix"))
        for row in rows:
            age = int(row["anchor_age"] or 0)
            los = float(row["los_hours"] or 0.0)
            feat = {
                "anchor_age": age,
                "gender_m": _gender_m(row.get("gender")),
                "los_hours": round(los, 3),
                "hospital_expire_flag": int(row.get("hospital_expire_flag") or 0),
            }
            conn.execute(
                text(
                    """
                    INSERT INTO feat.sample_matrix (stay_id, hour_index, feature_json)
                    VALUES (:stay_id, 0, CAST(:feature_json AS jsonb))
                    """
                ),
                {
                    "stay_id": row["stay_id"],
                    "feature_json": json.dumps(feat, ensure_ascii=False),
                },
            )
    return {"feat_rows": len(rows)}
