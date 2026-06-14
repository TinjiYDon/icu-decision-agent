"""P0 ETL：Layer0 icustays → staging + feat 占位."""

from __future__ import annotations

import json

from sqlalchemy import text

from data_access.mimic_repo import fetch_icustays
from infra.db import get_engine


def run_pipeline() -> dict:
    rows = fetch_icustays()
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE staging.icustays"))
        conn.execute(text("TRUNCATE feat.sample_matrix"))
        for row in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO staging.icustays (
                        stay_id, subject_id, hadm_id, first_careunit, last_careunit,
                        intime, outtime, los_hours
                    ) VALUES (
                        :stay_id, :subject_id, :hadm_id, :first_careunit, :last_careunit,
                        :intime, :outtime, :los_hours
                    )
                    """
                ),
                row,
            )
            conn.execute(
                text(
                    """
                    INSERT INTO feat.sample_matrix (stay_id, hour_index, feature_json)
                    VALUES (:stay_id, 0, CAST(:feature_json AS jsonb))
                    """
                ),
                {
                    "stay_id": row["stay_id"],
                    "feature_json": json.dumps(
                        {
                            "los_hours": row["los_hours"],
                            "first_careunit": row["first_careunit"],
                        },
                        ensure_ascii=False,
                    ),
                },
            )
    return {
        "staging_icustays": len(rows),
        "feat_rows": len(rows),
        "message": "staging + feat.sample_matrix (hour_index=0) loaded from Layer0",
    }
