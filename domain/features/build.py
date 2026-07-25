"""Build admission-time features → feat.sample_matrix (no outcome leak)."""

from __future__ import annotations

import json
import re

from sqlalchemy import text

from data_access.mimic_repo import fetch_cohort
from infra.db import get_engine

FEATURE_COLS = [
    "anchor_age",
    "gender_m",
    "careunit_micu",
    "careunit_sicu",
    "careunit_ccu",
    "careunit_other",
]


def _gender_m(g: str | None) -> int:
    return 1 if (g or "").upper() == "M" else 0


def _careunit_flags(name: str | None) -> dict[str, int]:
    u = (name or "").upper()
    micu = 1 if "MICU" in u or re.search(r"\bMICU\b", u) else 0
    sicu = 1 if "SICU" in u else 0
    ccu = 1 if re.search(r"\bCCU\b", u) or "CORONARY" in u else 0
    if micu == 0 and sicu == 0 and ccu == 0:
        other = 1
    else:
        other = 0
    return {
        "careunit_micu": micu,
        "careunit_sicu": sicu,
        "careunit_ccu": ccu,
        "careunit_other": other,
    }


def row_to_features(row: dict) -> dict:
    feat = {
        "anchor_age": int(row.get("anchor_age") or 0),
        "gender_m": _gender_m(row.get("gender")),
    }
    feat.update(_careunit_flags(row.get("first_careunit")))
    return feat


def build_features() -> dict:
    rows = fetch_cohort()
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE feat.sample_matrix"))
        for row in rows:
            feat = row_to_features(row)
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
    return {"feat_rows": len(rows), "feature_cols": FEATURE_COLS}
