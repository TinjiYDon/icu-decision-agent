"""Acceptance gates for dump restore / metrics dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import text

from domain.features.build import prediction_hours
from infra.db import get_engine

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "artifacts" / "models" / "metrics_mortality_12h.json"
EXPECTED_FEAT_ROWS = 472290
EXPECTED_STAYS = 94458


def layer1_counts() -> dict[str, Any]:
    """Query Layer1 feat/label counts and hour distribution."""
    engine = get_engine()
    with engine.connect() as conn:
        feat = int(conn.execute(text("SELECT COUNT(*) FROM feat.sample_matrix")).scalar_one())
        label = int(conn.execute(text("SELECT COUNT(*) FROM label.mortality_12h")).scalar_one())
        stays = int(conn.execute(text("SELECT COUNT(*) FROM staging.icustays")).scalar_one())
        by_hour = {
            int(r[0]): int(r[1])
            for r in conn.execute(
                text(
                    "SELECT hour_index, COUNT(*) FROM feat.sample_matrix "
                    "GROUP BY hour_index ORDER BY hour_index"
                )
            )
        }
    hours = prediction_hours()
    gate_ok = (
        feat >= EXPECTED_FEAT_ROWS * 0.99
        and label >= EXPECTED_FEAT_ROWS * 0.99
        and stays >= EXPECTED_STAYS * 0.99
        and set(by_hour) >= set(hours)
    )
    return {
        "feat_rows": feat,
        "label_rows": label,
        "stay_count": stays,
        "by_hour": by_hour,
        "expected_feat_rows": EXPECTED_FEAT_ROWS,
        "expected_hours": hours,
        "gate_ok": gate_ok,
        "status": "pass" if gate_ok else "fail",
    }


def load_metrics_artifact() -> dict[str, Any] | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
