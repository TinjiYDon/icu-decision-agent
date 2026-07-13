"""L4 API: list stays and predict mortality risk for one patient."""

from __future__ import annotations

from typing import Any

from data_access.mimic_repo import fetch_icustays
from domain.models.lgbm import predict_stay


def list_stays(limit: int = 200) -> list[dict[str, Any]]:
    """Return ICU stays for UI selection (no SQL in presentation layer)."""
    rows = fetch_icustays()
    return rows[:limit]


def predict_patient(stay_id: int) -> dict[str, Any]:
    """L4 contract: stay_id -> risk_score, top_factors, status."""
    return predict_stay(int(stay_id))
