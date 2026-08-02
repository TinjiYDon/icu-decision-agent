"""L4 API: list stays and predict mortality risk for one patient."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from data_access.mimic_repo import fetch_icustays
from domain.features.build import prediction_hour_index, prediction_hours
from domain.models.lgbm import predict_stay, predict_stay_trajectory
from infra.config import load_yaml


@lru_cache(maxsize=1)
def get_label_config() -> dict[str, Any]:
    return load_yaml("labels.yaml")


@lru_cache(maxsize=8)
def list_stays(limit: int = 200) -> tuple[dict[str, Any], ...]:
    """Return ICU stays for UI selection (cached tuple for hashability)."""
    rows = fetch_icustays()
    return tuple(rows[:limit])


def predict_patient(stay_id: int, hour_index: int | None = None) -> dict[str, Any]:
    """L4 contract: stay_id (+ optional hour_index) -> risk_score, top_factors, status."""
    h = prediction_hour_index() if hour_index is None else int(hour_index)
    return predict_stay(int(stay_id), hour_index=h)


def predict_patient_trajectory(stay_id: int) -> dict[str, Any]:
    """L4: multi-hour risk curve for S2 demo."""
    return predict_stay_trajectory(int(stay_id), hours=prediction_hours())
