"""L4 API: list stays and predict mortality risk for one patient."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from data_access.mimic_repo import fetch_icustays
from domain.models.lgbm import predict_stay
from infra.config import load_yaml


@lru_cache(maxsize=1)
def get_label_config() -> dict[str, Any]:
    return load_yaml("labels.yaml")


@lru_cache(maxsize=8)
def list_stays(limit: int = 200) -> tuple[dict[str, Any], ...]:
    """Return ICU stays for UI selection (cached tuple for hashability)."""
    rows = fetch_icustays()
    return tuple(rows[:limit])


def predict_patient(stay_id: int) -> dict[str, Any]:
    """L4 contract: stay_id -> risk_score, top_factors, status."""
    return predict_stay(int(stay_id))
