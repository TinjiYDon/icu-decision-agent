"""MCP tool wrappers for L4 predict_patient (no MCP SDK required for unit tests)."""

from __future__ import annotations

import json
from typing import Any

from application.predict_patient import predict_patient
from domain.features.build import prediction_hour_index

PREDICT_RISK_SCHEMA: dict[str, Any] = {
    "name": "predict_risk",
    "description": (
        "Predict 12h ICU mortality risk for one stay_id at optional hour_index "
        "(S2 grid). Returns risk_score, recommend band, and SHAP top_factors via L4."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "stay_id": {"type": "integer", "description": "MIMIC icustay stay_id"},
            "hour_index": {
                "type": "integer",
                "description": "Hours after ICU intime (S2: 0/1/2/4/6). Default from config.",
            },
        },
        "required": ["stay_id"],
    },
}


def predict_risk(stay_id: int, hour_index: int | None = None) -> dict[str, Any]:
    """MCP tool body: stay_id (+ optional hour_index) -> L4 predict_patient JSON."""
    h = prediction_hour_index() if hour_index is None else int(hour_index)
    return predict_patient(int(stay_id), hour_index=h)


def predict_risk_json(stay_id: int, hour_index: int | None = None) -> str:
    return json.dumps(predict_risk(stay_id, hour_index=hour_index), ensure_ascii=False)
