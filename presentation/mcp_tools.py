"""MCP tool wrappers for L4 predict_patient (no MCP SDK required for unit tests)."""

from __future__ import annotations

import json
from typing import Any

from application.predict_patient import predict_patient

PREDICT_RISK_SCHEMA: dict[str, Any] = {
    "name": "predict_risk",
    "description": (
        "Predict 12h ICU mortality risk for one stay_id. "
        "Returns risk_score, recommend band, and SHAP top_factors via L4."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "stay_id": {"type": "integer", "description": "MIMIC icustay stay_id"},
        },
        "required": ["stay_id"],
    },
}


def predict_risk(stay_id: int) -> dict[str, Any]:
    """MCP tool body: stay_id -> L4 predict_patient JSON."""
    return predict_patient(int(stay_id))


def predict_risk_json(stay_id: int) -> str:
    return json.dumps(predict_risk(stay_id), ensure_ascii=False)
