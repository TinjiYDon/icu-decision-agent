from unittest.mock import patch

from presentation.mcp_tools import PREDICT_RISK_SCHEMA, predict_risk


def test_predict_risk_schema():
    assert PREDICT_RISK_SCHEMA["name"] == "predict_risk"
    assert "stay_id" in PREDICT_RISK_SCHEMA["inputSchema"]["properties"]
    assert PREDICT_RISK_SCHEMA["inputSchema"]["required"] == ["stay_id"]


def test_predict_risk_forwards_l4():
    fake = {
        "stay_id": 42,
        "status": "ok",
        "risk_score": 0.31,
        "score_kind": "probability",
        "recommend": {"band": "recheck", "label": "复查（中低风险）"},
        "top_factors": [{"feature": "los_hours", "shap": 0.1}],
    }
    with patch("presentation.mcp_tools.predict_patient", return_value=fake) as mocked:
        out = predict_risk(42)
    mocked.assert_called_once_with(42)
    assert out["status"] == "ok"
    assert out["recommend"]["band"] == "recheck"
    assert out["risk_score"] == 0.31
