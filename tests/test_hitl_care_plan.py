"""HITL audit + care plan unit tests."""

from __future__ import annotations

from domain.explain.hitl_audit import append_feedback, list_recent
from domain.planning.care_plan import build_care_plan


def test_hitl_append_and_list(tmp_path, monkeypatch):
    log = tmp_path / "audit.jsonl"
    monkeypatch.setattr("domain.explain.hitl_audit.DEFAULT_LOG", log)
    append_feedback(stay_id=1, decision="accept", risk_score=0.2, recommend="observe", reason="ok")
    append_feedback(stay_id=2, decision="reject", reason="disagree")
    rows = list_recent(10)
    assert len(rows) == 2
    assert rows[-1]["decision"] == "reject"


def test_care_plan_escalate_band():
    plan = build_care_plan(
        stay_id=99,
        risk_score=0.85,
        recommend="escalate",
        top_factors=[{"feature": "lactate", "shap": 0.1}],
        trajectory=[{"risk_score": 0.5}, {"risk_score": 0.8}],
    )
    assert plan["status"] == "care_plan_ok"
    assert plan["band"] == "escalate"
    assert any(a["action"] == "urgent_clinical_review" for a in plan["actions"])
    assert "lactate" in plan["drivers"]
    assert "risk_score>=0.7" in plan["escalate_triggers"]
