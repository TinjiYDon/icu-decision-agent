"""HITL summarize unit test."""

from __future__ import annotations

import json

from application.summarize_hitl import summarize_hitl


def test_summarize_hitl_empty(tmp_path, monkeypatch):
    log = tmp_path / "empty.jsonl"
    monkeypatch.setattr("application.summarize_hitl.audit_path", lambda: log)
    out = summarize_hitl(days=7, log_path=log)
    assert out["status"] == "empty"
    assert out["n"] == 0


def test_summarize_hitl_rates(tmp_path):
    log = tmp_path / "audit.jsonl"
    rows = [
        {"ts": "2099-01-01T00:00:00+00:00", "decision": "accept", "recommend": "observe"},
        {"ts": "2099-01-01T01:00:00+00:00", "decision": "reject", "recommend": "escalate"},
        {"ts": "2099-01-01T02:00:00+00:00", "decision": "accept", "recommend": "monitor"},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = summarize_hitl(days=None, log_path=log)
    assert out["n"] == 3
    assert out["accept_rate"] == round(2 / 3, 4)
    assert out["reject_rate"] == round(1 / 3, 4)
