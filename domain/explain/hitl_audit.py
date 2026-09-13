"""HITL audit log for AIGC explanations (D2) — local JSONL, no scheduling couple."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "artifacts" / "hitl" / "explain_audit.jsonl"

Decision = Literal["accept", "reject", "edit"]


def audit_path() -> Path:
    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    return DEFAULT_LOG


def append_feedback(
    *,
    stay_id: int,
    decision: Decision,
    risk_score: float | None = None,
    recommend: str | None = None,
    reason: str = "",
    edited_text: str = "",
    model_type: str = "lgbm",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if decision not in ("accept", "reject", "edit"):
        raise ValueError(f"invalid decision: {decision}")
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stay_id": int(stay_id),
        "decision": decision,
        "risk_score": risk_score,
        "recommend": recommend,
        "reason": reason,
        "edited_text": edited_text,
        "model_type": model_type,
        **(extra or {}),
    }
    path = audit_path()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def list_recent(limit: int = 20) -> list[dict[str, Any]]:
    path = audit_path()
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-max(1, limit) :]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
