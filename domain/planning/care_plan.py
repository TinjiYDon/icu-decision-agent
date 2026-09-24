"""In-repo care / disposition planning (D3) — NOT bed CP-SAT."""

from __future__ import annotations

from typing import Any


_BAND_ACTIONS: dict[str, list[dict[str, Any]]] = {
    "observe": [
        {"priority": 1, "action": "continue_routine_monitoring", "label_zh": "维持常规监护", "recheck_hours": 6},
        {"priority": 2, "action": "review_vitals_trend", "label_zh": "复查生命体征趋势", "recheck_hours": 4},
    ],
    "recheck": [
        {"priority": 1, "action": "repeat_labs", "label_zh": "复查关键化验", "recheck_hours": 2},
        {"priority": 2, "action": "increase_observation", "label_zh": "加密床旁观察", "recheck_hours": 2},
    ],
    "monitor": [
        {"priority": 1, "action": "escalate_monitoring", "label_zh": "加强连续监护", "recheck_hours": 1},
        {"priority": 2, "action": "notify_covering_clinician", "label_zh": "通知值班医师评估", "recheck_hours": 1},
        {"priority": 3, "action": "prepare_intervention_checklist", "label_zh": "准备升级处置核对清单", "recheck_hours": 2},
    ],
    "escalate": [
        {"priority": 1, "action": "urgent_clinical_review", "label_zh": "紧急床旁评估", "recheck_hours": 0.5},
        {"priority": 2, "action": "consider_higher_acuity", "label_zh": "评估更高监护级别/转出倾向", "recheck_hours": 1},
        {"priority": 3, "action": "document_rapid_response", "label_zh": "记录快速反应触发条件", "recheck_hours": 1},
    ],
}


def build_care_plan(
    *,
    stay_id: int,
    risk_score: float,
    recommend: str,
    top_factors: list[dict[str, Any]] | None = None,
    trajectory: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rule-first structured disposition plan; LLM may narrate separately."""
    band = (recommend or "observe").strip().lower()
    if band not in _BAND_ACTIONS:
        band = "observe"
    actions = [dict(a) for a in _BAND_ACTIONS[band]]
    drivers = []
    for f in (top_factors or [])[:5]:
        name = f.get("feature") or f.get("name") or f.get("factor")
        if name:
            drivers.append(str(name))
    escalate_if = []
    if risk_score >= 0.7:
        escalate_if.append("risk_score>=0.7")
    if trajectory and len(trajectory) >= 2:
        try:
            first = float(trajectory[0].get("risk_score", trajectory[0].get("score", 0)))
            last = float(trajectory[-1].get("risk_score", trajectory[-1].get("score", 0)))
            if last - first >= 0.15:
                escalate_if.append("risk_trajectory_rise>=0.15")
        except (TypeError, ValueError):
            pass
    return {
        "stay_id": int(stay_id),
        "band": band,
        "risk_score": float(risk_score),
        "actions": actions,
        "drivers": drivers,
        "escalate_triggers": escalate_if or ["clinician_judgment"],
        "disclaimer": "辅助建议，非医嘱；不执行床位分配。",
        "status": "care_plan_ok",
    }
