"""L4 care plan entry (D3)."""

from __future__ import annotations

from typing import Any

from application.predict_patient import predict_patient, predict_patient_trajectory
from domain.planning.care_plan import build_care_plan


def _band_from_recommend(recommend: Any) -> str:
    if isinstance(recommend, dict):
        return str(recommend.get("band") or recommend.get("action") or "observe")
    return str(recommend or "observe")


def care_plan_for_stay(
    stay_id: int,
    *,
    hour_index: int | None = None,
    model_type: str = "lgbm",
) -> dict[str, Any]:
    pred = predict_patient(int(stay_id), hour_index=hour_index, model_type=model_type)
    if pred.get("status") != "ok" and pred.get("risk_score") is None:
        return {"status": "care_plan_unavailable", "detail": pred}
    traj: list[dict[str, Any]] = []
    try:
        traj_raw = predict_patient_trajectory(int(stay_id))
        if isinstance(traj_raw, dict):
            points = traj_raw.get("points") or traj_raw.get("trajectory") or traj_raw.get("hours")
            if isinstance(points, list):
                traj = points
    except Exception:
        traj = []
    plan = build_care_plan(
        stay_id=int(stay_id),
        risk_score=float(pred.get("risk_score") or 0.0),
        recommend=_band_from_recommend(pred.get("recommend")),
        top_factors=list(pred.get("top_factors") or []),
        trajectory=traj,
    )
    plan["prediction"] = {
        "risk_score": pred.get("risk_score"),
        "recommend": pred.get("recommend"),
        "model_type": model_type,
        "status": pred.get("status"),
    }
    return plan
