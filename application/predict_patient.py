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


def predict_patient_with_explanation(
    stay_id: int,
    hour_index: int | None = None,
) -> dict[str, Any]:
    """L4 扩展接口：风险预测 + SHAP+LLM 可解释性报告。

    严格遵循分层契约（ADR-001）：L4 编排 L3 预测与 L3.5 解释，
    L3（lgbm.predict_stay）保持不变，L3.5 不反向调用 L3。

    Returns:
        在 predict_patient 输出基础上增加 "explanation" 字段：
        {
            ...原有预测字段...,
            "explanation": {
                "status": "ok" | "fallback",
                "explanation": str,      # Markdown 解释文本
                "structured": dict,      # 结构化 JSON
                "references": list,
                "disclaimers": list,
                "elapsed_ms": int,
            }
        }
    """
    # 延迟导入，避免 L4 在未配置 LLM 时整体不可用
    from domain.explain.shap_llm import generate_explanation

    prediction = predict_patient(int(stay_id), hour_index=hour_index)
    if prediction.get("status") != "ok":
        return prediction

    explanation = generate_explanation(
        stay_id=int(stay_id),
        hour_index=prediction["hour_index"],
        shap_output=prediction.get("top_factors", []),
        risk_score=prediction["risk_score"],
        recommendation=prediction.get("recommend", {}),
        features_display=prediction.get("features"),
    )
    prediction["explanation"] = explanation
    return prediction
