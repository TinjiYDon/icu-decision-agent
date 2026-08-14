"""SHAP 结构化：将 L3 predict_stay() 的原始输出转为解释层可用的结构化数据。

输入（L3 predict_stay 的 top_factors）：
    [{"feature": "lab_lactate", "value": 2.3, "shap": 0.0823}, ...]

输出：
    [{
        "feature": "lab_lactate",
        "standard_name": "Lactate（乳酸）",
        "value": 2.3,
        "unit": "mmol/L",
        "shap": 0.0823,
        "shap_direction": "positive",
        "shap_abs": 0.0823,
        "normal_range": [0.5, 1.7],
        "critical_high": 4.0,
        "critical_low": null,
        "evidence_level": "A",
        "is_abnormal": True,
        "abnormal_note": "高于正常范围上限",
    }, ...]
"""

from __future__ import annotations

from typing import Any

from domain.explain.config import get_feature_meta


def _classify_value(value: Any, normal_range: list | None, critical_low: Any, critical_high: Any) -> tuple[bool, str]:
    """判断实测值是否异常，返回 (is_abnormal, abnormal_note)。"""
    if value is None or normal_range is None:
        return False, ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, ""
    low, high = normal_range[0], normal_range[1]
    if critical_low is not None and v <= float(critical_low):
        return True, f"低于危急值 {critical_low}"
    if critical_high is not None and v >= float(critical_high):
        return True, f"达到或超过危急值 {critical_high}"
    if v < low:
        return True, f"低于正常范围下限 {low}"
    if v > high:
        return True, f"高于正常范围上限 {high}"
    return False, "在正常范围内"


def shap_to_structured(
    top_factors: list[dict[str, Any]],
    features_display: dict[str, Any] | None = None,
    max_factors: int = 4,
) -> dict[str, Any]:
    """将 SHAP 原始输出结构化。

    Args:
        top_factors: L3 predict_stay() 返回的 top_factors 列表
        features_display: 可选，完整特征值字典（用于补充未在 top_factors 中的特征）
        max_factors: 最多解释多少个特征

    Returns:
        {
            "factors": [结构化特征列表],
            "coverage_pct": float,  # 累计 SHAP 绝对值占比
            "total_shap_abs": float,
        }
    """
    meta = get_feature_meta()
    features_meta = meta.get("features", {})
    denied = set(meta.get("denied", []))

    # 过滤 denied 字段，保留 api_allowed=true 的特征
    filtered: list[dict[str, Any]] = []
    for f in top_factors:
        name = f.get("feature", "")
        if name in denied:
            continue
        fmeta = features_meta.get(name, {})
        if not fmeta.get("api_allowed", True):
            continue
        filtered.append(f)

    # 按 SHAP 绝对值降序
    filtered.sort(key=lambda x: abs(float(x.get("shap", 0) or 0)), reverse=True)
    filtered = filtered[:max_factors]

    structured_factors: list[dict[str, Any]] = []
    total_shap_abs = sum(abs(float(f.get("shap", 0) or 0)) for f in filtered)

    for f in filtered:
        name = f.get("feature", "")
        value = f.get("value")
        shap_val = float(f.get("shap", 0) or 0)
        fmeta = features_meta.get(name, {})

        standard_name = fmeta.get("standard_name", name)
        unit = fmeta.get("unit", "")
        normal_range = fmeta.get("normal_range")
        critical_low = fmeta.get("critical_low")
        critical_high = fmeta.get("critical_high")
        evidence_level = fmeta.get("evidence_level", "C")
        clinical_note = fmeta.get("clinical_note", "")

        is_abnormal, abnormal_note = _classify_value(value, normal_range, critical_low, critical_high)

        structured_factors.append({
            "feature": name,
            "standard_name": standard_name,
            "value": value,
            "unit": unit,
            "shap": round(shap_val, 4),
            "shap_abs": round(abs(shap_val), 4),
            "shap_direction": "positive" if shap_val >= 0 else "negative",
            "normal_range": normal_range,
            "critical_low": critical_low,
            "critical_high": critical_high,
            "evidence_level": evidence_level,
            "clinical_note": clinical_note,
            "is_abnormal": is_abnormal,
            "abnormal_note": abnormal_note,
        })

    # 计算 coverage_pct（已解释特征的累计 SHAP 绝对值占比）
    # 注意：这是相对于 top_factors 本身的占比，不是全特征占比
    if total_shap_abs > 0:
        coverage_pct = round(100.0 * sum(sf["shap_abs"] for sf in structured_factors) / total_shap_abs, 1)
    else:
        coverage_pct = 0.0

    return {
        "factors": structured_factors,
        "coverage_pct": coverage_pct,
        "total_shap_abs": round(total_shap_abs, 4),
    }


def format_factors_for_prompt(structured: dict[str, Any]) -> str:
    """将结构化特征格式化为 Prompt 文本。"""
    lines: list[str] = []
    for i, sf in enumerate(structured["factors"], 1):
        value_str = f"{sf['value']}" if sf["value"] is not None else "缺失"
        abnormal_str = f"（{sf['abnormal_note']}）" if sf["is_abnormal"] else ""
        normal_str = ""
        if sf["normal_range"]:
            normal_str = f"正常范围 {sf['normal_range'][0]}-{sf['normal_range'][1]} {sf['unit']}"
        lines.append(
            f"{i}. {sf['standard_name']}（{sf['feature']}）\n"
            f"   实测值: {value_str} {sf['unit']}{abnormal_str}\n"
            f"   SHAP贡献: {sf['shap']}（{'升高风险' if sf['shap_direction']=='positive' else '降低风险'}）\n"
            f"   {normal_str}"
        )
    cov = structured["coverage_pct"]
    lines.append(f"\n（已解释特征累计SHAP贡献占比：{cov}%）")
    return "\n".join(lines)
