"""SHAP+LLM 解释层主入口：generate_explanation()。

流程：
    1. SHAP 原始值 → 结构化（特征元数据 + 异常判定 + 累计占比）
    2. 对每个特征做 RAG 检索（父子 chunk + 路由 + 元数据过滤）
    3. 构造 Prompt（<data> 包裹 + 强约束）
    4. 调用 Agnes LLM（Function Calling 强制结构化输出）
    5. 后处理：因果词替换 + 注入检测
    6. 引用一致性校验（关键事实核对）
    7. 组装最终输出（含降级模式）

降级策略（PR-补充）：
    - RAG 不可用 → 标记 rag_available=false，仍调用 LLM（prompt 中告知无参考）
    - LLM 失败 → 用规则模板生成基础解释，标记 fallback_mode=true
    - 引用全部无效 → references.valid=false，UI 不展示引用
"""

from __future__ import annotations

import logging
import time
from typing import Any

from domain.explain.config import get_config
from domain.explain.llm_client import LLMClient
from domain.explain.prompts import (
    DISCLAIMERS,
    SYSTEM_PROMPT,
    build_user_prompt,
    detect_injection,
    sanitize_causal_words,
)
from domain.explain.rag_retriever import (
    format_hits_for_prompt,
    is_available as rag_is_available,
    retrieve_for_feature,
)
from domain.explain.reference_validator import (
    mark_references_validity,
    validate_factor_references,
)
from domain.explain.shap_structurer import format_factors_for_prompt, shap_to_structured

logger = logging.getLogger(__name__)


def _build_fallback_explanation(
    stay_id: int,
    hour_index: int,
    risk_score: float,
    risk_band_label: str,
    structured: dict[str, Any],
    rag_available: bool,
    llm_error: str = "",
) -> dict[str, Any]:
    """降级输出：用规则模板生成基础解释（不调用 LLM）。"""
    factors_out: list[dict[str, Any]] = []
    md_lines: list[str] = []
    md_lines.append("## 风险评估解释（降级模式）")
    md_lines.append("")
    if not rag_available:
        md_lines.append("> ⚠️ 当前无医学参考依据，以下解释基于模型原始数值，仅供参考。")
    if llm_error:
        md_lines.append(f"> ⚠️ AI 解释服务暂不可用：{llm_error}")
    md_lines.append("")
    md_lines.append(f"**风险概述**：入ICU后 {hour_index} 小时预测未来 12 小时死亡风险为 {risk_score:.1%}（{risk_band_label}）")
    md_lines.append("")
    md_lines.append("**关键驱动因素（模型归因）**：")
    for i, sf in enumerate(structured["factors"], 1):
        value_str = f"{sf['value']} {sf['unit']}" if sf["value"] is not None else "缺失"
        direction = "升高风险" if sf["shap_direction"] == "positive" else "降低风险"
        abnormal = f"（{sf['abnormal_note']}）" if sf["is_abnormal"] else ""
        interp = f"模型统计观察：{sf['standard_name']} 实测值 {value_str}{abnormal}，SHAP 贡献 {sf['shap']}（{direction}）"
        factors_out.append({
            "feature": sf["feature"],
            "value": sf["value"],
            "unit": sf["unit"],
            "shap": sf["shap"],
            "shap_direction": sf["shap_direction"],
            "clinical_interpretation": interp,
            "reference_id": "",
            "reference_valid": False,
        })
        md_lines.append(f"{i}. {sf['standard_name']}（{value_str}）：{interp}")
    md_lines.append("")
    md_lines.append(f"**已解释特征累计SHAP贡献占比**：{structured['coverage_pct']}%")
    md_lines.append("")
    md_lines.append("**免责声明**：")
    for d in DISCLAIMERS:
        md_lines.append(f"- {d}")

    return {
        "status": "fallback",
        "explanation": "\n".join(md_lines),
        "structured": {
            "stay_id": stay_id,
            "hour_index": hour_index,
            "risk_score": risk_score,
            "risk_band": risk_band_label,
            "summary": f"入ICU后 {hour_index} 小时预测未来 12 小时死亡风险为 {risk_score:.1%}（{risk_band_label}）",
            "factor_analysis": factors_out,
            "coverage_pct": structured["coverage_pct"],
            "references": [],
            "disclaimers": DISCLAIMERS,
            "rag_available": rag_available,
            "fallback_mode": True,
        },
        "references": [],
        "disclaimers": DISCLAIMERS,
        "elapsed_ms": 0,
    }


def generate_explanation(
    stay_id: int,
    hour_index: int,
    shap_output: list[dict[str, Any]],
    risk_score: float,
    recommendation: dict[str, Any],
    features_display: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成患者风险的完整解释报告。

    Args:
        stay_id: ICU stay ID（不传入 LLM，仅用于内部标识）
        hour_index: 预测时刻
        shap_output: L3 predict_stay() 返回的 top_factors 列表
        risk_score: 风险分数
        recommendation: L3 recommend_action() 返回的字典
        features_display: 可选，完整特征值字典

    Returns:
        {
            "status": "ok" | "fallback",
            "explanation": str,        # Markdown 解释文本
            "structured": dict,        # JSON 结构化输出
            "references": list,        # 溯源引用列表
            "disclaimers": list,
            "elapsed_ms": int,
        }
    """
    cfg = get_config()
    t0 = time.time()
    risk_band_label = recommendation.get("label", "未分级")

    # 1. SHAP 结构化
    structured = shap_to_structured(
        shap_output,
        features_display=features_display,
        max_factors=cfg.output.max_factor_analysis,
    )

    # 2. RAG 检索（按特征检索）
    rag_available = rag_is_available()
    all_hits = []
    if rag_available:
        for sf in structured["factors"]:
            hits = retrieve_for_feature(sf["feature"], sf["value"], sf["standard_name"])
            all_hits.extend(hits)
        # 去重：同一 parent_id 只保留得分最高的
        seen: set[str] = set()
        deduped_hits = []
        for h in sorted(all_hits, key=lambda x: x.score, reverse=True):
            if h.parent_id in seen:
                continue
            seen.add(h.parent_id)
            deduped_hits.append(h)
        all_hits = deduped_hits[: cfg.rag.top_k * 2]  # 控制上下文长度

    rag_context, references = format_hits_for_prompt(all_hits)

    # 3. 若 RAG 不可用，直接降级
    if not rag_available:
        elapsed_ms = int((time.time() - t0) * 1000)
        result = _build_fallback_explanation(
            stay_id, hour_index, risk_score, risk_band_label, structured, rag_available=False
        )
        result["elapsed_ms"] = elapsed_ms
        logger.info("[explain] RAG 不可用，降级输出，stay_id=%s, elapsed=%dms", stay_id, elapsed_ms)
        return result

    # 4. 构造 Prompt
    shap_formatted = format_factors_for_prompt(structured)
    user_prompt = build_user_prompt(
        hour_index=hour_index,
        risk_score=f"{risk_score:.1%}",
        risk_band_label=risk_band_label,
        shap_features_formatted=shap_formatted,
        rag_context=rag_context,
    )

    # 5. 调用 LLM
    try:
        client = LLMClient()
        resp = client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            use_function_calling=True,
        )
    except Exception as e:
        logger.exception("[explain] LLM 客户端初始化失败")
        elapsed_ms = int((time.time() - t0) * 1000)
        result = _build_fallback_explanation(
            stay_id, hour_index, risk_score, risk_band_label, structured,
            rag_available=True, llm_error=f"{type(e).__name__}: {e}"
        )
        result["elapsed_ms"] = elapsed_ms
        return result

    if not resp.ok:
        elapsed_ms = int((time.time() - t0) * 1000)
        result = _build_fallback_explanation(
            stay_id, hour_index, risk_score, risk_band_label, structured,
            rag_available=True, llm_error=resp.error
        )
        result["elapsed_ms"] = elapsed_ms
        logger.warning("[explain] LLM 调用失败，降级输出，stay_id=%s, error=%s", stay_id, resp.error)
        return result

    # 6. 解析 LLM 输出
    parsed = resp.parsed
    if not parsed:
        # Function Calling 失败，尝试从 content 解析 JSON
        elapsed_ms = int((time.time() - t0) * 1000)
        result = _build_fallback_explanation(
            stay_id, hour_index, risk_score, risk_band_label, structured,
            rag_available=True, llm_error="LLM 未返回结构化输出"
        )
        result["elapsed_ms"] = elapsed_ms
        return result

    # 6b. 校验 parsed 结构：factor_analysis 应为 dict 列表
    factor_analysis_raw = parsed.get("factor_analysis", [])
    if not isinstance(factor_analysis_raw, list):
        # factor_analysis 被 LLM 当字符串返回了，尝试整体 JSON 恢复
        logger.warning(
            "[explain] factor_analysis 类型异常 %s，尝试 JSON 恢复", type(factor_analysis_raw).__name__
        )
        if isinstance(factor_analysis_raw, str) and factor_analysis_raw.strip().startswith("["):
            try:
                factor_analysis_raw = json.loads(factor_analysis_raw)
            except json.JSONDecodeError:
                pass
    if not isinstance(factor_analysis_raw, list) or not all(
        isinstance(f, dict) for f in factor_analysis_raw
    ):
        logger.warning(
            "[explain] LLM 返回的 factor_analysis 结构异常，降级输出，type=%s, sample=%s",
            type(factor_analysis_raw).__name__,
            str(factor_analysis_raw)[:80] if not isinstance(factor_analysis_raw, list) else f"len={len(factor_analysis_raw)} first_type={type(factor_analysis_raw[0]).__name__ if factor_analysis_raw else 'empty'}",
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        result = _build_fallback_explanation(
            stay_id, hour_index, risk_score, risk_band_label, structured,
            rag_available=True, llm_error="LLM 返回的 factor_analysis 结构异常"
        )
        result["elapsed_ms"] = elapsed_ms
        return result

    # 7. 后处理：因果词替换 + 注入检测
    factor_analysis = factor_analysis_raw
    for fa in factor_analysis:
        interp = fa.get("clinical_interpretation", "")
        # 注入检测
        if detect_injection(interp):
            logger.warning("[explain] 检测到注入内容，已拦截：stay_id=%s, feature=%s", stay_id, fa.get("feature"))
            fa["clinical_interpretation"] = "模型统计观察：（内容已被安全策略拦截）"
            fa["reference_id"] = ""
            continue
        # 因果词替换
        fa["clinical_interpretation"] = sanitize_causal_words(interp)

    # 8. 引用一致性校验
    factor_analysis = validate_factor_references(factor_analysis, references)
    references = mark_references_validity(references, factor_analysis)

    # 9. 补充每个 factor 的 value/unit/shap 等字段（从 structured 取）
    sf_map = {sf["feature"]: sf for sf in structured["factors"]}
    for fa in factor_analysis:
        sf = sf_map.get(fa.get("feature", ""), {})
        fa.setdefault("value", sf.get("value"))
        fa.setdefault("unit", sf.get("unit", ""))
        fa.setdefault("shap", sf.get("shap", 0))
        fa.setdefault("shap_direction", sf.get("shap_direction", "positive"))

    # 10. 组装最终输出
    summary = parsed.get("summary", "")
    coverage_note = parsed.get("coverage_note", f"已解释特征累计SHAP贡献占比：{structured['coverage_pct']}%")

    md_lines: list[str] = []
    md_lines.append("## 风险评估解释报告")
    md_lines.append("")
    md_lines.append(f"**风险概述**：{summary}")
    md_lines.append("")
    md_lines.append("**关键驱动因素分析**：")
    for i, fa in enumerate(factor_analysis, 1):
        sf = sf_map.get(fa.get("feature", ""), {})
        sn = sf.get("standard_name", fa.get("feature", ""))
        value_str = f"{fa.get('value')} {fa.get('unit', '')}".strip() if fa.get("value") is not None else "缺失"
        md_lines.append(f"{i}. **{sn}**（实测值：{value_str}，SHAP贡献：{fa.get('shap', 0)}）")
        md_lines.append(f"   {fa.get('clinical_interpretation', '')}")
        if fa.get("reference_id") and fa.get("reference_valid"):
            md_lines.append(f"   📚 引用：{fa['reference_id']}")
        elif fa.get("reference_id") and not fa.get("reference_valid"):
            md_lines.append(f"   ⚠️ 引用 {fa['reference_id']} 校验未通过")
    md_lines.append("")
    md_lines.append(f"**覆盖度**：{coverage_note}")
    md_lines.append("")
    if references:
        md_lines.append("**解释溯源**：")
        for ref in references:
            valid_mark = "✓" if ref.get("valid") else "✗"
            md_lines.append(f"- [{ref['id']}] {valid_mark} {ref['source']} - {ref['title']}（相似度 {ref.get('score', 0)}）")
        md_lines.append("")
    md_lines.append("**免责声明**：")
    for d in DISCLAIMERS:
        md_lines.append(f"- {d}")

    elapsed_ms = int((time.time() - t0) * 1000)
    final_structured = {
        "stay_id": stay_id,
        "hour_index": hour_index,
        "risk_score": risk_score,
        "risk_band": recommendation.get("band", "unknown"),
        "risk_band_label": risk_band_label,
        "summary": summary,
        "factor_analysis": factor_analysis,
        "coverage_pct": structured["coverage_pct"],
        "coverage_note": coverage_note,
        "references": references,
        "disclaimers": DISCLAIMERS,
        "rag_available": rag_available,
        "fallback_mode": False,
        "llm_available": True,
        "llm_usage": resp.usage,
    }

    logger.info(
        "[explain] 完成 stay_id=%s, factors=%d, refs=%d, elapsed=%dms, tokens=%s",
        stay_id, len(factor_analysis), len(references), elapsed_ms, resp.usage,
    )

    return {
        "status": "ok",
        "explanation": "\n".join(md_lines),
        "structured": final_structured,
        "references": references,
        "disclaimers": DISCLAIMERS,
        "elapsed_ms": elapsed_ms,
    }
