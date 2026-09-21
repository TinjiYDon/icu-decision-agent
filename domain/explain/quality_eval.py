"""LLM-as-Judge 质量评估（D1.3）。

用 Agnes LLM 作为裁判，对已生成的解释报告做以下评判：
- evidence_support_score（0-5）：解释是否有充分文献支撑
- hallucination_check（bool）：是否存在与 RAG context 明显矛盾的内容
- clinical_coherence_score（0-5）：解释整体临床逻辑是否自洽

注意：此模块为可选增强，不影响主解释链（shap_llm.py）的正常运行。
若 LLM 调用失败，返回默认值而非抛异常。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from domain.explain.llm_client import LLMClient

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """你是一位 ICU 临床专家兼医学研究方法学家。
你的任务是评估一份 ICU 死亡风险解释报告的质量。

评估维度：
1. evidence_support_score（0-5分）：
   - 5分：每个关键断言都有明确的文献或数据支撑，引用具体且相关
   - 3分：大部分断言有依据，但个别缺乏支撑
   - 1分：多处断言无依据或引用无关

2. hallucination_check（true/false）：
   - true：存在与提供的医学参考片段明显矛盾的事实性陈述
   - false：所有陈述与参考片段一致或参考片段不足时明确标注

3. clinical_coherence_score（0-5分）：
   - 5分：整体逻辑自洽，风险表述与SHAP方向一致
   - 3分：基本自洽，个别表述模糊
   - 1分：存在明显逻辑矛盾（如高风险因子说降低风险）

只输出 JSON，不要其他文字。格式：
{"evidence_support_score": N, "hallucination_check": bool, "clinical_coherence_score": N}"""


JUDGE_USER_TEMPLATE = """请评估以下解释报告：

### 患者信息
- 预测时刻：入ICU后 {hour_index} 小时
- 未来12小时死亡风险：{risk_score}
- 风险分级：{risk_band_label}

### 医学参考片段
{rag_context}

### 解释报告
{explanation_text}

### 关键因子SHAP方向
{shap_directions}

请输出 JSON 评估结果。"""


def llm_judge_quality(
    explanation_result: dict[str, Any],
    rag_context: str = "",
) -> dict[str, Any]:
    """对单个解释结果做 LLM-as-Judge 评判。

    Args:
        explanation_result: generate_explanation() 返回的完整结果 dict
        rag_context: RAG 检索到的医学参考片段文本（用于 hallucination 检测）

    Returns:
        {
            "evidence_support_score": int (0-5),
            "hallucination_check": bool,
            "clinical_coherence_score": int (0-5),
            "error": str | None,
        }
    """
    structured = explanation_result.get("structured", {})
    factors = structured.get("factor_analysis", [])
    risk_score = structured.get("risk_score", 0)
    risk_band_label = structured.get("risk_band_label", "未知")
    hour_index = structured.get("hour_index", 0)

    # 提取解释文本
    explanation_lines = []
    for i, fa in enumerate(factors, 1):
        feat = fa.get("feature", "")
        interp = fa.get("clinical_interpretation", "")
        ref_id = fa.get("reference_id", "")
        line = f"{i}. {feat}: {interp}"
        if ref_id:
            line += f" [引用:{ref_id}]"
        explanation_lines.append(line)
    explanation_text = "\n".join(explanation_lines)

    # 提取 SHAP 方向（用于一致性检查）
    shap_lines = []
    for fa in factors:
        feat = fa.get("feature", "")
        shap = fa.get("shap", 0)
        direction = "升高风险" if shap >= 0 else "降低风险"
        shap_lines.append(f"- {feat}: SHAP={shap} ({direction})")
    shap_directions = "\n".join(shap_lines)

    user_prompt = JUDGE_USER_TEMPLATE.format(
        hour_index=hour_index,
        risk_score=f"{risk_score:.1%}",
        risk_band_label=risk_band_label,
        rag_context=rag_context[:2000] if rag_context else "（无参考片段）",
        explanation_text=explanation_text,
        shap_directions=shap_directions,
    )

    try:
        client = LLMClient()
        resp = client.generate(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            use_function_calling=False,
        )
        if not resp.ok or not resp.content:
            return {
                "evidence_support_score": None,
                "hallucination_check": None,
                "clinical_coherence_score": None,
                "error": resp.error or "empty response",
            }

        content = resp.content.strip()
        # 尝试从 content 中提取 JSON
        if content.startswith("{"):
            parsed = json.loads(content)
        else:
            # 可能在代码块中
            import re
            m = re.search(r'\{[^}]+\}', content)
            if m:
                parsed = json.loads(m.group())
            else:
                raise ValueError(f"无法解析 JSON: {content[:100]}")

        return {
            "evidence_support_score": int(parsed.get("evidence_support_score", 3)),
            "hallucination_check": bool(parsed.get("hallucination_check", False)),
            "clinical_coherence_score": int(parsed.get("clinical_coherence_score", 3)),
            "error": None,
        }
    except Exception as e:
        logger.warning("[quality_eval] LLM-as-Judge 调用失败: %s", e)
        return {
            "evidence_support_score": None,
            "hallucination_check": None,
            "clinical_coherence_score": None,
            "error": str(e),
        }


def batch_judge(
    results: list[dict[str, Any]],
    rag_contexts: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """批量对多个解释结果做 LLM-as-Judge。

    Args:
        results: generate_explanation() 返回的结果列表
        rag_contexts: stay_id → rag_context 映射（可选，提高 hallucination 检测精度）

    Returns:
        每个结果的 judge 评分列表
    """
    rag_contexts = rag_contexts or {}
    judgments = []
    for ex in results:
        stay_id = ex.get("structured", {}).get("stay_id", 0)
        ctx = rag_contexts.get(stay_id, "")
        judge = llm_judge_quality(ex, rag_context=ctx)
        judge["stay_id"] = stay_id
        judgments.append(judge)
    return judgments
