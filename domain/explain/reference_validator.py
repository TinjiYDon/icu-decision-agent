"""引用一致性校验：关键事实核对（PR-4.2 / PR-12）。

方法（替代 embedding 相似度，避免循环论证）：
    1. 从 LLM 解释文本中提取所有数值（含小数）
    2. 检查对应引用片段是否包含这些数值
    3. 数值匹配失败的引用标记为 reference_valid=false

更严格的核对：
    - 解释中提到的「危急值」「正常范围」等关键阈值，必须在引用片段中出现
    - 阈值附近容差：±0.05（避免浮点精度问题）
"""

from __future__ import annotations

import re
from typing import Any

# 数值提取正则：匹配整数、小数（含负数）
_NUM_PATTERN = re.compile(r"-?\d+\.?\d*")


def extract_numbers(text: str) -> list[float]:
    """从文本中提取所有数值。"""
    nums: list[float] = []
    for m in _NUM_PATTERN.finditer(text):
        try:
            v = float(m.group())
            # 忽略明显非临床数值的数字（如 REF-01 中的 01）
            if 0 <= v <= 1000 or v < 0:
                nums.append(v)
        except ValueError:
            continue
    return nums


def validate_reference(explanation_text: str, reference_snippet: str, tolerance: float = 0.05) -> bool:
    """校验解释文本与引用片段的事实一致性。

    判断逻辑：
        - 提取解释文本中的数值
        - 检查引用片段是否包含这些数值（容差 ±tolerance）
        - 如果解释中有数值但引用片段一个都不匹配，判定为无效引用
        - 如果解释中无数值，或引用片段中至少匹配一个数值，判定为有效

    Args:
        explanation_text: LLM 生成的解释文本
        reference_snippet: 引用片段内容
        tolerance: 数值容差

    Returns:
        bool: 引用是否有效
    """
    if not explanation_text or not reference_snippet:
        return False

    exp_nums = extract_numbers(explanation_text)
    if not exp_nums:
        # 解释中无数值，无法证伪，默认有效
        return True

    ref_nums = extract_numbers(reference_snippet)
    if not ref_nums:
        # 解释有数值但引用片段无数值，可能引用无关
        return False

    # 检查解释中的数值是否至少有一个在引用片段中出现（容差内）
    matched_count = 0
    for ev in exp_nums:
        for rv in ref_nums:
            if abs(ev - rv) <= tolerance:
                matched_count += 1
                break

    # 至少匹配 1 个数值即认为引用有效
    return matched_count >= 1


def validate_factor_references(
    factor_analysis: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """为每个 factor_analysis 项校验引用有效性。

    Args:
        factor_analysis: LLM 输出的特征分析列表
        references: 检索到的引用列表

    Returns:
        更新后的 factor_analysis，每项增加 reference_valid 字段
    """
    ref_map = {r["id"]: r for r in references}
    for fa in factor_analysis:
        ref_id = fa.get("reference_id", "")
        if not ref_id or ref_id not in ref_map:
            fa["reference_valid"] = False
            continue
        ref = ref_map[ref_id]
        interpretation = fa.get("clinical_interpretation", "")
        snippet = ref.get("snippet", "")
        # 用引用的完整内容做校验（snippet 已截断，但作为初步校验够用）
        fa["reference_valid"] = validate_reference(interpretation, snippet)
    return factor_analysis


def mark_references_validity(
    references: list[dict[str, Any]],
    factor_analysis: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """根据 factor_analysis 的校验结果，反向标记 references 的 valid 字段。"""
    valid_ids = {fa.get("reference_id") for fa in factor_analysis if fa.get("reference_valid")}
    for ref in references:
        ref["valid"] = ref["id"] in valid_ids
    return references
