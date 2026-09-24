"""引用一致性校验：关键事实核对（PR-4.2 / PR-12）。

方法（替代 embedding 相似度，避免循环论证）：
    1. 从 LLM 解释文本中提取所有数值（含小数）
    2. 检查对应引用片段是否包含这些数值
    3. 数值匹配失败的引用标记为 reference_valid=false

D1.2 优化（2026-09-15）：
    - 容差改为**分段策略**：比值型 ±0.5 / 浓度 ±15%相对 / 百分比 ±2 / 计数 ±20%相对
    - 校验改用 **parent_content**（完整父chunk），snippet 仅用于 UI 展示
    - 无效引用增加原因分类（无匹配/截断丢失/无数值）
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


# ---------- 分段容差策略 ----------

def _smart_tolerance(expected_val: float, reference_val: float) -> float:
    """根据数值类型动态计算容差。

    策略：
        - 比值型（SOFA分值、休克指数等，通常 0-10）：绝对容差 ±0.5
        - 百分比型（SpO2 0-100，GCS 0-15）：绝对容差 ±2
        - 计数型（WBC 0-50, PLT 0-500）：相对 ±20% 或绝对 ±1（取较大者）
        - 浓度/量值型（乳酸、肌酐、BUN等，通常 >10）：相对 ±15% 或绝对 ±0.5（取较大者）

    类型判断逻辑：基于 expected_val 的值域推断，避免将 4.2 mmol/L 误判为比值。
    """
    v = expected_val if expected_val > 0 else reference_val

    # 比值型：SOFA 0-10, 休克指数 0-3（两端都小才判定为比值）
    if v <= 10 and reference_val <= 10:
        return 0.5
    # 百分比型：SpO2 0-100, GCS 0-15
    if 0 < v <= 100:
        return 2.0
    # 计数型：WBC 0-50
    if v <= 50:
        return max(1.0, expected_val * 0.20 if expected_val > 0 else 1.0)
    # 默认：浓度/量值型，相对 ±15%
    return max(0.5, expected_val * 0.15 if expected_val > 0 else 0.5)


def validate_reference(
    explanation_text: str,
    reference_content: str,
    *,
    use_smart_tolerance: bool = True,
) -> tuple[bool, str]:
    """校验解释文本与引用内容的事实一致性。

    判断逻辑：
        - 提取解释文本中的数值
        - 检查引用内容是否包含这些数值（容差内）
        - 如果解释中有数值但引用内容一个都不匹配，判定为无效引用
        - 如果解释中无数值，或引用内容中至少匹配一个数值，判定为有效

    D1.2 优化：
        - 使用分段容差（_smart_tolerance）替代固定 ±0.05
        - 支持传入完整内容（parent_content），避免 snippet 截断问题

    Args:
        explanation_text: LLM 生成的解释文本
        reference_content: 引用内容（优先使用 parent_content，回退到 snippet）
        use_smart_tolerance: 是否启用智能容差（默认 True）

    Returns:
        (bool, str): (引用是否有效, 原因说明)
    """
    if not explanation_text or not reference_content:
        return False, "empty"

    exp_nums = extract_numbers(explanation_text)
    if not exp_nums:
        # 解释中无数值，无法证伪，默认有效
        return True, "no_numbers"

    ref_nums = extract_numbers(reference_content)
    if not ref_nums:
        # 解释有数值但引用内容无数值，可能引用无关
        return False, "ref_no_numbers"

    # 检查解释中的数值是否至少有一个在引用内容中出现（容差内）
    matched_count = 0
    match_details = []
    for ev in exp_nums:
        for rv in ref_nums:
            if use_smart_tolerance:
                tol = _smart_tolerance(ev, rv)
            else:
                tol = 0.05
            if abs(ev - rv) <= tol:
                matched_count += 1
                match_details.append(f"{ev}≈{rv}(tol={tol:.2f})")
                break

    if matched_count >= 1:
        return True, f"matched({matched_count}):{'/'.join(match_details)}"
    return False, f"no_match(exp={exp_nums},ref={ref_nums})"


def validate_factor_references(
    factor_analysis: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """为每个 factor_analysis 项校验引用有效性。

    D1.2 优化：使用 parent_content 做校验（而非 snippet），避免截断导致误判。

    Args:
        factor_analysis: LLM 输出的特征分析列表
        references: 检索到的引用列表（每项应包含 parent_content 字段）

    Returns:
        更新后的 factor_analysis，每项增加 reference_valid 和 reference_reason 字段
    """
    ref_map = {r["id"]: r for r in references}
    for fa in factor_analysis:
        ref_id = fa.get("reference_id", "")
        if not ref_id or ref_id not in ref_map:
            fa["reference_valid"] = False
            fa["reference_reason"] = "no_ref_id"
            continue
        ref = ref_map[ref_id]
        interpretation = fa.get("clinical_interpretation", "")
        # 优先用 parent_content（完整内容），回退到 snippet（截断内容）
        content = ref.get("parent_content", ref.get("snippet", ""))
        valid, reason = validate_reference(interpretation, content)
        fa["reference_valid"] = valid
        fa["reference_reason"] = reason
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
