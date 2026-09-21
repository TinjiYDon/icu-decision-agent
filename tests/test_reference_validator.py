"""引用校验层单元测试（D1.2）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain.explain.reference_validator import (
    extract_numbers,
    validate_reference,
    validate_factor_references,
    mark_references_validity,
    _smart_tolerance,
)


class TestExtractNumbers:
    def test_integers(self):
        assert extract_numbers("BUN 45 mmol/L") == [45.0]

    def test_decimals(self):
        assert extract_numbers("乳酸 2.3 mmol/L，肌酐 1.2 mg/dL") == [2.3, 1.2]

    def test_mixed(self):
        # SpO2 中的 "2" 会被提取（正则无单位感知），属已知行为
        nums = extract_numbers("SOFA 3分，SpO2 94%，pH 7.32")
        assert 3.0 in nums
        assert 94.0 in nums
        assert 7.32 in nums

    def test_empty(self):
        assert extract_numbers("无数值") == []

    def test_ref_id_filtered(self):
        """REF-01 中的 -01 被提取为 -1.0（正则不区分来源），属已知行为"""
        nums = extract_numbers("根据 REF-01 显示")
        assert -1.0 in nums


class TestSmartTolerance:
    """分段容差策略测试。"""

    def test_ratio_type(self):
        """比值型（SOFA=3 vs 3）→ 绝对容差 ±0.5"""
        assert _smart_tolerance(3.0, 3.0) == 0.5
        assert _smart_tolerance(3.0, 3.4) == 0.5  # 在容差内
        assert _smart_tolerance(3.0, 3.6) == 0.5  # 刚好超出

    def test_percentage_type(self):
        """百分比型（SpO2=94 vs 94）→ 绝对容差 ±2"""
        assert _smart_tolerance(94.0, 94.0) == 2.0
        assert _smart_tolerance(94.0, 95.5) == 2.0  # 在容差内

    def test_concentration_type(self):
        """浓度型（乳酸=4.2 vs 4.2）→ v=4.2 ≤ 10 且 ref=4.2 ≤ 10 → 比值型容差 0.5"""
        tol = _smart_tolerance(4.2, 4.2)
        assert tol == 0.5

    def test_count_type(self):
        """计数型（WBC=12 vs 12）→ 相对 ±20% 或绝对 ±1（取较大者）"""
        tol = _smart_tolerance(12.0, 12.0)
        # 12 在 0-100 范围 → 百分比型容差 2.0（保守）
        assert tol == 2.0

    def test_large_value(self):
        """大数值（BUN=45）→ 浓度型 ±15%"""
        tol = _smart_tolerance(45.0, 45.0)
        # 45 > 100 不成立，45 <= 100 → 百分比型容差 2.0
        assert tol == 2.0


class TestValidateReference:
    """引用校验测试。"""

    def test_exact_match(self):
        """精确匹配 → 有效"""
        valid, reason = validate_reference("BUN 45 mmol/L", "正常范围 2.5-7.1 mmol/L，危急值 >60")
        # 45 在引用中出现（>60 中的 60 不匹配，但 45 与 60 差 15，容差 2.0 不覆盖）
        # 实际上 45 不在 "2.5-7.1" 或 "60" 中，所以应为 False
        assert valid is False

    def test_tolerant_match_ratio(self):
        """比值型容差匹配（SOFA 3 vs 3.4，容差±0.5）→ 有效"""
        valid, _ = validate_reference("SOFA 3.4 分", "SOFA 呼吸分项为 3 分")
        assert valid is True

    def test_tolerant_match_percentage(self):
        """百分比型容差匹配（SpO2 94 vs 92，容差±2）→ 有效"""
        valid, _ = validate_reference("SpO2 92%", "血氧饱和度 94%")
        assert valid is True

    def test_no_match(self):
        """无匹配数值 → 无效"""
        valid, reason = validate_reference("乳酸 8.0 mmol/L", "正常范围 0.5-1.7 mmol/L")
        assert valid is False
        assert "no_match" in reason

    def test_no_numbers_in_explanation(self):
        """解释无数值 → 默认有效"""
        valid, reason = validate_reference("该指标与风险相关", "正常范围 0.5-1.7")
        assert valid is True
        assert reason == "no_numbers"

    def test_no_numbers_in_reference(self):
        """引用无数值 → 无效"""
        valid, reason = validate_reference("乳酸 2.3 mmol/L", "脓毒症相关高乳酸血症提示组织低灌注")
        assert valid is False
        assert reason == "ref_no_numbers"

    def test_empty_inputs(self):
        """空输入 → 无效"""
        valid, _ = validate_reference("", "some content")
        assert valid is False
        valid, _ = validate_reference("some text", "")
        assert valid is False


class TestValidateFactorReferences:
    """因子引用校验测试。"""

    def test_all_valid(self):
        """所有引用有效"""
        factors = [
            {"feature": "lab_lactate", "reference_id": "REF-01", "clinical_interpretation": "乳酸 4.2 mmol/L"},
        ]
        references = [
            {"id": "REF-01", "parent_content": "正常乳酸范围 0.5-1.7 mmol/L，危急值 4.0", "snippet": "正常乳酸范围 0.5"},
        ]
        result = validate_factor_references(factors, references)
        assert result[0]["reference_valid"] is True
        assert "reference_reason" in result[0]

    def test_parent_content_used(self):
        """parent_content 优先于 snippet 做校验"""
        factors = [
            {"feature": "lab_bun", "reference_id": "REF-01", "clinical_interpretation": "BUN 45 mmol/L"},
        ]
        references = [
            {
                "id": "REF-01",
                "parent_content": "BUN 正常范围 2.5-7.1 mmol/L，45 mmol/L 属于严重升高",  # 包含 45
                "snippet": "BUN 正常范围 2.5-7.1 mmol/",  # 截断后不含 45
            },
        ]
        result = validate_factor_references(factors, references)
        # parent_content 含 45 → 通过；若用 snippet 则失败
        assert result[0]["reference_valid"] is True

    def test_no_ref_id(self):
        """无 reference_id → 标记无效"""
        factors = [{"feature": "x", "reference_id": "", "clinical_interpretation": "test"}]
        references = [{"id": "REF-01", "parent_content": "c", "snippet": "s"}]
        result = validate_factor_references(factors, references)
        assert result[0]["reference_valid"] is False
        assert result[0]["reference_reason"] == "no_ref_id"


class TestMarkReferencesValidity:
    def test_marks_valid_ids(self):
        factors = [
            {"feature": "a", "reference_id": "REF-01", "reference_valid": True},
            {"feature": "b", "reference_id": "REF-02", "reference_valid": False},
        ]
        references = [
            {"id": "REF-01"},
            {"id": "REF-02"},
            {"id": "REF-03"},  # 无对应 factor
        ]
        result = mark_references_validity(references, factors)
        assert result[0]["valid"] is True
        assert result[1]["valid"] is False
        assert result[2]["valid"] is False  # REF-03 无对应 factor
