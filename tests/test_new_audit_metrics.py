"""D1.3 新增指标单元测试：risk_consistency / clinical_term_density / readability_score。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain.explain.audit import (
    compute_risk_consistency,
    compute_clinical_term_density,
    compute_readability_score,
)


# ── risk_consistency ─────────────────────────────────────────────────────────

class TestComputeRiskConsistency:
    def test_all_consistent_positive(self):
        """SHAP 为正 → 解释中无「降低风险」表述 → 一致"""
        shap = [{"feature": "lab_lactate", "shap": 0.08}]
        result = {"structured": {"factor_analysis": [
            {"feature": "lab_lactate", "clinical_interpretation": "模型统计观察：乳酸升高与风险正相关，SHAP贡献 0.08（升高风险）"},
        ]}}
        assert compute_risk_consistency(shap, result) == 1.0

    def test_all_consistent_negative(self):
        """SHAP 为负 → 解释中无「升高风险」表述 → 一致"""
        shap = [{"feature": "vital_gcs_total", "shap": -0.03}]
        result = {"structured": {"factor_analysis": [
            {"feature": "vital_gcs_total", "clinical_interpretation": "模型统计观察：GCS 14分，SHAP贡献 -0.03（降低风险）"},
        ]}}
        assert compute_risk_consistency(shap, result) == 1.0

    def test_mixed_consistency(self):
        """3个因子中2个一致，1个矛盾 → 2/3"""
        shap = [
            {"feature": "a", "shap": 0.1},   # 正
            {"feature": "b", "shap": -0.05},  # 负
            {"feature": "c", "shap": 0.02},  # 正
        ]
        result = {"structured": {"factor_analysis": [
            {"feature": "a", "clinical_interpretation": "乳酸升高，升高风险"},  # 一致
            {"feature": "b", "clinical_interpretation": "GCS高，降低风险"},    # 一致（负SHAP说降低）
            {"feature": "c", "clinical_interpretation": "年龄因素，升高风险"},  # 一致
        ]}}
        assert compute_risk_consistency(shap, result) == 1.0

    def test_contradiction_positive_factor(self):
        """正SHAP因子说「降低风险」→ 不一致"""
        shap = [{"feature": "lab_lactate", "shap": 0.08}]
        result = {"structured": {"factor_analysis": [
            {"feature": "lab_lactate", "clinical_interpretation": "模型统计观察：该指标升高反而降低风险"},
        ]}}
        assert compute_risk_consistency(shap, result) == 0.0

    def test_contradiction_negative_factor(self):
        """负SHAP因子说「升高风险」→ 不一致"""
        shap = [{"feature": "vital_gcs_total", "shap": -0.03}]
        result = {"structured": {"factor_analysis": [
            {"feature": "vital_gcs_total", "clinical_interpretation": "模型统计观察：GCS高反而与升高风险相关"},
        ]}}
        assert compute_risk_consistency(shap, result) == 0.0

    def test_empty(self):
        assert compute_risk_consistency([], {"structured": {}}) == 1.0
        assert compute_risk_consistency([{"feature": "x", "shap": 0.1}], {}) == 1.0


# ── clinical_term_density ────────────────────────────────────────────────────

class TestComputeClinicalTermDensity:
    def test_with_standard_name(self):
        """解释中包含 feature_meta.yaml 的 standard_name → 计数"""
        result = {"structured": {"factor_analysis": [
            {"feature": "lab_lactate", "clinical_interpretation": "模型统计观察：Lactate（乳酸）实测值 2.3"},
        ]}}
        # 应包含 standard_name "Lactate（乳酸）" 或特征名 "lab_lactate"
        score = compute_clinical_term_density(result)
        assert score >= 0.0

    def test_with_feature_name(self):
        """解释中包含特征名 → 计数"""
        result = {"structured": {"factor_analysis": [
            {"feature": "lab_lactate", "clinical_interpretation": "模型统计观察：lab_lactate 实测值 2.3"},
        ]}}
        assert compute_clinical_term_density(result) == 1.0

    def test_no_match(self):
        """解释中既无标准名也无特征名 → 0.0"""
        result = {"structured": {"factor_analysis": [
            {"feature": "lab_lactate", "clinical_interpretation": "某项指标异常"},
        ]}}
        assert compute_clinical_term_density(result) == 0.0

    def test_empty(self):
        assert compute_clinical_term_density({"structured": {"factor_analysis": []}}) == 1.0


# ── readability_score ────────────────────────────────────────────────────────

class TestComputeReadabilityScore:
    def test_ideal_length(self):
        """长度在 [100, 300] → 1.0"""
        long_interp = "模型统计观察：" + "X" * 120  # 总长 ~134，在区间内
        result = {"structured": {"factor_analysis": [
            {"clinical_interpretation": long_interp},
        ]}}
        assert compute_readability_score(result) == 1.0

    def test_too_short(self):
        """长度 < 80 → 线性衰减"""
        short_interp = "异常"  # 2字符
        result = {"structured": {"factor_analysis": [
            {"clinical_interpretation": short_interp},
        ]}}
        score = compute_readability_score(result)
        assert 0.0 < score < 1.0

    def test_too_long(self):
        """长度 > 300 → 线性衰减"""
        long_interp = "模型统计观察：" + "X" * 350  # 总长 ~364
        result = {"structured": {"factor_analysis": [
            {"clinical_interpretation": long_interp},
        ]}}
        score = compute_readability_score(result)
        assert 0.0 < score < 1.0

    def test_mixed_lengths(self):
        """3个因子：1个理想+1个太短+1个太长 → 平均"""
        result = {"structured": {"factor_analysis": [
            {"clinical_interpretation": "模型统计观察：" + "A" * 100},  # 理想
            {"clinical_interpretation": "短"},                          # 太短
            {"clinical_interpretation": "模型统计观察：" + "B" * 300},   # 太长
        ]}}
        score = compute_readability_score(result)
        assert 0.0 < score < 1.0

    def test_empty(self):
        assert compute_readability_score({"structured": {"factor_analysis": []}}) == 1.0
