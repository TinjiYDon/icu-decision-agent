"""D1 解释链安全对照 · 单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain.explain.audit import (
    MODE_FULL,
    MODE_LLM_NO_RAG,
    MODE_RULE,
    compute_overcommit_rate,
    compute_ref_validity_rate,
    compute_shap_align,
    compute_feature_coverage,
    compute_factual_grounding,
    compute_evidence_specificity,
)


MOCK_PREDICTION = {
    "stay_id": 30000001,
    "hour_index": 1,
    "status": "ok",
    "risk_score": 0.185,
    "score_kind": "probability",
    "recommend": {"band": "observe", "label": "观察（低风险）"},
    "top_factors": [
        {"feature": "lab_lactate", "value": 2.3, "shap": 0.0823},
        {"feature": "shock_index", "value": 1.1, "shap": 0.0456},
        {"feature": "vital_gcs_total", "value": 14, "shap": -0.0289},
        {"feature": "lab_ph", "value": 7.32, "shap": 0.0156},
        {"feature": "anchor_age", "value": 68, "shap": 0.0098},
    ],
    "features": {"lab_lactate": 2.3, "shock_index": 1.1, "vital_gcs_total": 14},
}


class TestComputeShapAlign:
    def test_perfect_align(self):
        """LLM 输出因子与 SHAP 完全一致 → Jaccard=1.0（k=3 取前3个）"""
        llm_factors = [
            {"feature": "lab_lactate"},
            {"feature": "shock_index"},
            {"feature": "vital_gcs_total"},
        ]
        # SHAP top3 = {lab_lactate, shock_index, vital_gcs_total}，与 LLM 完全重合
        assert compute_shap_align(MOCK_PREDICTION["top_factors"], llm_factors, k=3) == 1.0

    def test_partial_align(self):
        """3/5 因子重合 → Jaccard = 3/5 = 0.6"""
        llm_factors = [
            {"feature": "lab_lactate"},
            {"feature": "shock_index"},
            {"feature": "anchor_age"},  # 不在 SHAP top5 前3但整体在top5
        ]
        result = compute_shap_align(MOCK_PREDICTION["top_factors"], llm_factors, k=5)
        # SHAP top5 = {lab_lactate, shock_index, vital_gcs_total, lab_ph, anchor_age}
        # LLM top3 = {lab_lactate, shock_index, anchor_age}
        assert result == pytest.approx(3 / 5, abs=0.01)

    def test_no_overlap(self):
        """无共同因子 → Jaccard=0"""
        llm_factors = [
            {"feature": "random_feat_a"},
            {"feature": "random_feat_b"},
        ]
        assert compute_shap_align(MOCK_PREDICTION["top_factors"], llm_factors, k=5) == 0.0

    def test_empty(self):
        """双方都为空 → 返回 1.0"""
        assert compute_shap_align([], [], k=5) == 1.0


class TestComputeRefValidity:
    def test_all_valid(self):
        result = {"structured": {"references": [
            {"id": "REF-01", "valid": True},
            {"id": "REF-02", "valid": True},
        ]}}
        assert compute_ref_validity_rate(result) == 1.0

    def test_some_invalid(self):
        result = {"structured": {"references": [
            {"id": "REF-01", "valid": True},
            {"id": "REF-02", "valid": False},
            {"id": "REF-03", "valid": True},
        ]}}
        assert compute_ref_validity_rate(result) == pytest.approx(2 / 3, abs=0.01)

    def test_no_references(self):
        """无引用 → None"""
        result = {"structured": {"references": []}}
        assert compute_ref_validity_rate(result) is None


class TestComputeOvercommit:
    def test_no_overcommit(self):
        """无决策词 → 0.0"""
        result = {"structured": {"factor_analysis": [
            {"clinical_interpretation": "模型统计观察：BUN 升高与风险正相关"},
            {"clinical_interpretation": "模型统计观察：心率增快提示高风险"},
        ]}}
        assert compute_overcommit_rate(result) == 0.0

    def test_with_overcommit(self):
        """含决策词 → > 0；3个因子中2个触发，1个安全表述"""
        result = {"structured": {"factor_analysis": [
            {"clinical_interpretation": "模型统计观察：建议密切监测 BUN"},
            {"clinical_interpretation": "模型统计观察：应尽快处理休克指数"},
            {"clinical_interpretation": "模型统计观察：乳酸升高与风险正相关"},
        ]}}
        # 前2个触发决策词（建议、应），第3个无决策词 → 2/3
        assert compute_overcommit_rate(result) == pytest.approx(2 / 3, abs=0.01)

    def test_empty(self):
        """无因子 → None"""
        result = {"structured": {"factor_analysis": []}}
        assert compute_overcommit_rate(result) is None


# ── 新增指标测试 ────────────────────────────────────────────────────────────

class TestComputeFeatureCoverage:
    """特征覆盖密度：SHAP top-k 因子名实际出现在解释文本中的比例。"""

    def test_full_coverage(self):
        """所有因子名都出现在解释文本中 → 1.0"""
        shap_factors = [
            {"feature": "lab_lactate", "value": 2.3},
            {"feature": "shock_index", "value": 1.1},
        ]
        result = {"structured": {"factor_analysis": [
            {"clinical_interpretation": "lab_lactate 升高提示组织低灌注"},
            {"clinical_interpretation": "shock_index 反映循环状态"},
        ]}}
        assert compute_feature_coverage(shap_factors, result) == 1.0

    def test_partial_coverage(self):
        """2/3 因子出现 → 0.667"""
        shap_factors = [
            {"feature": "lab_lactate", "value": 2.3},
            {"feature": "vital_hr", "value": 110},
            {"feature": "anchor_age", "value": 68},
        ]
        result = {"structured": {"factor_analysis": [
            {"clinical_interpretation": "lab_lactate 升高与风险正相关"},
            {"clinical_interpretation": "该指标提示高风险"},  # vital_hr 未提及
            {"clinical_interpretation": "anchor_age 为高龄因素"},
        ]}}
        assert compute_feature_coverage(shap_factors, result) == pytest.approx(2 / 3, abs=0.01)

    def test_no_coverage(self):
        """因子名均未出现 → 0.0"""
        shap_factors = [{"feature": "lab_bun", "value": 45}]
        result = {"structured": {"factor_analysis": [
            {"clinical_interpretation": "某项肾功能指标异常"},  # 未提 lab_bun
        ]}}
        assert compute_feature_coverage(shap_factors, result) == 0.0

    def test_empty_shap(self):
        """无 SHAP 因子 → 1.0（无约束）"""
        assert compute_feature_coverage([], {"structured": {}}) == 1.0


class TestComputeFactualGrounding:
    """事实 grounding 率：解释中数值与真实输入值匹配的比例。"""

    def test_all_grounding(self):
        """所有数值都出现在解释文本中 → 1.0"""
        shap_factors = [
            {"feature": "lab_lactate", "value": 2.3},
            {"feature": "vital_hr", "value": 110},
        ]
        result = {"structured": {"factor_analysis": [
            {"feature": "lab_lactate", "clinical_interpretation": "lab_lactate 实测 2.3，升高"},
            {"feature": "vital_hr", "clinical_interpretation": "心率 110 次/分，心动过速"},
        ]}}
        assert compute_factual_grounding(shap_factors, result) == 1.0

    def test_partial_grounding(self):
        """2/3 数值匹配 → ~0.67"""
        shap_factors = [
            {"feature": "lab_bun", "value": 45},
            {"feature": "vital_temp", "value": 38.5},
            {"feature": "gcs_total", "value": 14},
        ]
        result = {"structured": {"factor_analysis": [
            {"feature": "lab_bun", "clinical_interpretation": "BUN 45 mmol/L，高于正常"},  # 匹配
            {"feature": "vital_temp", "clinical_interpretation": "体温升高，提示感染"},  # 未提 38.5
            {"feature": "gcs_total", "clinical_interpretation": "GCS 14 分，轻度障碍"},  # 匹配
        ]}}
        assert compute_factual_grounding(shap_factors, result) == pytest.approx(2 / 3, abs=0.01)

    def test_rounded_match(self):
        """允许一位小数近似匹配"""
        shap_factors = [{"feature": "lab_k", "value": 4.15}]
        result = {"structured": {"factor_analysis": [
            {"feature": "lab_k", "clinical_interpretation": "血钾 4.2 mmol/L"},
        ]}}
        assert compute_factual_grounding(shap_factors, result) == 1.0

    def test_empty(self):
        """无 SHAP 因子或无结果 → 1.0"""
        assert compute_factual_grounding([], {"structured": {}}) == 1.0
        assert compute_factual_grounding([{"feature": "x"}], {}) == 1.0


class TestComputeEvidenceSpecificity:
    """证据具体度：含具体证据的因子占比。"""

    def test_high_specificity(self):
        """含数值+单位+正常范围 → 1.0"""
        result = {"structured": {"factor_analysis": [
            {"clinical_interpretation": "BUN 45.2 mmol/L，正常范围 2.5-7.1，危急值 >60"},
        ]}}
        assert compute_evidence_specificity(result) == 1.0

    def test_low_specificity(self):
        """纯模糊描述 → 0.0"""
        result = {"structured": {"factor_analysis": [
            {"clinical_interpretation": "该指标升高提示风险增加"},
            {"clinical_interpretation": "模型统计观察：某参数异常"},
        ]}}
        assert compute_evidence_specificity(result) == 0.0

    def test_mixed(self):
        """3个因子中2个有具体证据 → 0.67"""
        result = {"structured": {"factor_analysis": [
            {"clinical_interpretation": "乳酸 2.3 mmol/L，高于正常范围 0.5-1.7"},  # 有：数值+单位+正常范围
            {"clinical_interpretation": "心率 110 次/分，提示交感兴奋"},           # 有：数值+单位
            {"clinical_interpretation": "某指标异常，模型观察到风险上升"},          # 无
        ]}}
        assert compute_evidence_specificity(result) == pytest.approx(2 / 3, abs=0.01)

    def test_empty(self):
        """无因子 → 1.0"""
        assert compute_evidence_specificity({"structured": {"factor_analysis": []}}) == 1.0
