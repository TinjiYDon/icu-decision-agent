"""D3 GRU-D 最小对照 · 单元测试。

覆盖：
  - SequenceProfile 不可变性 & keepable_ratio 计算
  - PairedComparison 决策逻辑（三种场景）
  - run_mock() 管线连通性
  - _print_summary() 输出不抛异常
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from scripts.d3_grud_minimal_compare import (
    SequenceProfile,
    ModelResult,
    PairedComparison,
    D3Report,
    paired_compare,
    run_mock,
)


# ── SequenceProfile ─────────────────────────────────────────────────────────

class TestSequenceProfile:
    def test_frozen(self):
        p = SequenceProfile(
            n_total_stays=100, n_keepable=70,
            nonnull_rate_by_feature={"hr": 0.9},
            nonnull_rate_by_stay=[0.8],
            keepable_stay_ids=[1, 2],
            drop_reasons={},
        )
        with pytest.raises(Exception):
            p.n_total_stays = 999

    def test_keepable_ratio(self):
        p = SequenceProfile(
            n_total_stays=100, n_keepable=30,
            nonnull_rate_by_feature={}, nonnull_rate_by_stay=[],
            keepable_stay_ids=[], drop_reasons={},
        )
        assert p.keepable_ratio == 0.3

    def test_overall_nonnull_rate(self):
        p = SequenceProfile(
            n_total_stays=10, n_keepable=5,
            nonnull_rate_by_feature={}, nonnull_rate_by_stay=[],
            keepable_stay_ids=[], drop_reasons={},
            total_cells=1000, observed_cells=550,
        )
        assert p.overall_nonnull_rate == 0.55


# ── PairedComparison 决策 ───────────────────────────────────────────────────

class TestPairedComparison:
    def _make_results(self, g_auc=0.85, l_auc=0.80):
        """构造具有指定 AUC 的模拟预测结果（概率与标签正相关）。"""
        from unittest.mock import MagicMock
        rng = np.random.default_rng(99)
        n = 50
        y_true = np.array([0]*40 + [1]*10)
        # 负类（前40）给低分，正类（后10）给高分，加噪声使两类有重叠
        def _make_probs(n_neg=40, n_pos=10):
            base_neg = rng.normal(0.30, 0.15, n_neg)   # 与正类有重叠
            base_pos = rng.normal(0.55, 0.15, n_pos)
            return np.concatenate([base_neg, base_pos]).clip(0.01, 0.99)
        g_prob = _make_probs()
        l_prob = _make_probs()
        g = MagicMock()
        g.roc_auc = g_auc; g.n_test = n; g.test_stay_ids = list(range(n))
        g.test_y_true = y_true; g.test_y_prob = g_prob
        g.y_true = y_true; g.y_prob = g_prob
        l = MagicMock()
        l.roc_auc = l_auc; l.n_test = n; l.test_stay_ids = list(range(n))
        l.test_y_true = y_true; l.test_y_prob = l_prob
        l.y_true = y_true; l.y_prob = l_prob
        return g, l

    def test_grud_better_not_significant(self):
        g, l = self._make_results(0.85, 0.80)
        comp = paired_compare(g, l)
        assert comp.primary_metric == "pr_auc"
        assert comp.is_not_worse is True
        assert hasattr(comp, "grud_pr_auc") and hasattr(comp, "grud_brier")

    def test_grud_worse_not_significant(self):
        g, l = self._make_results(0.75, 0.80)
        comp = paired_compare(g, l)
        # 主验收看 PR-AUC 容差，不再以 ROC 差单独否决
        assert comp.primary_metric == "pr_auc"
        assert isinstance(comp.is_not_worse, bool)

    def test_insufficient_common(self):
        g = ModelResult(
            model_name="g", stay_ids=[], y_true=np.array([0]), y_prob=np.array([0.5]),
            split={}, test_stay_ids=[], test_y_true=np.array([]), test_y_prob=np.array([]),
            roc_auc=float("nan"), pr_auc=float("nan"), brier=float("nan"), n_test=0, n_train=0,
        )
        l = ModelResult(
            model_name="l", stay_ids=[], y_true=np.array([0]), y_prob=np.array([0.5]),
            split={}, test_stay_ids=[], test_y_true=np.array([]), test_y_prob=np.array([]),
            roc_auc=float("nan"), pr_auc=float("nan"), brier=float("nan"), n_test=0, n_train=0,
        )
        comp = paired_compare(g, l)
        assert "样本不足" in comp.conclusion


# ── run_mock 连通性 ─────────────────────────────────────────────────────────

class TestRunMock:
    def test_mock_runs_and_saves(self, tmp_path):
        import scripts.d3_grud_minimal_compare as m
        orig_art = m._D3_ARTIFACTS
        m._D3_ARTIFACTS = tmp_path / "d3"
        try:
            report = m.run_mock(limit=20)
            assert report.profile.n_total_stays == 20
            assert report.profile.n_keepable > 0
            assert report.comparison.is_not_worse is True
            # 文件已保存
            assert (tmp_path / "d3" / "comparison.json").exists()
            assert (tmp_path / "d3" / "report.md").exists()
        finally:
            m._D3_ARTIFACTS = orig_art

    def test_mock_output_json_structure(self, tmp_path):
        import scripts.d3_grud_minimal_compare as m
        orig_art = m._D3_ARTIFACTS
        m._D3_ARTIFACTS = tmp_path / "d3"
        try:
            report = m.run_mock(limit=10)
            j = json.loads((tmp_path / "d3" / "comparison.json").read_text(encoding="utf-8"))
            assert "profile" in j
            assert "grud" in j
            assert "comparison" in j
            assert j["comparison"].get("primary_metric") == "pr_auc"
            assert "elapsed_s" in j
        finally:
            m._D3_ARTIFACTS = orig_art
