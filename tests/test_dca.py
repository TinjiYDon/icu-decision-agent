"""Unit tests for DCA module (domain.models.dca)."""

from __future__ import annotations

import numpy as np
import pytest

from domain.models.dca import (
    PointResult,
    WorkingPoint,
    compute_net_benefit,
    compute_net_benefit_curve,
    find_working_point,
)


# ── PointResult 基础测试 ─────────────────────────────────────────────────────

def test_compute_net_benefit_basic():
    """基础测试：完美预测在 t=0.5 应获得最大净受益。"""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    result = compute_net_benefit(y, p, threshold=0.5, cost_ratio=1.0)
    assert isinstance(result, PointResult)
    assert result.threshold == 0.5
    assert result.cost_ratio == 1.0
    assert result.net_benefit_treat_none == 0.0
    # 完美预测下，nb_model >= nb_treat_all
    assert result.net_benefit_model >= result.net_benefit_treat_all - 1e-9


def test_compute_net_benefit_cost_ratio():
    """代价比影响净受益：cost_ratio 越大，模型净受益越小（更保守）。"""
    y = np.array([0] * 90 + [1] * 10)
    p = np.array([0.05] * 90 + [0.9] * 10)
    r1 = compute_net_benefit(y, p, threshold=0.5, cost_ratio=1.0)
    r4 = compute_net_benefit(y, p, threshold=0.5, cost_ratio=4.0)
    # cost_ratio 越大，nb_model 越小（FP 惩罚更重）
    assert r4.net_benefit_model <= r1.net_benefit_model + 1e-9


def test_compute_net_benefit_invalid_threshold():
    """阈值超出范围应报错。"""
    y = np.array([0, 1])
    p = np.array([0.3, 0.7])
    with pytest.raises(ValueError):
        compute_net_benefit(y, p, threshold=1.5)
    with pytest.raises(ValueError):
        compute_net_benefit(y, p, threshold=-0.1)


def test_compute_net_benefit_tp_fp_fn_tn():
    """验证混淆矩阵计数正确。"""
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.3, 0.7, 0.9])
    # threshold=0.5 → pred=[0, 0, 1, 0, 1, 1]
    result = compute_net_benefit(y, p, threshold=0.5)
    assert result.tn == 2   # y=0, pred=0: idx 0,1
    assert result.fp == 1   # y=0, pred=1: idx 2
    assert result.fn == 1   # y=1, pred=0: idx 3
    assert result.tp == 2   # y=1, pred=1: idx 4,5


# ── CurveResult 基础测试 ─────────────────────────────────────────────────────

def test_compute_net_benefit_curve_shape():
    """曲线输出维度正确。"""
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.1).astype(int)
    p = rng.random(200)
    curve = compute_net_benefit_curve(y, p, thresholds=[0.1, 0.3, 0.5], n_bootstrap=0)
    assert curve.status == "ok"
    assert len(curve.thresholds) == 3
    assert len(curve.nb_model) == 3
    assert len(curve.nb_treat_all) == 3
    assert len(curve.nb_treat_none) == 3
    assert curve.n == 200
    assert curve.nb_model_ci_lower is None   # n_bootstrap=0 不应有 CI
    assert curve.nb_model_ci_upper is None


def test_compute_net_benefit_curve_ci_with_bootstrap():
    """Bootstrap CI 生成非空区间。"""
    rng = np.random.default_rng(42)
    y = (rng.random(300) < 0.15).astype(int)
    # 让模型有一定区分度
    p = y * 0.7 + (1 - y) * 0.2 + rng.normal(0, 0.1, 300)
    p = np.clip(p, 0.01, 0.99)
    curve = compute_net_benefit_curve(y, p, n_bootstrap=100, seed=42)
    assert curve.status == "ok"
    assert curve.nb_model_ci_lower is not None
    assert curve.nb_model_ci_upper is not None
    assert len(curve.nb_model_ci_lower) == len(curve.thresholds)
    # CI 下限 <= 均值 <= CI 上限（统计上几乎必然成立）
    for lo, mid, hi in zip(curve.nb_model_ci_lower, curve.nb_model, curve.nb_model_ci_upper):
        assert lo <= mid <= hi


def test_compute_net_benefit_curve_prevalence():
    """阳性率计算正确。"""
    y = np.array([0] * 90 + [1] * 10)
    p = np.random.rand(100)
    curve = compute_net_benefit_curve(y, p, n_bootstrap=0)
    assert abs(curve.prevalence - 0.1) < 1e-6


# ── WorkingPoint 测试 ────────────────────────────────────────────────────────

def test_find_working_point_f1():
    """F1 工作点应返回合理的阈值和指标。"""
    y = np.array([0] * 80 + [1] * 20)
    p = np.sort(np.random.RandomState(0).rand(100))
    wp = find_working_point(y, p, method="f1")
    assert isinstance(wp, WorkingPoint)
    assert 0 < wp.threshold < 1
    assert 0 <= wp.precision <= 1
    assert 0 <= wp.recall <= 1
    assert 0 <= wp.f1 <= 1
    assert wp.tp + wp.fp + wp.fn + wp.tn == len(y)


def test_find_working_point_clinical_cost():
    """临床成本工作点应返回合理的阈值。"""
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.random.uniform(0.05, 0.3, 90), np.random.uniform(0.4, 0.95, 10)])
    wp = find_working_point(y, p, method="clinical_cost", cost_ratio=4.0)
    assert isinstance(wp, WorkingPoint)
    assert wp.cost_ratio == 4.0
    assert 0 < wp.threshold < 1


def test_working_point_deterministic():
    """相同输入应产生相同工作点。"""
    y = np.array([0] * 50 + [1] * 50)
    p = np.sort(np.random.RandomState(0).rand(100))
    wp1 = find_working_point(y, p, method="f1")
    wp2 = find_working_point(y, p, method="f1")
    assert wp1.threshold == wp2.threshold


# ── 边界条件测试 ─────────────────────────────────────────────────────────────

def test_empty_input_raises():
    """空输入应报错。"""
    y = np.array([], dtype=int)
    p = np.array([], dtype=float)
    with pytest.raises(ZeroDivisionError):
        compute_net_benefit(y, p, threshold=0.5)


def test_all_same_label():
    """全同标签（阴性）+ 随机预测 → fp>0 属正常行为。"""
    rng = np.random.default_rng(7)
    y = np.zeros(10, dtype=int)
    p = rng.random(10)
    result = compute_net_benefit(y, p, threshold=0.5)
    assert result.tp == 0
    assert result.fn == 0
    assert result.tn + result.fp == 10  # 总数对
    assert result.net_benefit_treat_none == 0.0


def test_curve_consistency_with_pointwise():
    """曲线的单点值应与 pointwise 计算一致。"""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.3, 0.7, 0.8])
    curve = compute_net_benefit_curve(y, p, thresholds=[0.5], n_bootstrap=0)
    point = compute_net_benefit(y, p, threshold=0.5)
    assert abs(curve.nb_model[0] - point.net_benefit_model) < 1e-12
