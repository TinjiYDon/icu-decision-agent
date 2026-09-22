"""Unit tests for domain.models.dca (D2 decision curve analysis)."""

from __future__ import annotations

import numpy as np
import pytest

from domain.models.dca import (
    NetBenefitPoint,
    WorkingPoint,
    dca_compare_models,
    dca_curve_with_ci,
    dca_working_point,
    net_benefit_at_threshold,
    net_benefit_curve,
)


# ---------------------------------------------------------------------------
# net_benefit_at_threshold
# ---------------------------------------------------------------------------

def test_net_benefit_at_threshold_basic():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    out = net_benefit_at_threshold(y, p, threshold=0.5)
    assert out["threshold"] == 0.5
    assert out["net_benefit_treat_none"] == 0.0
    assert out["net_benefit_model"] >= out["net_benefit_treat_all"] - 1e-9
    assert out["tp"] == 2 and out["fp"] == 0
    assert out["tn"] == 2 and out["fn"] == 0


def test_net_benefit_at_threshold_raises_on_bad_threshold():
    with pytest.raises(ValueError):
        net_benefit_at_threshold([0, 1], [0.5, 0.5], threshold=0.0)
    with pytest.raises(ValueError):
        net_benefit_at_threshold([0, 1], [0.5, 0.5], threshold=1.0)


def test_net_benefit_at_threshold_no_predictions():
    y = np.array([0, 0, 0, 0])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    out = net_benefit_at_threshold(y, p, threshold=0.5)
    assert out["net_benefit_model"] == 0.0
    assert out["tp"] == 0 and out["fp"] == 0


# ---------------------------------------------------------------------------
# net_benefit_curve
# ---------------------------------------------------------------------------

def test_net_benefit_curve_matches_pointwise():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.7, 0.4, 0.6])
    curve = net_benefit_curve(y, p, thresholds=[0.3, 0.5])
    point = net_benefit_at_threshold(y, p, threshold=0.5)
    assert abs(curve["net_benefit_model"][1] - point["net_benefit_model"]) < 1e-12
    assert len(curve["thresholds"]) == 2
    assert curve["n"] == 4


# ---------------------------------------------------------------------------
# dca_curve_with_ci
# ---------------------------------------------------------------------------

def test_dca_curve_with_ci_too_few_samples():
    y = np.array([0, 1])
    p = np.array([0.3, 0.7])
    result = dca_curve_with_ci(y, p, n_boot=10)
    assert result.status == "too_few"
    assert result.thresholds == []


def test_dca_curve_with_ci_small_sample():
    """Small sample should produce CI but with a warning."""
    np.random.seed(0)
    y = np.random.binomial(1, 0.1, size=30)
    p = np.random.beta(2, 5, size=30)
    result = dca_curve_with_ci(y, p, n_boot=50, seed=42)
    assert result.status == "ok"
    assert result.ci_lower_model is not None
    assert result.ci_upper_model is not None
    assert len(result.thresholds) > 0
    assert len(result.ci_lower_model) == len(result.thresholds)
    assert len(result.ci_upper_model) == len(result.thresholds)
    # CI band should be reasonable (lower ≤ upper)
    for lo, hi in zip(result.ci_lower_model, result.ci_upper_model):
        assert lo <= hi + 1e-9


def test_dca_curve_with_ci_larger_sample():
    """Larger sample → tighter CI (standard deviation decreases)."""
    np.random.seed(1)
    y = np.array([0] * 200 + [1] * 5)  # ~2.5% prevalence
    p = np.concatenate([
        np.random.beta(1, 10, 200),
        np.random.beta(5, 2, 5),
    ])
    result = dca_curve_with_ci(y, p, n_boot=200, thresholds=[0.05, 0.10, 0.20], seed=42)
    assert result.status == "ok"
    assert result.message == ""  # no warnings for n >= 50
    assert result.n_samples == 205
    assert result.positive_rate == 5 / 205


def test_dca_curve_with_ci_ci_band_nonnegative_width():
    np.random.seed(2)
    y = np.random.binomial(1, 0.05, size=100)
    p = np.random.beta(2, 15, size=100)
    result = dca_curve_with_ci(y, p, n_boot=100, seed=99)
    assert result.status == "ok"
    for lo, hi in zip(result.ci_lower_model, result.ci_upper_model):
        assert hi >= lo - 1e-9  # numerical tolerance


# ---------------------------------------------------------------------------
# dca_working_point
# ---------------------------------------------------------------------------

def test_dca_working_point_default_cbr():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.random.beta(1, 5, 90), np.random.beta(3, 1, 10)])
    wp = dca_working_point(y, p, cost_benefit_ratio=1.0 / 9.0)
    assert abs(wp.threshold - 0.1) < 1e-9
    assert abs(wp.cbr - 1.0 / 9.0) < 1e-9
    assert 0.0 <= wp.sensitivity <= 1.0
    assert 0.0 <= wp.specificity <= 1.0
    assert 0.0 <= wp.ppv <= 1.0
    assert 0.0 <= wp.npv <= 1.0


def test_dca_working_point_raises_on_bad_cbr():
    with pytest.raises(ValueError):
        dca_working_point([0, 1], [0.5, 0.5], cost_benefit_ratio=0.0)
    with pytest.raises(ValueError):
        dca_working_point([0, 1], [0.5, 0.5], cost_benefit_ratio=100.0)


def test_dca_working_point_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.99, 0.98])
    wp = dca_working_point(y, p, cost_benefit_ratio=1.0 / 9.0)
    assert wp.threshold == pytest.approx(0.1)
    assert wp.sensitivity == 1.0
    assert wp.specificity == 1.0
    assert wp.ppv == 1.0
    assert wp.net_benefit_model > wp.net_benefit_treat_all


# ---------------------------------------------------------------------------
# dca_compare_models
# ---------------------------------------------------------------------------

def test_dca_compare_models_two_models():
    np.random.seed(3)
    y = np.array([0] * 80 + [1] * 20)
    # Model A: decent
    p_a = np.concatenate([np.random.beta(1, 3, 80), np.random.beta(4, 2, 20)])
    # Model B: worse (near random)
    p_b = np.random.beta(2, 2, 100)
    result = dca_compare_models(y, {"model_a": p_a, "model_b": p_b}, cost_benefit_ratio=1/9, n_boot=50, seed=42)
    assert "model_a" in result["models"]
    assert "model_b" in result["models"]
    assert result["models"]["model_a"]["status"] == "ok"
    assert result["models"]["model_b"]["status"] == "ok"
    assert "ci_95" in result["models"]["model_a"]
    assert result["models"]["model_a"]["ci_95"]["lower"] <= result["models"]["model_a"]["ci_95"]["upper"]


def test_dca_compare_models_size_mismatch():
    y = np.array([0, 1, 0, 1])
    p_bad = np.array([0.5])  # wrong size
    result = dca_compare_models(y, {"bad": p_bad})
    assert result["models"]["bad"]["status"] == "size_mismatch"


def test_dca_compare_models_reference():
    np.random.seed(4)
    y = np.array([0] * 95 + [1] * 5)
    p = np.random.beta(2, 10, 100)
    result = dca_compare_models(y, {"m": p}, cost_benefit_ratio=1/9, n_boot=50)
    assert "reference_treat_all_nb" in result
    assert isinstance(result["reference_treat_all_nb"], float)
