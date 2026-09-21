"""Unit tests for decision-curve net benefit (H1 working point)."""

import numpy as np

from domain.models.evaluation import binary_metrics, net_benefit_at_threshold, net_benefit_curve


def test_net_benefit_at_threshold_bounds():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    out = net_benefit_at_threshold(y, p, threshold=0.5)
    assert out["threshold"] == 0.5
    assert out["net_benefit_treat_none"] == 0.0
    assert out["net_benefit_model"] >= out["net_benefit_treat_all"] - 1e-9


def test_binary_metrics_includes_net_benefit():
    y = np.array([0, 1, 0, 1, 0, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.4])
    m = binary_metrics(y, p, threshold=0.5)
    assert "net_benefit_model" in m
    assert m["net_benefit_model"] is not None


def test_net_benefit_curve_matches_pointwise():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.7, 0.4, 0.6])
    curve = net_benefit_curve(y, p, thresholds=[0.3, 0.5])
    point = net_benefit_at_threshold(y, p, threshold=0.5)
    assert abs(curve["net_benefit_model"][1] - point["net_benefit_model"]) < 1e-12
