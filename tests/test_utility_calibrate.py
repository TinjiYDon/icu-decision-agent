from __future__ import annotations

import numpy as np

from domain.models.utility_calibrate import (
    compare_f1_vs_utility,
    select_threshold_by_net_benefit,
)


def test_select_threshold_by_net_benefit_in_range():
    rng = np.random.default_rng(0)
    y = np.array([0] * 90 + [1] * 10)
    p = np.clip(y * 0.7 + rng.random(100) * 0.3, 0, 1)
    t = select_threshold_by_net_benefit(y, p, cost_ratio=4.0, max_alert_rate=0.2)
    assert 0.05 <= t <= 0.40
    assert float((p >= t).mean()) <= 0.20 + 1e-9


def test_compare_f1_vs_utility_structure():
    rng = np.random.default_rng(1)
    yv = np.array([0] * 80 + [1] * 20)
    yt = np.array([0] * 80 + [1] * 20)
    pv = np.clip(yv * 0.6 + rng.random(100) * 0.4, 0, 1)
    pt = np.clip(yt * 0.6 + rng.random(100) * 0.4, 0, 1)
    out = compare_f1_vs_utility(yv, pv, yt, pt, cost_ratio=4.0)
    assert out["status"] == "ok"
    assert "f1_policy" in out and "utility_policy" in out
    assert out["f1_policy"]["test"]["pr_auc"] == out["utility_policy"]["test"]["pr_auc"]
