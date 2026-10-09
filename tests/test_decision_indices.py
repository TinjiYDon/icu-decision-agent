from __future__ import annotations

import numpy as np

from domain.models.decision_indices import (
    alert_normalized_benefit,
    decision_efficiency_index,
    utility_efficiency_index,
)
from domain.models.sufh import build_sufh_design, recency_weighted_trajectory, train_sufh


def test_dei_positive_when_better_than_treat_all():
    y = np.array([0] * 80 + [1] * 20)
    # High precision low alert vs random
    p = np.array([0.05] * 80 + [0.9] * 20)
    dei = decision_efficiency_index(y, p, threshold=0.5, cost_ratio=1.0)
    assert dei["DEI"] > 0


def test_anb_and_uei_finite():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    anb = alert_normalized_benefit(y, p, threshold=0.5, cost_ratio=1.0)
    uei = utility_efficiency_index(y, p, threshold=0.5, cost_ratio=1.0)
    assert np.isfinite(anb["ANB"])
    assert np.isfinite(uei["UEI"])


def test_recency_weights_favor_recent():
    hour_prob = {(1, 0): 0.1, (1, 6): 0.9}
    out = recency_weighted_trajectory([1], 6, hour_prob, hours=(0, 6), beta=1.0)
    # weight at h=6 is e^0=1, at h=0 is e^{-6} tiny → near 0.9
    assert out[0] > 0.7


def test_sufh_trains():
    rng = np.random.default_rng(0)
    y = np.array([0] * 60 + [1] * 40)
    pc = np.clip(y * 0.5 + rng.random(100) * 0.4, 0, 1)
    pr = np.clip(pc + rng.normal(0, 0.05, 100), 0, 1)
    c = rng.random(100)
    X = build_sufh_design(pc, pr, c)
    assert X.shape == (100, 4)
    m = train_sufh(pc, pr, c, y)
    assert hasattr(m, "predict_proba")
