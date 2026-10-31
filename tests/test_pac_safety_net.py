"""H9-I4 PAC 双阈值安全网单元测试：纯合成、不读数据库/网络。"""

import numpy as np
import pytest

from domain.models.pac_safety_net import (
    DoubleThresholdResult,
    alert_mask,
    missed_mass,
    release_mask,
    review_mask,
    select_double_threshold,
)


def _synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.4, 6.0, size=n)
    y = (rng.random(n) < 0.15 * p + 0.02).astype(float)
    return p, y


def test_masks_are_disjoint_and_cover_all():
    p = np.array([0.0, 0.09, 0.1, 0.29, 0.3, 0.31, 0.7, 1.0])
    rel = release_mask(p, 0.1)
    rev = review_mask(p, 0.1, 0.3)
    alert = alert_mask(p, 0.3)
    assert not np.any(rel & rev)
    assert not np.any(rel & alert)
    assert not np.any(rev & alert)
    assert np.all(rel | rev | alert)
    assert rel.tolist() == [True, True, False, False, False, False, False, False]
    assert rev.tolist() == [False, False, True, True, False, False, False, False]
    assert alert.tolist() == [False, False, False, False, True, True, True, True]


def test_all_negative_labels_are_feasible_and_release_most():
    rng = np.random.default_rng(7)
    p = rng.random(5000)
    y = np.zeros_like(p)
    res = select_double_threshold(p, y, alpha=0.2, delta=0.1)
    assert isinstance(res, DoubleThresholdResult)
    assert res.feasible
    assert res.n_release / res.n_cal > 0.8
    assert res.n_release + res.n_review + res.n_alert == res.n_cal
    assert res.empirical_risk == 0.0


def test_alpha_zero_raises_and_tiny_alpha_is_infeasible():
    p, y = _synthetic(n=500, seed=1)
    # alpha<=0 非法（见 docs/PAC_SAFETY_NET.md §4 约定）
    with pytest.raises(ValueError):
        select_double_threshold(p, y, alpha=0.0)
    # alpha 极小（正数）时没有候选能通过 Hoeffding 上界 -> 不可行回落
    res = select_double_threshold(p, y, alpha=1e-12, delta=0.1)
    assert not res.feasible
    assert (res.tau_low, res.tau_high) == (0.0, 1.0)
    assert res.risk_upper_bound == 1.0
    assert res.n_release == 0
    assert res.n_release + res.n_review + res.n_alert == res.n_cal


def test_risk_upper_bound_dominates_empirical_risk():
    p, y = _synthetic(n=2000, seed=2)
    res = select_double_threshold(p, y, alpha=0.5, delta=0.2)
    assert res.feasible
    assert res.risk_upper_bound >= res.empirical_risk
    assert res.risk_upper_bound <= 0.5 + 1e-12


def test_invalid_inputs_raise():
    p = np.linspace(0.0, 1.0, 100)
    y = np.zeros(100)
    with pytest.raises(ValueError):
        select_double_threshold(p, y[:-1])  # 长度不一致
    with pytest.raises(ValueError):
        select_double_threshold(p, y, alpha=-0.1)  # alpha <= 0
    with pytest.raises(ValueError):
        select_double_threshold(p, y, delta=0.0)  # delta <= 0
    with pytest.raises(ValueError):
        select_double_threshold(p, y, n_low=1)  # n_low < 2
    with pytest.raises(ValueError):
        select_double_threshold(p, y, n_high=1)  # n_high < 2
    bad_y = np.where(np.arange(100) % 2 == 0, 0.0, 2.0)
    with pytest.raises(ValueError):
        select_double_threshold(p, bad_y)  # y 非 0/1
    with pytest.raises(ValueError):
        missed_mass(p, bad_y, 0.3)
    with pytest.raises(ValueError):
        review_mask(p, 0.5, 0.4)  # tau_high < tau_low


def test_selected_bound_is_within_alpha_on_synthetic_data():
    p, y = _synthetic(n=4000, seed=3)
    res = select_double_threshold(p, y, alpha=0.05, delta=0.1, n_low=20, n_high=20)
    assert res.feasible
    assert res.risk_upper_bound <= 0.05
    # 风险定义与 missed_mass 一致
    assert res.empirical_risk == pytest.approx(
        missed_mass(p, y, res.tau_low), abs=1e-12
    )
    # 条件漏检率 = R_hat / release_rate
    release_rate = res.n_release / res.n_cal
    assert res.conditional_miss_rate == pytest.approx(
        res.empirical_risk / max(release_rate, 1e-12), abs=1e-12
    )


def test_selection_is_deterministic_with_fixed_seed():
    p, y = _synthetic(n=4000, seed=4)
    a = select_double_threshold(p, y, alpha=0.05, delta=0.1, n_low=20, n_high=20)
    b = select_double_threshold(p, y, alpha=0.05, delta=0.1, n_low=20, n_high=20)
    assert a == b
    assert a.feasible
    assert a.tau_low < a.tau_high
