"""H9 SBR 单元测试：不依赖数据库/网络，全部用合成数据。"""

import numpy as np
import pytest

from domain.models.selective_review import (
    ALERT,
    RELEASE,
    REVIEW,
    CostModel,
    DoubleThresholdPolicy,
    FixedThresholdPolicy,
    MyopicReviewPolicy,
    Reviewer,
    default_transition,
    estimate_transition,
    evaluate_policy,
    make_synthetic_cases,
    simulate_case,
    solve_sbr,
)


def test_reviewer_posterior_and_positive_rate():
    r = Reviewer(sensitivity=0.8, specificity=0.8)
    assert r.posterior(0.5, 1) == pytest.approx(0.8, abs=1e-6)
    assert r.posterior(0.5, 0) == pytest.approx(0.2, abs=1e-6)
    assert r.positive_rate(0.5) == pytest.approx(0.5, abs=1e-6)


def test_default_transition_rows_sum_to_one():
    grid = np.linspace(0, 1, 11)
    t = default_transition(grid, persistence=0.8)
    assert t.shape == (11, 11)
    np.testing.assert_allclose(t.sum(axis=1), 1.0)


def test_estimate_transition_smoothing():
    grid = np.linspace(0, 1, 11)
    trajs = [[0.0, 0.1, 0.2], [0.2, 0.3, 0.4]]
    t = estimate_transition(trajs, grid)
    np.testing.assert_allclose(t.sum(axis=1), 1.0)


def test_dp_budget_zero_matches_terminal_min_cost():
    grid = np.linspace(0, 1, 21)
    cost = CostModel()
    reviewer = Reviewer()
    policy = solve_sbr(grid, default_transition(grid), cost, reviewer, horizon=3, budget=0)
    t_last = policy.action.shape[0] - 1
    for p in (0.1, 0.5, 0.9):
        a = policy.act(t_last, p, 0)
        assert a in (RELEASE, ALERT)
        expected = min(cost.release_cost(p), cost.alert_cost(p))
        idx = policy._idx(p)
        assert policy.value[t_last, idx, 0] == pytest.approx(expected, abs=1e-9)


def test_more_budget_never_increases_expected_cost():
    grid = np.linspace(0, 1, 21)
    cost = CostModel()
    reviewer = Reviewer(sensitivity=0.9, specificity=0.9)
    trans = default_transition(grid)
    v0 = [solve_sbr(grid, trans, cost, reviewer, horizon=4, budget=k).value[0, 10, k]
          for k in (0, 1, 3)]
    assert v0[0] >= v0[1] - 1e-9
    assert v0[1] >= v0[2] - 1e-9


def test_dp_uses_review_for_mid_risk_with_budget():
    grid = np.linspace(0, 1, 21)
    cost = CostModel()
    reviewer = Reviewer(sensitivity=0.95, specificity=0.95)
    policy = solve_sbr(grid, default_transition(grid), cost, reviewer, horizon=4, budget=2)
    t_last = policy.action.shape[0] - 1
    assert policy.act(t_last, 0.5, 2) == REVIEW


def test_myopic_policy_reviews_informative_mid_risk():
    cost = CostModel()
    reviewer = Reviewer(sensitivity=0.95, specificity=0.95)
    policy = MyopicReviewPolicy(cost, reviewer, threshold=0.5)
    assert policy.act(0, 0.5, 1) == REVIEW
    assert policy.act(0, 0.5, 0) in (RELEASE, ALERT)


def test_simulate_case_returns_bounded_stats():
    traj = [0.2, 0.3, 0.4]
    cost = CostModel()
    reviewer = Reviewer()
    rng = np.random.default_rng(0)
    out = simulate_case(traj, 0, FixedThresholdPolicy(0.5), cost, reviewer, budget=2, rng=rng)
    assert set(out) == {"cost", "reviews", "changed", "action", "missed"}
    assert out["cost"] >= 0
    assert out["reviews"] in (0, 1)


def test_evaluate_policy_smoke():
    trajs, labels = make_synthetic_cases(n=200, horizon=5, seed=7)
    cost = CostModel()
    reviewer = Reviewer()
    sbr = solve_sbr(np.linspace(0, 1, 21), default_transition(np.linspace(0, 1, 21)),
                    cost, reviewer, horizon=4, budget=2)
    metrics = evaluate_policy(sbr, trajs, labels, cost, reviewer, budget=2, seed=7)
    assert metrics["n"] == 200
    for key in ("mean_cost", "miss_rate", "alert_rate", "review_rate", "review_yield", "oracle_gap"):
        assert isinstance(metrics[key], float)
        assert np.isfinite(metrics[key])
    assert 0.0 <= metrics["miss_rate"] <= 1.0
    assert 0.0 <= metrics["alert_rate"] <= 1.0


def test_double_threshold_policy():
    p = DoubleThresholdPolicy(0.2, 0.6)
    assert p.act(0, 0.1, 3) == RELEASE
    assert p.act(0, 0.4, 3) == REVIEW
    assert p.act(0, 0.8, 3) == ALERT
