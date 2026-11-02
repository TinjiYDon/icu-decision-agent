"""LAR-Policy（标签歧义鲁棒双阈值）单测：纯合成、不读数据库/网络。"""

import math
from itertools import pairwise

import numpy as np
import pytest

from domain.models.lar_policy import (
    LARPolicy,
    expected_cost,
    fit_lar_policy,
    label_specific_policies,
    max_regret,
)
from domain.models.lare import build_coarse_label, double_policy, evaluate_policy


def _synthetic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.4, 6.0, size=n)
    y_death = (rng.random(n) < 0.15 * p + 0.02).astype(float)
    y_dod = build_coarse_label(y_death, p, add_rate=0.5, rng=np.random.default_rng(seed))
    return p, y_dod, y_death


# 1
def test_expected_cost_matches_manual_definition():
    p = np.array([0.05, 0.25, 0.75])
    y = np.array([1.0, 0.0, 1.0])
    # release: p<0.1 -> 样本0；review: [0.1,0.5) -> 样本1；alert: p>=0.5 -> 样本2
    # cost = 4·1{y=1,release}/3 + 1·1{y=0,alert}/3 + 0.2·1{review}/3
    expected = (4.0 * 1.0 + 1.0 * 0.0 + 0.2 * 1.0) / 3.0
    assert expected_cost(y, p, 0.1, 0.5) == pytest.approx(expected)
    # 全放行 + 全为阴性 -> 复核/误报代价为 0（漏报为 0）
    assert expected_cost(np.zeros(3), p, 1.0, 1.0) == pytest.approx(0.0)


# 2
def test_robust_cost_is_max_of_two_label_costs_when_feasible():
    p, y_dod, y_death = _synthetic(n=2000, seed=0)
    pol = fit_lar_policy(p, y_dod, y_death, K=0.3, alpha=0.1, n_grid=40)
    assert isinstance(pol, LARPolicy)
    assert pol.feasible
    assert pol.tau_low < pol.tau_high
    assert pol.robust_cost == pytest.approx(max(pol.cost_dod, pol.cost_death))
    # 与 expected_cost 的定义一致
    assert pol.cost_dod == pytest.approx(
        expected_cost(y_dod, p, pol.tau_low, pol.tau_high)
    )
    assert pol.cost_death == pytest.approx(
        expected_cost(y_death, p, pol.tau_low, pol.tau_high)
    )
    # 约束确实被满足（两套标签 + 预算）
    out = evaluate_policy(
        p, y_dod, y_death,
        double_policy(p, pol.tau_low, pol.tau_high),
        K=0.3,
        alpha=0.1,
    )
    assert out["compliance"] is True
    assert out["budget"] <= 0.3 + 1e-12


# 3
def test_larger_budget_does_not_increase_robust_cost():
    p, y_dod, y_death = _synthetic(n=2000, seed=1)
    costs = []
    for K in (0.2, 0.3, 0.5, 0.8):
        pol = fit_lar_policy(p, y_dod, y_death, K=K, alpha=0.1, n_grid=40)
        assert pol.feasible
        assert pol.n_release / p.size >= 1.0 - K - 1e-12
        costs.append(pol.robust_cost)
    for prev, nxt in pairwise(costs):
        assert nxt <= prev + 1e-12


# 4
def test_infeasible_alpha_returns_infinite_cost_placeholder():
    p, y_dod, y_death = _synthetic(n=800, seed=2)
    pol = fit_lar_policy(p, y_dod, y_death, K=0.2, alpha=1e-9, n_grid=20)
    assert isinstance(pol, LARPolicy)
    assert not pol.feasible
    assert (pol.tau_low, pol.tau_high) == (0.0, 1.0)
    assert math.isinf(pol.robust_cost) and pol.robust_cost > 0
    assert math.isinf(pol.cost_dod) and math.isinf(pol.cost_death)
    assert pol.n_release == 0
    # 无可行候选时 regret 无有限参照，按 0 计
    assert max_regret(p, y_dod, y_death, pol, K=0.2, alpha=1e-9, n_grid=20) == 0.0


# 5
def test_max_regret_is_nonnegative_and_zero_for_label_specific_policy():
    p, y_dod, y_death = _synthetic(n=2000, seed=3)
    pol = fit_lar_policy(p, y_dod, y_death, K=0.3, alpha=0.1, n_grid=40)
    regret = max_regret(p, y_dod, y_death, pol, K=0.3, alpha=0.1, n_grid=40)
    assert regret >= 0.0
    # 同一个网格 + 同一约束：鲁棒策略的后悔应不小于 0，且不超过“最差标签的优化空间”
    assert regret <= max(pol.cost_dod, pol.cost_death) + 1e-12

    ls = label_specific_policies(p, y_dod, y_death, K=0.3, alpha=0.1, n_grid=40)
    for name, labels in (("dod", y_dod), ("death", y_death)):
        cost = expected_cost(labels, p, ls[name]["tau_low"], ls[name]["tau_high"])
        # 标签专属最优不会比任意策略（含鲁棒策略）更差
        assert cost <= expected_cost(labels, p, pol.tau_low, pol.tau_high) + 1e-12
        assert max_regret(
            p, y_dod, y_death, ls[name], K=0.3, alpha=0.1, n_grid=40
        ) >= 0.0


# 6
def test_label_specific_policies_respect_their_own_constraint():
    p, y_dod, y_death = _synthetic(n=2000, seed=4)
    ls = label_specific_policies(p, y_dod, y_death, K=0.4, alpha=0.1, n_grid=40)
    assert set(ls) == {"dod", "death"}
    for name, labels in (("dod", y_dod), ("death", y_death)):
        row = ls[name]
        assert row["feasible"]
        assert row["tau_low"] < row["tau_high"]
        masks = double_policy(p, row["tau_low"], row["tau_high"])
        budget = float(np.mean(masks["review"] | masks["alert"]))
        assert budget <= 0.4 + 1e-12
        risk = float(np.mean((labels == 1.0) & masks["release"]))
        assert risk <= 0.1 + 1e-12
        assert row["cost"] == pytest.approx(
            expected_cost(labels, p, row["tau_low"], row["tau_high"])
        )


# 7
def test_invalid_inputs_raise():
    p = np.linspace(0.0, 1.0, 100)
    y = np.zeros(100)
    with pytest.raises(ValueError):
        fit_lar_policy(p, y[:-1], y, K=0.2, alpha=0.05)   # 长度不一致
    bad = np.where(np.arange(100) % 2 == 0, 0.0, 2.0)
    with pytest.raises(ValueError):
        fit_lar_policy(p, bad, y)                         # 非 0/1 标签
    with pytest.raises(ValueError):
        fit_lar_policy(p, y, y, K=1.5)                    # K 越界
    with pytest.raises(ValueError):
        fit_lar_policy(p, y, y, alpha=0.0)                # alpha <= 0
    with pytest.raises(ValueError):
        fit_lar_policy(p, y, y, n_grid=1)                 # n_grid < 2
    with pytest.raises(ValueError):
        fit_lar_policy(p, y, y, cost_fn=-1.0)             # 代价为负
    with pytest.raises(ValueError):
        expected_cost(y[:-1], p, 0.1, 0.3)                # 长度不一致
    with pytest.raises(ValueError):
        expected_cost(bad, p, 0.1, 0.3)                   # 非 0/1 标签
    with pytest.raises(ValueError):
        label_specific_policies(p, y, y, unknown=1)       # 未知参数
    with pytest.raises(ValueError):
        max_regret(p, y, y, "not-a-policy")


# 8
def test_hoeffding_bound_infeasible_for_tiny_alpha():
    p, y_dod, y_death = _synthetic(n=800, seed=5)
    # 经验风险在 alpha=1e-3 下已不可行；加上 Hoeffding 半径只会更严
    pol = fit_lar_policy(
        p, y_dod, y_death, K=0.2, alpha=1e-3, n_grid=20, bound="hoeffding"
    )
    assert not pol.feasible
    assert (pol.tau_low, pol.tau_high) == (0.0, 1.0)
    assert math.isinf(pol.robust_cost) and math.isinf(pol.cost_dod)
    assert math.isinf(pol.risk_ub_dod) and math.isinf(pol.risk_ub_death)
    assert pol.n_release == 0


# 9
def test_hoeffding_bound_is_conservative_and_reports_upper_bounds():
    p, y_dod, y_death = _synthetic(n=2000, seed=6)
    kw = {"K": 0.3, "alpha": 0.1, "n_grid": 40}
    plain = fit_lar_policy(p, y_dod, y_death, **kw)
    bounded = fit_lar_policy(p, y_dod, y_death, bound="hoeffding", delta=0.1, **kw)
    assert plain.feasible and bounded.feasible
    # 经验可行的候选集包含“上界可行”的候选集 ⇒ 加界后代价不会更优
    assert bounded.robust_cost >= plain.robust_cost - 1e-12
    # 上界 = 经验放行风险 + Hoeffding 半径，且两套标签都不超 alpha
    n_pairs = 40 * 39 // 2
    radius = math.sqrt(math.log(n_pairs / 0.1) / (2 * p.size))
    masks = double_policy(p, bounded.tau_low, bounded.tau_high)
    for ub, labels in (
        (bounded.risk_ub_dod, y_dod),
        (bounded.risk_ub_death, y_death),
    ):
        empirical = float(np.mean((labels == 1.0) & masks["release"]))
        assert ub == pytest.approx(empirical + radius)
        assert ub <= 0.1 + 1e-12
    # 未启用上界时字段保持 inf（向后兼容）
    assert math.isinf(plain.risk_ub_dod) and math.isinf(plain.risk_ub_death)


# 10
def test_bound_none_matches_default_and_invalid_arguments_raise():
    p, y_dod, y_death = _synthetic(n=1000, seed=7)
    kw = {"K": 0.3, "alpha": 0.1, "n_grid": 30}
    assert fit_lar_policy(p, y_dod, y_death, bound="none", **kw) == fit_lar_policy(
        p, y_dod, y_death, **kw
    )
    with pytest.raises(ValueError):
        fit_lar_policy(p, y_dod, y_death, bound="clopper-pearson", **kw)
    with pytest.raises(ValueError):
        fit_lar_policy(p, y_dod, y_death, bound="hoeffding", delta=1.5, **kw)
    with pytest.raises(ValueError):
        fit_lar_policy(p, y_dod, y_death, bound="hoeffding", delta=0.0, **kw)
