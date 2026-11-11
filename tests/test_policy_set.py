"""ε-最优策略集单测：纯合成数据，不读数据库 / 不跑 MIMIC / 无网络。"""

import numpy as np
import pytest

from domain.models.lar_policy import expected_cost, fit_lar_policy
from domain.models.lare import double_policy
from domain.models.policy_set import enumerate_policies, epsilon_optimal, summarize_set


def _synthetic(n=2000, seed=0):
    """两套标签：`y_dod` 是 `y_death` 的粗化超集（附加阳性随 p 增加）。"""
    rng = np.random.default_rng(seed)
    p = rng.beta(0.5, 5.0, size=n)
    y_death = (rng.random(n) < 0.2 * p + 0.01).astype(float)
    extra = (rng.random(n) < 0.4 * p).astype(float)
    y_dod = np.maximum(y_death, extra)
    return p, y_dod, y_death


# 1 —— 候选数与网格结构
def test_enumerate_grid_size_and_ordering():
    p, dod, death = _synthetic(200)
    recs = enumerate_policies(p, dod, death, n_grid=5, K=0.5, alpha=0.5)
    assert len(recs) == 5 * 4 // 2 == 10
    lows = sorted({rec["tau_low"] for rec in recs})
    highs = sorted({rec["tau_high"] for rec in recs})
    # 同一 linspace：两端只在另一侧缺失（tau_low<tau_high 才配对）
    assert sorted(set(lows) | {1.0}) == sorted(set(highs) | {0.0})
    assert lows and highs and lows[0] == 0.0 and highs[-1] == 1.0
    assert all(rec["tau_low"] < rec["tau_high"] for rec in recs)
    # 顺序稳定：tau_low 外层升序、tau_high 内层升序
    order = [(rec["tau_low"], rec["tau_high"]) for rec in recs]
    assert order == sorted(order)


# 2 —— 键完整性 + 类型契约
def test_record_keys_and_types():
    p, dod, death = _synthetic(300)
    recs = enumerate_policies(p, dod, death, n_grid=4, K=0.4, alpha=0.3)
    keys = {
        "tau_low",
        "tau_high",
        "cost_dod",
        "cost_death",
        "robust_cost",
        "risk_dod",
        "risk_death",
        "budget",
        "n_release",
        "feasible",
    }
    for rec in recs:
        assert set(rec) == keys
        assert isinstance(rec["n_release"], int)
        assert isinstance(rec["feasible"], (bool, np.bool_))
        assert rec["robust_cost"] == pytest.approx(max(rec["cost_dod"], rec["cost_death"]))


# 3 —— 代价与 lar_policy.expected_cost 同口径（跨模块一致性，防公式漂移）
def test_cost_matches_lar_policy_expected_cost():
    p, dod, death = _synthetic(400, seed=1)
    recs = enumerate_policies(p, dod, death, n_grid=8, K=0.3, alpha=0.2)
    sample = recs[:: len(recs) // 7][:8]
    for rec in sample:
        assert rec["cost_dod"] == pytest.approx(
            expected_cost(dod, p, rec["tau_low"], rec["tau_high"]), abs=1e-12
        )
        assert rec["cost_death"] == pytest.approx(
            expected_cost(death, p, rec["tau_low"], rec["tau_high"]), abs=1e-12
        )


# 4 —— 风险 / 预算 / 可行性与文档定义一致
def test_risk_budget_and_feasibility_definitions():
    p, dod, death = _synthetic(500, seed=2)
    K, alpha = 0.25, 0.04
    recs = enumerate_policies(p, dod, death, n_grid=6, K=K, alpha=alpha)
    for rec in recs[::7]:
        masks = double_policy(p, rec["tau_low"], rec["tau_high"])
        review_or_alert = masks["review"] | masks["alert"]
        assert rec["budget"] == pytest.approx(float(np.mean(review_or_alert)))
        assert rec["n_release"] == int(np.count_nonzero(masks["release"]))
        assert rec["risk_dod"] == pytest.approx(float(np.mean((dod == 1.0) & masks["release"])))
        assert rec["risk_death"] == pytest.approx(
            float(np.mean((death == 1.0) & masks["release"]))
        )
        expect = (
            rec["budget"] <= K + 1e-12
            and rec["risk_dod"] <= alpha + 1e-12
            and rec["risk_death"] <= alpha + 1e-12
        )
        assert bool(rec["feasible"]) is expect


# 5 —— 松弛 K/alpha，可行候选单调不减
def test_feasible_set_monotone_in_K_and_alpha():
    p, dod, death = _synthetic(400, seed=3)
    tight = enumerate_policies(p, dod, death, n_grid=6, K=0.1, alpha=0.02)
    loose = enumerate_policies(p, dod, death, n_grid=6, K=0.4, alpha=0.6)
    assert sum(r["feasible"] for r in loose) >= sum(r["feasible"] for r in tight)
    assert sum(r["feasible"] for r in tight) >= 0


# 6 —— ε-最优：包含最优点、ε 单调、ε=0 即精确最优集
def test_epsilon_optimal_contains_optimum_and_is_monotone():
    p, dod, death = _synthetic(800, seed=4)
    recs = enumerate_policies(p, dod, death, n_grid=10, K=0.3, alpha=0.1)
    eps = 0.004
    small = epsilon_optimal(recs, eps)
    big = epsilon_optimal(recs, 10 * eps)
    assert len(small) >= 1
    assert len(big) >= len(small)

    feasible = [r for r in recs if r["feasible"]]
    if feasible:
        opt = min(r["robust_cost"] for r in feasible)
        assert any(r["robust_cost"] == pytest.approx(opt) for r in small)
        assert all(r["feasible"] for r in small)
        assert all(r["robust_cost"] <= opt + eps + 1e-12 for r in small)
        assert all(r["robust_cost"] <= opt + 10 * eps + 1e-12 for r in big)
    # ε=0 → 只有严格最优（同分并列可多条）
    zero = epsilon_optimal(recs, 0.0)
    if feasible:
        opt = min(r["robust_cost"] for r in feasible)
        assert all(r["robust_cost"] == pytest.approx(opt, abs=1e-12) for r in zero)


# 7 —— ε-最优的顺序保持输入顺序
def test_epsilon_optimal_preserves_input_order():
    p, dod, death = _synthetic(300, seed=5)
    recs = enumerate_policies(p, dod, death, n_grid=8, K=0.5, alpha=0.5)
    chosen = epsilon_optimal(recs, 1.0)
    positions = [i for i, rec in enumerate(recs) if any(rec is c for c in chosen)]
    assert positions == sorted(positions)


# 8 —— 无可行记录 → []（不可行本身是结论，不抛异常）
def test_epsilon_optimal_empty_when_infeasible():
    p, dod, death = _synthetic(200, seed=6)
    recs = enumerate_policies(p, dod, death, n_grid=5, K=0.0, alpha=1e-9)
    assert all(not r["feasible"] for r in recs)
    assert epsilon_optimal(recs, 0.004) == []
    assert epsilon_optimal([], 1.0) == []


# 9 —— epsilon 校验
def test_epsilon_optimal_rejects_negative_epsilon():
    p, dod, death = _synthetic(200, seed=7)
    recs = enumerate_policies(p, dod, death, n_grid=4, K=0.5, alpha=0.5)
    with pytest.raises(ValueError):
        epsilon_optimal(recs, -0.001)
    with pytest.raises(ValueError):
        epsilon_optimal(recs, float("nan"))
    with pytest.raises(ValueError):
        epsilon_optimal([{"tau_low": 0.1, "tau_high": 0.2}], 0.1)


# 10 —— 集合摘要：键、区间自洽、代价跨度 <= ε
def test_summarize_set_keys_and_ranges():
    p, dod, death = _synthetic(1000, seed=8)
    recs = enumerate_policies(p, dod, death, n_grid=10, K=0.3, alpha=0.1)
    eps = 0.004
    chosen = epsilon_optimal(recs, eps)
    summary = summarize_set(p, chosen)
    expected_keys = {
        "n_policies",
        "tau_low_min",
        "tau_low_max",
        "tau_high_min",
        "tau_high_max",
        "robust_cost_min",
        "robust_cost_max",
        "n_release_min",
        "n_release_max",
        "consensus_rate",
        "contested_rate",
    }
    assert set(summary) == expected_keys
    assert summary["n_policies"] == len(chosen)
    assert summary["tau_low_min"] <= summary["tau_low_max"]
    assert summary["tau_high_min"] <= summary["tau_high_max"]
    assert summary["n_release_min"] <= summary["n_release_max"]
    assert summary["robust_cost_min"] <= summary["robust_cost_max"]
    assert summary["robust_cost_max"] - summary["robust_cost_min"] <= eps + 1e-12
    assert 0.0 <= summary["consensus_rate"] <= 1.0
    assert 0.0 <= summary["contested_rate"] <= 1.0
    # 共识区 ⊆ 每条策略的复核集 ⊆ budget（review|alert） → 上界由最小 budget 给出
    assert summary["consensus_rate"] <= min(rec["budget"] for rec in chosen) + 1e-12


# 11 —— 共识 / 争议比例与直接重建掩码一致
def test_summarize_set_rates_match_manual_masks():
    p, dod, death = _synthetic(600, seed=9)
    recs = enumerate_policies(p, dod, death, n_grid=8, K=0.4, alpha=0.3)
    chosen = epsilon_optimal(recs, 0.02)
    summary = summarize_set(p, chosen)
    masks = [double_policy(p, rec["tau_low"], rec["tau_high"])["review"] for rec in chosen]
    inter = np.logical_and.reduce(masks)
    union = np.logical_or.reduce(masks)
    n_union = int(np.count_nonzero(union))
    assert summary["consensus_rate"] == pytest.approx(float(np.mean(inter)))
    assert summary["contested_rate"] == pytest.approx(
        float((n_union - int(np.count_nonzero(inter))) / n_union)
    )


# 12 —— 单点集合：共识 = 该策略复核率，争议 = 0
def test_summarize_set_single_policy():
    p, dod, death = _synthetic(400, seed=10)
    recs = enumerate_policies(p, dod, death, n_grid=8, K=0.4, alpha=0.3)
    chosen = epsilon_optimal(recs, 0.0)[:1]
    if not chosen:
        pytest.skip("该合成样本上无可行候选")
    summary = summarize_set(p, chosen)
    review = double_policy(p, chosen[0]["tau_low"], chosen[0]["tau_high"])["review"]
    assert summary["n_policies"] == 1
    assert summary["consensus_rate"] == pytest.approx(float(np.mean(review)))
    assert summary["contested_rate"] == pytest.approx(0.0)
    assert summary["tau_low_min"] == summary["tau_low_max"]


# 13 —— include_rates / max_rate_n 关闭时两键为 None
def test_summarize_set_rates_opt_out():
    p, dod, death = _synthetic(500, seed=11)
    recs = enumerate_policies(p, dod, death, n_grid=6, K=0.5, alpha=0.5)
    chosen = epsilon_optimal(recs, 1.0)
    no_rates = summarize_set(p, chosen, include_rates=False)
    assert no_rates["consensus_rate"] is None and no_rates["contested_rate"] is None
    assert no_rates["n_policies"] == len(chosen)
    capped = summarize_set(p, chosen, max_rate_n=10)
    assert capped["consensus_rate"] is None and capped["contested_rate"] is None


# 14 —— 空集合：区间为 None，不抛异常
def test_summarize_set_empty():
    p = np.array([0.1, 0.2, 0.3])
    summary = summarize_set(p, [])
    assert summary["n_policies"] == 0
    assert summary["tau_low_min"] is None
    assert summary["consensus_rate"] is None
    assert summary["contested_rate"] is None


# 15 —— 输入校验（长度 / 标签 / 网格 / K / alpha / 代价）
def test_input_validation():
    p = np.array([0.1, 0.4, 0.7, 0.9])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    enumerate_policies(p, y, y, n_grid=3)

    with pytest.raises(ValueError):
        enumerate_policies(p, y[:-1], y, n_grid=3)
    with pytest.raises(ValueError):
        enumerate_policies(p, y, y[:-1], n_grid=3)
    with pytest.raises(ValueError):
        enumerate_policies(p, np.array([0.0, 2.0, 0.0, 1.0]), y, n_grid=3)
    with pytest.raises(ValueError):
        enumerate_policies(p, y, np.array([0.0, 0.5, 0.0, 1.0]), n_grid=3)
    with pytest.raises(ValueError):
        enumerate_policies(p, y, y, n_grid=1)
    with pytest.raises(ValueError):
        enumerate_policies(p, y, y, n_grid=3, K=1.5)
    with pytest.raises(ValueError):
        enumerate_policies(p, y, y, n_grid=3, K=-0.1)
    with pytest.raises(ValueError):
        enumerate_policies(p, y, y, n_grid=3, alpha=0.0)
    with pytest.raises(ValueError):
        enumerate_policies(p, y, y, n_grid=3, alpha=-0.5)
    with pytest.raises(ValueError):
        enumerate_policies(p, y, y, n_grid=3, cost_fn=-1.0)
    with pytest.raises(ValueError):
        enumerate_policies(p, y, y, n_grid=3, cost_fp=-1e-9)
    with pytest.raises(ValueError):
        enumerate_policies(p, y, y, n_grid=3, review_cost=-0.1)


# 16 —— 与 fit_lar_policy 一致：选中点必须是 ε-最优集合成员（ε>=0）
def test_selected_point_belongs_to_epsilon_set():
    p, dod, death = _synthetic(1200, seed=12)
    K, alpha, n_grid = 0.3, 0.05, 20
    recs = enumerate_policies(p, dod, death, n_grid=n_grid, K=K, alpha=alpha)
    chosen = epsilon_optimal(recs, 0.004)
    policy = fit_lar_policy(p, dod, death, n_grid=n_grid, K=K, alpha=alpha, bound="none")
    if not policy.feasible or not chosen:
        pytest.skip("该合成样本上无可行候选，一致性检查不适用")
    match = [
        rec
        for rec in recs
        if rec["tau_low"] == pytest.approx(policy.tau_low)
        and rec["tau_high"] == pytest.approx(policy.tau_high)
    ]
    assert match, "fit_lar_policy 的选中阈值必须落在同一网格上"
    assert match[0]["feasible"]
    assert match[0]["robust_cost"] <= policy.robust_cost + 1e-9
    assert match[0] in chosen


# 17 —— 冒烟：ACCEPTANCE 的命令级用例（默认 50 网格）
def test_smoke_default_grid():
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    death = (rng.random(2000) < 0.05).astype(float)
    dod = death.copy()
    recs = enumerate_policies(p, dod, death)
    assert len(recs) == 50 * 49 // 2 == 1225
    chosen = epsilon_optimal(recs, 0.004)
    assert len(chosen) >= 1
    summary = summarize_set(p, chosen)
    assert summary["tau_low_min"] <= summary["tau_low_max"]
    assert 0.0 <= summary["consensus_rate"] <= 1.0
    assert 0.0 <= summary["contested_rate"] <= 1.0
