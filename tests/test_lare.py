"""LARE 评估口径单测：纯合成、不读数据库/网络。"""

import math

import numpy as np
import pytest

from domain.models.lare import (
    budget_compliance,
    build_coarse_label,
    double_policy,
    evaluate_policy,
    fixed_policy,
    missed_imminent,
    priority_spearman,
    review_yield,
    reviewed_jaccard,
)


def _synthetic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.4, 6.0, size=n)
    y_death = (rng.random(n) < 0.15 * p + 0.02).astype(float)
    y_dod = build_coarse_label(y_death, p, add_rate=0.5, rng=np.random.default_rng(seed))
    return p, y_dod, y_death


# 1
def test_coarse_label_is_superset_of_exact_label():
    rng = np.random.default_rng(3)
    p = rng.random(500)
    y_death = (rng.random(500) < 0.1).astype(float)
    y_dod = build_coarse_label(y_death, p, add_rate=0.7, rng=rng)
    assert np.all(y_dod[y_death == 1.0] == 1.0)
    assert set(np.unique(y_dod)).issubset({0.0, 1.0})
    assert y_dod.sum() >= y_death.sum()
    # add_rate=0 -> 完全等于精确标签
    same = build_coarse_label(y_death, p, add_rate=0.0, rng=rng)
    assert np.array_equal(same, y_death)
    # add_rate 越大，附加阳性越多
    low = build_coarse_label(y_death, p, add_rate=0.2, rng=np.random.default_rng(11))
    high = build_coarse_label(y_death, p, add_rate=1.0, rng=np.random.default_rng(11))
    assert high.sum() >= low.sum()


# 2
def test_review_yield_edges():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    # 全部复核 -> 基准率
    assert review_yield(y, np.ones(4, dtype=bool)) == pytest.approx(0.5)
    # 复核集合为空 -> 0.0（文档化契约）
    assert review_yield(y, np.zeros(4, dtype=bool)) == 0.0
    # 只复核阳性 -> 1.0；只复核阴性 -> 0.0
    assert review_yield(y, np.array([False, True, False, True])) == pytest.approx(1.0)
    assert review_yield(y, np.array([True, False, True, False])) == pytest.approx(0.0)


# 3
def test_missed_imminent_boundary():
    y_death = np.array([1.0, 0.0, 1.0, 0.0])
    release = np.array([True, True, False, False])
    # E[1{y=1}·1{release}] = 1/4（只有第一个阳性被放行）
    assert missed_imminent(y_death, release) == pytest.approx(0.25)
    assert missed_imminent(y_death, np.zeros(4, dtype=bool)) == 0.0
    assert missed_imminent(y_death, np.ones(4, dtype=bool)) == pytest.approx(0.5)


# 4
def test_reviewed_jaccard_edges():
    a = np.array([True, True, False, False])
    b = np.array([False, True, True, False])
    # |A∩B|=1, |A∪B|=3
    assert reviewed_jaccard(a, b) == pytest.approx(1.0 / 3.0)
    assert reviewed_jaccard(a, a.copy()) == 1.0
    assert reviewed_jaccard(a, ~a) == 0.0
    # 两个空集合视为完全一致
    empty = np.zeros(4, dtype=bool)
    assert reviewed_jaccard(empty, empty) == 1.0


# 5
def test_priority_spearman_signs_and_degeneracy():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # 3 阴 / 2 阳：秩相关严格为 sqrt(3)/2（平均秩，非 ±1）
    high = priority_spearman(scores, np.array([False, False, False, True, True]))
    low = priority_spearman(scores, np.array([True, True, False, False, False]))
    assert high == pytest.approx(math.sqrt(3.0) / 2.0)
    assert low == pytest.approx(-math.sqrt(3.0) / 2.0)
    # 单调 scores + 高分端复核 -> 强正相关；反之为强负相关
    assert high > 0.8 and low < -0.8
    # 复核/放行全同 -> 方差为 0 -> 0.0
    assert priority_spearman(scores, np.ones(5, dtype=bool)) == 0.0


# 6
def test_budget_compliance():
    mask = np.array([True, True, False, False])  # 50%
    assert budget_compliance(mask, 0.5)
    assert not budget_compliance(mask, 0.49)
    assert budget_compliance(np.zeros(4, dtype=bool), 0.0)
    with pytest.raises(ValueError):
        budget_compliance(mask, 1.5)


# 7
def test_fixed_and_double_policy_masks_are_disjoint_and_cover_all():
    p = np.array([0.0, 0.09, 0.1, 0.29, 0.3, 0.7, 1.0])
    fixed = fixed_policy(p, 0.3)
    assert not fixed["review"].any()
    assert np.array_equal(fixed["release"], p < 0.3)
    assert np.array_equal(fixed["release"] | fixed["review"] | fixed["alert"], np.ones(7, dtype=bool))

    dbl = double_policy(p, 0.1, 0.3)
    rel, rev, alert = dbl["release"], dbl["review"], dbl["alert"]
    assert not np.any(rel & rev)
    assert not np.any(rel & alert)
    assert not np.any(rev & alert)
    assert np.all(rel | rev | alert)
    assert rel.tolist() == [True, True, False, False, False, False, False]
    assert rev.tolist() == [False, False, True, True, False, False, False]
    assert alert.tolist() == [False, False, False, False, True, True, True]
    assert dbl["release"].sum() + dbl["review"].sum() + dbl["alert"].sum() == p.size


# 8
def test_evaluate_policy_reports_two_label_calibers():
    p = np.array([0.05, 0.25, 0.75])
    y_dod = np.array([1.0, 1.0, 1.0])
    y_death = np.array([1.0, 0.0, 1.0])
    masks = double_policy(p, 0.1, 0.5)
    out = evaluate_policy(p, y_dod, y_death, masks, K=0.7, alpha=0.4)
    for key in (
        "yield_dod",
        "yield_death",
        "missed_imminent",
        "release_risk_dod",
        "release_risk_death",
        "budget",
        "compliance",
    ):
        assert key in out
    assert out["yield_dod"] == pytest.approx(1.0)      # 复核区唯一的样本是 dod 阳性
    assert out["yield_death"] == pytest.approx(0.0)    # 但在精确口径下是阴性
    assert out["missed_imminent"] == pytest.approx(1.0 / 3.0)
    assert out["release_risk_dod"] == pytest.approx(1.0 / 3.0)
    assert out["release_risk_death"] == pytest.approx(1.0 / 3.0)
    assert out["budget"] == pytest.approx(2.0 / 3.0)
    assert out["compliance"] is True
    # K 收紧后不合规
    assert evaluate_policy(p, y_dod, y_death, masks, K=0.5)["compliance"] is False
    # alpha 收紧后不合规（放行区漏检质量 1/3 > 0.2）
    assert evaluate_policy(p, y_dod, y_death, masks, alpha=0.2)["compliance"] is False
    assert out["n_release"] + out["n_review"] + out["n_alert"] == 3


# 9
def test_invalid_inputs_raise():
    p = np.linspace(0.0, 1.0, 10)
    y = np.zeros(10)
    with pytest.raises(ValueError):
        build_coarse_label(y[:-1], p)                 # 长度不一致
    with pytest.raises(ValueError):
        build_coarse_label(y, p, add_rate=1.5)        # add_rate 越界
    with pytest.raises(ValueError):
        build_coarse_label(np.where(np.arange(10) % 2 == 0, 0.0, 2.0), p)  # 非 0/1
    with pytest.raises(ValueError):
        review_yield(y, np.ones(9, dtype=bool))       # mask 长度不一致
    with pytest.raises(ValueError):
        reviewed_jaccard(np.ones(5, dtype=bool), np.ones(6, dtype=bool))
    with pytest.raises(ValueError):
        fixed_policy(p, 1.5)
    with pytest.raises(ValueError):
        double_policy(p, 0.5, 0.4)                    # tau_high < tau_low
    # 重叠的三区（release ∩ review 非空）必须报错
    bad = {
        "release": np.ones(10, dtype=bool),
        "review": np.ones(10, dtype=bool),
        "alert": np.zeros(10, dtype=bool),
    }
    with pytest.raises(ValueError):
        evaluate_policy(p, y, y, bad)
    # 缺键
    with pytest.raises(ValueError):
        evaluate_policy(p, y, y, {"release": np.ones(10, dtype=bool)})
    with pytest.raises(ValueError):
        evaluate_policy(p, y, y, double_policy(p, 0.2, 0.5), alpha=0.0)


# 10
def test_two_label_evaluation_on_synthetic_is_consistent():
    p, y_dod, y_death = _synthetic(n=2000, seed=5)
    masks = double_policy(p, 0.1, 0.3)
    out = evaluate_policy(p, y_dod, y_death, masks, K=1.0)
    # 粗化超集 -> 复核 yield 不低于精确口径
    assert out["yield_dod"] >= out["yield_death"] - 1e-12
    # 放行区漏检质量同样是粗化口径 >= 精确口径
    assert out["release_risk_dod"] >= out["release_risk_death"] - 1e-12
    assert out["missed_imminent"] == pytest.approx(out["release_risk_death"])
    assert out["budget"] == pytest.approx(1.0 - masks["release"].mean())
