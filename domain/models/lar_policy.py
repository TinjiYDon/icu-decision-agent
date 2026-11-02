"""LAR-Policy · Label-Ambiguity-Robust Review Policy（标签歧义鲁棒的复核双阈值）。

**动机**（见 `docs/LARE.md`、`docs/LABEL_AUDIT.md`）：`patients.dod` 给出的死亡标签是
`admissions.deathtime` 的**粗化超集**，两套口径下同一策略的经验表现与最优阈值都不同。
若按其中一套标签调阈值，部署到另一套口径（或反过来）就会付出“口径切换代价”。

LAR-Policy 的做法是 **min-max 鲁棒选择**：在同一个候选网格上，要求策略在**两套标签下都**
满足约束，并最小化两套标签中**较差的那一个**的期望代价：

    min_{tau_low < tau_high}  max( cost_dod(tau), cost_death(tau) )
    s.t.  (n_review + n_alert) / n <= K                  # 复核+升级预算
          E[1{y_dod=1}    · 1{p < tau_low}] <= alpha      # 放行区漏检质量（粗化标签）
          E[1{y_death=1}  · 1{p < tau_low}] <= alpha      # 放行区漏检质量（精确标签）

期望代价（与 `selective_review.CostModel` 同口径）：

    cost = mean( cost_fn · 1{y=1, release}
               + cost_fp · 1{y=0, alert}
               + review_cost · 1{review} )

- `y=1, release` → 漏报（最贵），`y=0, alert` → 误报，`review` → 人力成本；
- 复核区**不产生**漏报/误报代价（假定复核能纠正），只计 `review_cost`；
- 默认 `cost_fn=4, cost_fp=1, review_cost=0.2`，与 H4/H9 一致。

同分（robust_cost 相同）时取**放行人数最多**的候选；再同分时取更小的 `tau_high`
（复核带更宽、更保守）。无可行候选 → `feasible=False`、`(tau_low, tau_high) = (0, 1)`
（全部进复核）、`robust_cost = inf`。

纯函数、只依赖 `numpy`（复用 `domain.models.lare` 的掩码口径），不读 DB、不跑 MIMIC、无网络。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from domain.models.lare import double_policy

__all__ = [
    "LARPolicy",
    "expected_cost",
    "fit_lar_policy",
    "label_specific_policies",
    "max_regret",
]

INF = float("inf")


@dataclass(frozen=True)
class LARPolicy:
    """LAR 鲁棒双阈值选择结果。

    `robust_cost = max(cost_dod, cost_death)`；不可行时三项均为 `inf`，
    `tau_low=0.0, tau_high=1.0`（全部进复核）、`n_release=0`。
    """

    tau_low: float
    tau_high: float
    feasible: bool
    robust_cost: float
    cost_dod: float
    cost_death: float
    n_release: int


def _validate_inputs(p, y_dod, y_death, *, K, alpha, n_grid, cost_fn, cost_fp, review_cost) -> tuple:
    scores = np.clip(np.asarray(p, dtype=float).reshape(-1), 0.0, 1.0)
    if scores.size == 0:
        raise ValueError("p 不能为空")
    dod = np.asarray(y_dod, dtype=float).reshape(-1)
    death = np.asarray(y_death, dtype=float).reshape(-1)
    for name, arr in (("y_dod", dod), ("y_death", death)):
        if arr.size != scores.size:
            raise ValueError(f"{name} 与 p 长度不一致: {arr.size} vs {scores.size}")
        if not np.all(np.isin(arr, (0.0, 1.0))):
            raise ValueError(f"{name} 只接受 0/1 标签")
    if not 0.0 <= float(K) <= 1.0:
        raise ValueError("K 必须落在 [0, 1]")
    if not float(alpha) > 0.0:
        raise ValueError("alpha 必须 > 0")
    if int(n_grid) < 2:
        raise ValueError("n_grid 至少为 2")
    for name, value in (("cost_fn", cost_fn), ("cost_fp", cost_fp), ("review_cost", review_cost)):
        if not float(value) >= 0.0:
            raise ValueError(f"{name} 必须 >= 0")
    return scores, dod, death


def _grid(p: np.ndarray, y: np.ndarray, lows: np.ndarray, highs: np.ndarray, *,
          cost_fn: float, cost_fp: float, review_cost: float) -> tuple:
    """在网格上解析计算 代价矩阵 + 每行（每个 tau_low）的放行统计。

    `release` 只依赖 `tau_low`、`alert` 只依赖 `tau_high`（因为要求 `tau_low < tau_high`），
    `review` 是两者之差，因此

        cost[i, j] = cost_fn·R_i + cost_fp·A_j + review_cost·(q_i - g_j)

    其中 `R_i = P(y=1, p<low_i)`、`A_j = P(y=0, p>=high_j)`、`q_i = P(p>=low_i)`、
    `g_j = P(p>=high_j)`。这样 50×50 网格只需两次前缀统计，无 O(n·m) 掩码开销。
    """
    n = p.size
    pos = y == 1.0
    below = p[:, None] < lows[None, :]          # (n, n_low)
    above = p[:, None] >= highs[None, :]        # (n, n_high)
    risk = np.count_nonzero(pos[:, None] & below, axis=0) / n          # R_i
    n_release = np.count_nonzero(below, axis=0)                        # 放行人数
    ge_low = 1.0 - n_release / n                                       # q_i = P(p>=low_i)
    n_alert = np.count_nonzero(above, axis=0)                          # 升级人数
    neg_alert = np.count_nonzero((~pos)[:, None] & above, axis=0) / n  # A_j
    ge_high = n_alert / n                                              # g_j = P(p>=high_j)

    review_rate = ge_low[:, None] - ge_high[None, :]
    cost = (
        float(cost_fn) * risk[:, None]
        + float(cost_fp) * neg_alert[None, :]
        + float(review_cost) * review_rate
    )
    return cost, risk, n_release, review_rate


def expected_cost(y, p, tau_low, tau_high, *, cost_fn: float = 4.0,
                  cost_fp: float = 1.0, review_cost: float = 0.2) -> float:
    """给定双阈值的期望代价。

    `mean(cost_fn·1{y=1,release} + cost_fp·1{y=0,alert} + review_cost·1{review})`；
    其中 `release: p<tau_low`、`review: tau_low<=p<tau_high`、`alert: p>=tau_high`。
    `review_cost` 是**每次复核**的平均人力成本（含解释阅读/确认），与漏报/误报同量纲。
    """
    scores = np.clip(np.asarray(p, dtype=float).reshape(-1), 0.0, 1.0)
    labels = np.asarray(y, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError("p 不能为空")
    if labels.size != scores.size:
        raise ValueError(f"y 与 p 长度不一致: {labels.size} vs {scores.size}")
    if not np.all(np.isin(labels, (0.0, 1.0))):
        raise ValueError("y 只接受 0/1 标签")
    for name, value in (("cost_fn", cost_fn), ("cost_fp", cost_fp), ("review_cost", review_cost)):
        if not float(value) >= 0.0:
            raise ValueError(f"{name} 必须 >= 0")
    masks = double_policy(scores, tau_low, tau_high)
    rel, rev, alert = masks["release"], masks["review"], masks["alert"]
    total = (
        float(cost_fn) * np.mean((labels == 1.0) & rel)
        + float(cost_fp) * np.mean((labels == 0.0) & alert)
        + float(review_cost) * np.mean(rev)
    )
    return float(total)


def _select(cost: np.ndarray, ok_low: np.ndarray, lows: np.ndarray, highs: np.ndarray,
            n_release: np.ndarray) -> tuple[float, float, float, int] | None:
    """在 `ok_low` × 上三角中取最小代价；同分取放行最多、再取更小的 tau_high。"""
    valid = ok_low[:, None] & (lows[:, None] < highs[None, :])
    if not valid.any():
        return None
    best_key = None
    best = None
    for i, j in zip(*np.nonzero(valid)):
        c = float(cost[i, j])
        key = (-round(c, 12), int(n_release[i]), -float(highs[j]))
        if best_key is None or key > best_key:
            best_key = key
            best = (float(lows[i]), float(highs[j]), c, int(n_release[i]))
    return best


def _ok_low(risks: list[np.ndarray], rel_rate: np.ndarray, K: float, alpha: float) -> np.ndarray:
    ok = rel_rate >= 1.0 - float(K) - 1e-12
    for risk in risks:
        ok = ok & (risk <= float(alpha) + 1e-12)
    return ok


def fit_lar_policy(p, y_dod, y_death, *, K: float = 0.2, alpha: float = 0.01,
                   n_grid: int = 50, cost_fn: float = 4.0, cost_fp: float = 1.0,
                   review_cost: float = 0.2) -> LARPolicy:
    """在两套标签下做 min-max 鲁棒双阈值选择，返回 `LARPolicy`。"""
    scores, dod, death = _validate_inputs(
        p, y_dod, y_death, K=K, alpha=alpha, n_grid=n_grid,
        cost_fn=cost_fn, cost_fp=cost_fp, review_cost=review_cost,
    )
    lows = np.linspace(0.0, 1.0, int(n_grid))
    highs = np.linspace(0.0, 1.0, int(n_grid))

    c_dod, r_dod, n_rel, _ = _grid(
        scores, dod, lows, highs, cost_fn=cost_fn, cost_fp=cost_fp, review_cost=review_cost
    )
    c_death, r_death, _, _ = _grid(
        scores, death, lows, highs, cost_fn=cost_fn, cost_fp=cost_fp, review_cost=review_cost
    )
    robust = np.maximum(c_dod, c_death)
    ok = _ok_low([r_dod, r_death], n_rel / scores.size, K, alpha)
    best = _select(robust, ok, lows, highs, n_rel)
    if best is None:
        return LARPolicy(
            tau_low=0.0,
            tau_high=1.0,
            feasible=False,
            robust_cost=INF,
            cost_dod=INF,
            cost_death=INF,
            n_release=0,
        )
    low, high, _, n_release = best
    cost_dod = expected_cost(dod, scores, low, high, cost_fn=cost_fn, cost_fp=cost_fp,
                             review_cost=review_cost)
    cost_death = expected_cost(death, scores, low, high, cost_fn=cost_fn, cost_fp=cost_fp,
                               review_cost=review_cost)
    return LARPolicy(
        tau_low=low,
        tau_high=high,
        feasible=True,
        robust_cost=float(max(cost_dod, cost_death)),
        cost_dod=float(cost_dod),
        cost_death=float(cost_death),
        n_release=int(n_release),
    )


def label_specific_policies(p, y_dod, y_death, **kw) -> dict:
    """按**各自标签单独**选双阈值（网格与约束同 `fit_lar_policy`）。

    返回 `{"dod": {...}, "death": {...}}`，每项含
    `tau_low / tau_high / feasible / cost / n_release`。
    不可行时 `tau_low=0.0, tau_high=1.0, cost=inf, n_release=0`。
    """
    K = kw.pop("K", 0.2)
    alpha = kw.pop("alpha", 0.01)
    n_grid = kw.pop("n_grid", 50)
    cost_fn = kw.pop("cost_fn", 4.0)
    cost_fp = kw.pop("cost_fp", 1.0)
    review_cost = kw.pop("review_cost", 0.2)
    if kw:
        raise ValueError(f"未知参数: {sorted(kw)}")
    scores, dod, death = _validate_inputs(
        p, y_dod, y_death, K=K, alpha=alpha, n_grid=n_grid,
        cost_fn=cost_fn, cost_fp=cost_fp, review_cost=review_cost,
    )
    lows = np.linspace(0.0, 1.0, int(n_grid))
    highs = np.linspace(0.0, 1.0, int(n_grid))

    out: dict[str, dict] = {}
    for name, labels in (("dod", dod), ("death", death)):
        cost, risk, n_rel, _ = _grid(
            scores, labels, lows, highs, cost_fn=cost_fn, cost_fp=cost_fp,
            review_cost=review_cost,
        )
        ok = _ok_low([risk], n_rel / scores.size, K, alpha)
        best = _select(cost, ok, lows, highs, n_rel)
        if best is None:
            out[name] = {
                "tau_low": 0.0,
                "tau_high": 1.0,
                "feasible": False,
                "cost": INF,
                "n_release": 0,
            }
        else:
            low, high, c, n_release = best
            out[name] = {
                "tau_low": low,
                "tau_high": high,
                "feasible": True,
                "cost": float(c),
                "n_release": int(n_release),
            }
    return out


def _policy_taus(policy) -> tuple[float, float]:
    if isinstance(policy, LARPolicy):
        return float(policy.tau_low), float(policy.tau_high)
    if isinstance(policy, dict) and "tau_low" in policy and "tau_high" in policy:
        return float(policy["tau_low"]), float(policy["tau_high"])
    if isinstance(policy, (tuple, list)) and len(policy) == 2:
        return float(policy[0]), float(policy[1])
    raise ValueError("policy 需为 LARPolicy / 含 tau_low,tau_high 的 dict / 二元组")


def max_regret(p, y_dod, y_death, policy, **kw) -> float:
    """相对“标签专属最优”的最大后悔：`max(cost_dod - opt_dod, cost_death - opt_death)`。

    `opt_*` 由 `label_specific_policies` 在同一网格/同一约束下选出；若某标签下
    专属最优不可行（`cost = inf`），该标签的后悔按 `0.0` 计（无有限参照）。
    """
    scores = np.clip(np.asarray(p, dtype=float).reshape(-1), 0.0, 1.0)
    dod = np.asarray(y_dod, dtype=float).reshape(-1)
    death = np.asarray(y_death, dtype=float).reshape(-1)
    low, high = _policy_taus(policy)
    opts = label_specific_policies(scores, dod, death, **kw)
    regret = -INF
    for name, labels in (("dod", dod), ("death", death)):
        cost = expected_cost(
            labels, scores, low, high,
            cost_fn=kw.get("cost_fn", 4.0),
            cost_fp=kw.get("cost_fp", 1.0),
            review_cost=kw.get("review_cost", 0.2),
        )
        opt = float(opts[name]["cost"])
        regret = max(regret, 0.0 if not np.isfinite(opt) else max(0.0, cost - opt))
    return float(max(regret, 0.0))
