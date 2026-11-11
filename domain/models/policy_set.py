"""ε-最优策略集（epsilon-optimal policy set）。

**为什么需要“集合”而不是“一个阈值”**：`domain.models.lar_policy.fit_lar_policy` 在
50×50 双阈值网格上做 min-max 鲁棒选择，返回**唯一**的 `(tau_low, tau_high)`。但在有限
样本上，最优候选与其邻域的代价差异往往小于抽样噪声（`robust_cost` 差 1e-3 量级），
“唯一最优”是不可识别的——于是下游（人审、汇报、部署）把噪声当成结论。

本模块把选择结果从**单点**扩展为**ε-最优集合**（类似 Lasso 的 ε-最优路径 / Rashomon set）：

    S(ε) = { theta : feasible(theta) 且 robust_cost(theta) <= min_θ robust_cost(θ) + ε }

`S(ε)` 越大说明**决策越不敏感**（好消息：阈值怎么挪都差不多）；`S(ε)` 里 `tau_low` 的
跨度越大、或集合内策略的复核集合分歧（`contested_rate`）越高，说明**结论越不该被
当成唯一解**（需要在汇报里显式标注）。

三个纯函数：

- `enumerate_policies`  : 网格枚举全部候选（含代价 / 风险 / 预算 / 可行性）；
- `epsilon_optimal`     : 从候选中取 ε-最优集合（按 `robust_cost` 门槛）；
- `summarize_set`       : 集合摘要（阈值区间、代价区间、共识区与争议区比例）。

口径与 `lar_policy` **完全一致**（同一网格、同一代价公式、同一约束、同一三区定义）：

    release : p < tau_low          review : tau_low <= p < tau_high        alert : p >= tau_high
    cost    = mean(cost_fn·1{y=1, release} + cost_fp·1{y=0, alert} + review_cost·1{review})
    robust_cost = max(cost_dod, cost_death)                     # min-max 目标
    feasible    = budget <= K 且 risk_dod <= alpha 且 risk_death <= alpha

其中 `budget = mean(review | alert) = P(p >= tau_low)`、`risk = P(y=1 且 release)`
（全人群口径的放行区漏检质量，与 `domain.models.lare.evaluate_policy` 一致）。

纯函数：只依赖 `numpy`（并复用 `domain.models.lare.double_policy` 的重建掩码），
不读 DB、不跑 MIMIC、无网络、无随机性、不调 LLM。
"""

from __future__ import annotations

import numpy as np

from domain.models.lare import double_policy

__all__ = [
    "enumerate_policies",
    "epsilon_optimal",
    "summarize_set",
]

TOL = 1e-12


def _as_scores(p) -> np.ndarray:
    arr = np.asarray(p, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("p 不能为空")
    if not np.all(np.isfinite(arr)):
        raise ValueError("p 含 NaN/inf")
    return np.clip(arr, 0.0, 1.0)


def _as_labels(y, n: int, name: str) -> np.ndarray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} 与 p 长度不一致: {arr.size} vs {n}")
    if not np.all(np.isin(arr, (0.0, 1.0))):
        raise ValueError(f"{name} 只接受 0/1 标签")
    return arr


def _check_costs(cost_fn, cost_fp, review_cost) -> None:
    for name, value in (("cost_fn", cost_fn), ("cost_fp", cost_fp), ("review_cost", review_cost)):
        if not float(value) >= 0.0:
            raise ValueError(f"{name} 必须 >= 0")


class _ThresholdStats:
    """单套标签的**按阈值前缀统计**（排序一次后每个阈值 O(log n) 查询）。

    `n_below(t) = |{p < t}|`、`pos_below(t) = |{y=1 且 p < t}|`、
    `neg_above(t) = |{y=0 且 p >= t}|`；三区计数由前缀差给出，避免逐候选 O(n) 掩码。
    """

    __slots__ = ("n", "pos_prefix", "sp", "total_pos")

    def __init__(self, scores: np.ndarray, labels: np.ndarray) -> None:
        order = np.argsort(scores, kind="stable")
        self.sp = scores[order]
        pos = (labels[order] == 1.0).astype(np.int64)
        self.pos_prefix = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(pos)])
        self.total_pos = int(self.pos_prefix[-1])
        self.n = int(scores.size)

    def idx(self, tau: float) -> int:
        """严格小于 `tau` 的样本数（`p < tau`）。"""
        return int(np.searchsorted(self.sp, float(tau), side="left"))

    def pos_below(self, tau: float) -> int:
        return int(self.pos_prefix[self.idx(tau)])

    def neg_above(self, tau: float) -> int:
        idx = self.idx(tau)
        n_above = self.n - idx
        pos_above = self.total_pos - int(self.pos_prefix[idx])
        return int(n_above - pos_above)


def enumerate_policies(
    p,
    y_dod,
    y_death,
    *,
    n_grid: int = 50,
    K: float = 0.2,
    alpha: float = 0.05,
    cost_fn: float = 4.0,
    cost_fp: float = 1.0,
    review_cost: float = 0.2,
) -> list[dict]:
    """枚举双阈值网格上的**全部**候选策略（`tau_low < tau_high`）。

    `lows = highs = linspace(0, 1, n_grid)`，只保留 `tau_low < tau_high` 的
    `n_grid·(n_grid-1)/2` 对：即网格上三角，与 `fit_lar_policy` 的候选集合相同。

    每个候选返回 dict（键固定）：

    - `tau_low` / `tau_high`：双阈值；
    - `cost_dod` / `cost_death`：两套标签各自的经验期望代价（公式见模块文档）；
    - `robust_cost`：`max(cost_dod, cost_death)`（min-max 目标，可直接 `min`）；
    - `risk_dod` / `risk_death`：`mean(y=1 且 release)`（放行区漏检质量，全人群口径）；
    - `budget`：`mean(review | alert) = P(p >= tau_low)`（复核 + 升级占比）；
    - `n_release`：放行区人数（整数）；
    - `feasible`：`budget <= K + 1e-12` 且 `risk_dod <= alpha + 1e-12`
      且 `risk_death <= alpha + 1e-12`。

    顺序固定为 `tau_low` 外层、`tau_high` 内层（升序），便于测试与 diff 稳定。
    非法输入（长度不一致、标签非 0/1、`n_grid < 2`、`K ∉ [0,1]`、`alpha <= 0`、
    负代价）抛 `ValueError`。
    """
    scores = _as_scores(p)
    n = int(scores.size)
    dod = _as_labels(y_dod, n, "y_dod")
    death = _as_labels(y_death, n, "y_death")

    if int(n_grid) != n_grid or int(n_grid) < 2:
        raise ValueError("n_grid 必须是 >= 2 的整数")
    if not 0.0 <= float(K) <= 1.0:
        raise ValueError("K 必须落在 [0, 1]")
    if not float(alpha) > 0.0:
        raise ValueError("alpha 必须 > 0")
    _check_costs(cost_fn, cost_fp, review_cost)

    grid = np.linspace(0.0, 1.0, int(n_grid))
    stats_dod = _ThresholdStats(scores, dod)
    stats_death = _ThresholdStats(scores, death)
    k_lim = float(K) + TOL
    a_lim = float(alpha) + TOL
    cf, cp, cr = float(cost_fn), float(cost_fp), float(review_cost)

    records: list[dict] = []
    for i in range(grid.size):
        low = float(grid[i])
        idx_low = stats_dod.idx(low)
        pos_low_dod = int(stats_dod.pos_prefix[idx_low])
        pos_low_death = int(stats_death.pos_prefix[idx_low])
        n_release = idx_low
        budget = (n - n_release) / n
        risk_dod = pos_low_dod / n
        risk_death = pos_low_death / n
        budget_ok = budget <= k_lim
        risk_ok = risk_dod <= a_lim and risk_death <= a_lim
        for j in range(i + 1, grid.size):
            high = float(grid[j])
            idx_high = stats_dod.idx(high)
            n_review = idx_high - idx_low
            review_term = cr * n_review / n
            cost_dod = cf * risk_dod + cp * stats_dod.neg_above(high) / n + review_term
            cost_death = cf * risk_death + cp * stats_death.neg_above(high) / n + review_term
            records.append(
                {
                    "tau_low": low,
                    "tau_high": high,
                    "cost_dod": float(cost_dod),
                    "cost_death": float(cost_death),
                    "robust_cost": float(max(cost_dod, cost_death)),
                    "risk_dod": float(risk_dod),
                    "risk_death": float(risk_death),
                    "budget": float(budget),
                    "n_release": int(n_release),
                    "feasible": bool(budget_ok and risk_ok),
                }
            )
    return records


def epsilon_optimal(records, epsilon) -> list[dict]:
    """取 ε-最优集合：`feasible` 候选中 `robust_cost <= min(robust_cost) + ε` 的那些。

    - 只在 `feasible=True` 的记录里选（不可行策略代价再低也不能部署）；
    - 没有可行记录 → 返回 `[]`（而不是抛异常：不可行本身是一个结论）；
    - `epsilon < 0` 或非有限 → `ValueError`；
    - 返回顺序保持**输入顺序**（不重排、不去重），方便与 `enumerate_policies` 对齐。

    注意 ε 的语义是**代价绝对差**（与 `robust_cost` 同量纲），不是相对比例；
    是否用 `0.004` 之类的小值属于调用方的口径选择，本函数不给默认值。
    """
    eps = float(epsilon)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("epsilon 必须是有限且 >= 0")

    feasible = []
    for rec in records:
        if not isinstance(rec, dict):
            raise TypeError("records 的每个元素必须是 dict")
        for key in ("robust_cost", "feasible", "tau_low", "tau_high"):
            if key not in rec:
                raise ValueError(f"record 缺少键: {key}")
        if bool(rec["feasible"]):
            feasible.append(rec)
    if not feasible:
        return []

    opt = min(float(rec["robust_cost"]) for rec in feasible)
    threshold = opt + eps + TOL
    return [rec for rec in feasible if float(rec["robust_cost"]) <= threshold]


def summarize_set(p, policies, *, include_rates: bool = True, max_rate_n: int = 10000) -> dict:
    """ε-最优集合的摘要（区间 + 共识/争议比例）。

    返回键：

    - `n_policies`：集合大小；
    - `tau_low_min` / `tau_low_max`、`tau_high_min` / `tau_high_max`：阈值跨度
      （跨度越大 → 决策对阈值越不敏感，越不该把它当成唯一解）；
    - `robust_cost_min` / `robust_cost_max`：`summary` 的代价区间
      （`max - min <= ε` 是 ε-最优的直接校验）；
    - `n_release_min` / `n_release_max`：放行人数区间；
    - `consensus_rate`：**所有策略都复核**的样本比例
      `mean(∩_k review_k)`（高 → 集合内部对“谁来复核”共识强）；
    - `contested_rate`：并集里**不是所有人都复核**的比例
      `|∪ - ∩| / |∪|`（0 → 完全一致；1 → 毫无共识）。

    `consensus_rate` / `contested_rate` 需要逐策略重建 `review` 掩码（O(len(p)·n_policies)），
    因此 `include_rates=False` 或 `len(p) > max_rate_n` 时两者为 `None`（不做、不假装做了）。
    集合为空时所有区间键为 `None`、`n_policies=0`；`∪` 为空时两个比例都记 `0.0`。
    """
    scores = _as_scores(p)
    records = list(policies)
    out: dict = {"n_policies": len(records)}
    range_keys = (
        "tau_low_min",
        "tau_low_max",
        "tau_high_min",
        "tau_high_max",
        "robust_cost_min",
        "robust_cost_max",
        "n_release_min",
        "n_release_max",
    )
    out.update({key: None for key in range_keys})
    out["consensus_rate"] = None
    out["contested_rate"] = None
    if not records:
        return out

    taus_low = np.array([float(rec["tau_low"]) for rec in records], dtype=float)
    taus_high = np.array([float(rec["tau_high"]) for rec in records], dtype=float)
    costs = np.array([float(rec["robust_cost"]) for rec in records], dtype=float)
    n_release = np.array([int(rec["n_release"]) for rec in records], dtype=np.int64)
    out.update(
        {
            "tau_low_min": float(taus_low.min()),
            "tau_low_max": float(taus_low.max()),
            "tau_high_min": float(taus_high.min()),
            "tau_high_max": float(taus_high.max()),
            "robust_cost_min": float(costs.min()),
            "robust_cost_max": float(costs.max()),
            "n_release_min": int(n_release.min()),
            "n_release_max": int(n_release.max()),
        }
    )

    if not include_rates or scores.size > int(max_rate_n):
        return out

    intersection = np.ones(scores.size, dtype=bool)
    union = np.zeros(scores.size, dtype=bool)
    for rec in records:
        review = double_policy(scores, rec["tau_low"], rec["tau_high"])["review"]
        intersection &= review
        union |= review
    n_union = int(np.count_nonzero(union))
    n_inter = int(np.count_nonzero(intersection))
    out["consensus_rate"] = float(n_inter / scores.size) if scores.size else 0.0
    out["contested_rate"] = float((n_union - n_inter) / n_union) if n_union else 0.0
    return out
