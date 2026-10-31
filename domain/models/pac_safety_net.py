"""H9-I4 · PAC 双阈值安全网（Learn-then-Test + Hoeffding + Bonferroni）。

把“放行”这件事包上一层**有限样本**保证：

    release : p < tau_low
    review  : tau_low <= p < tau_high
    alert   : p >= tau_high

风险定义为**放行区漏检质量**（不是条件漏检率）：

    R(tau_low) = E[ 1{y=1} · 1{p < tau_low} ]            # 全人群口径
    conditional_miss_rate = R / max(release_rate, eps)   # 放行人群口径（仅报告）

在以 `1-delta` 置信下要求 `R <= alpha`：

1. **候选集**：`tau_low ∈ linspace(0,1,n_low)`，`tau_high ∈ linspace(0,1,n_high)`，
   只保留 `tau_low < tau_high` 的组合；候选数 `m = #pairs`。
2. **Bonferroni**：每个候选单独用 `delta_m = delta / m`。
3. **Hoeffding 上界**（有界变量 ∈ [0,1]）：
   `U = R_hat + sqrt( log(1/delta_m) / (2n) )`，`R_hat = (1/n) Σ 1{y_i=1, p_i<tau_low}`。
4. **LTT 选择**：在 `U <= alpha` 且 `release_rate >= min_release_rate` 的候选中，
   取**放行人数最多**的那个；同分时取更小的 `tau_high`（复核带更宽、更保守）。
   无可行候选 → `feasible=False`，回落到 `(tau_low, tau_high) = (0.0, 1.0)`、`U = 1.0`。

注意 `U` 只依赖 `tau_low` 与 `(n, m, delta)`；`tau_high` 只影响放行/复核/升级三区的划分，
不进入本风险定义（升级区的漏检需另行约束，见 `docs/PAC_SAFETY_NET.md` §6）。

本模块是**确定性纯函数**：不读数据库、不调网络、不跑 MIMIC、无全局状态。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DoubleThresholdResult",
    "alert_mask",
    "hoeffding_radius",
    "missed_mass",
    "release_mask",
    "review_mask",
    "select_double_threshold",
]


@dataclass(frozen=True)
class DoubleThresholdResult:
    """双阈值选择结果（含三区计数与风险上界）。"""

    tau_low: float
    tau_high: float
    feasible: bool
    empirical_risk: float
    risk_upper_bound: float
    alpha: float
    delta: float
    n_cal: int
    n_release: int
    n_review: int
    n_alert: int
    conditional_miss_rate: float


def _as_scores(p) -> np.ndarray:
    arr = np.asarray(p, dtype=float).reshape(-1)
    return np.clip(arr, 0.0, 1.0)


def _as_labels(y, n: int) -> np.ndarray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if arr.size != n:
        raise ValueError(f"p/y 长度不一致: {n} vs {arr.size}")
    if not np.all(np.isin(arr, (0.0, 1.0))):
        raise ValueError("y 只接受 0/1 标签")
    return arr


def _check_pair(p, y) -> tuple[np.ndarray, np.ndarray]:
    scores = _as_scores(p)
    labels = _as_labels(y, scores.size)
    if scores.size == 0:
        raise ValueError("校准集为空")
    return scores, labels


def release_mask(p, tau_low) -> np.ndarray:
    """放行区：`p < tau_low`。"""
    return _as_scores(p) < float(tau_low)


def review_mask(p, tau_low, tau_high) -> np.ndarray:
    """复核区：`tau_low <= p < tau_high`（要求 `tau_low <= tau_high`）。"""
    if float(tau_high) < float(tau_low):
        raise ValueError("需要 tau_high >= tau_low")
    scores = _as_scores(p)
    return (scores >= float(tau_low)) & (scores < float(tau_high))


def alert_mask(p, tau_high) -> np.ndarray:
    """升级区：`p >= tau_high`。"""
    return _as_scores(p) >= float(tau_high)


def missed_mass(p, y, tau_low) -> float:
    """`R_hat = mean(1{y=1} · 1{p<tau_low})`。"""
    scores, labels = _check_pair(p, y)
    return float(np.mean((labels == 1.0) & (scores < float(tau_low))))


def hoeffding_radius(n: int, delta_m: float) -> float:
    """Hoeffding 半宽：`sqrt(log(1/delta_m) / (2n))`。"""
    if n <= 0:
        raise ValueError("n 必须为正")
    if not 0.0 < float(delta_m) < 1.0:
        raise ValueError("delta_m 必须落在 (0, 1)")
    return float(np.sqrt(np.log(1.0 / float(delta_m)) / (2.0 * int(n))))


def _check_hyper(alpha, delta, n_low, n_high, min_release_rate) -> None:
    if not float(alpha) > 0.0:
        raise ValueError("alpha 必须 > 0")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta 必须落在 (0, 1)")
    if int(n_low) < 2 or int(n_high) < 2:
        raise ValueError("n_low / n_high 至少为 2")
    if not 0.0 <= float(min_release_rate) <= 1.0:
        raise ValueError("min_release_rate 必须落在 [0, 1]")


def _infeasible(alpha, delta, n, scores, tau_high=1.0) -> DoubleThresholdResult:
    """无可行候选时的回落结果（显式标记不可用）。"""
    n_release = 0
    n_alert = int(np.sum(scores >= float(tau_high)))
    n_review = int(n) - n_release - n_alert
    return DoubleThresholdResult(
        tau_low=0.0,
        tau_high=float(tau_high),
        feasible=False,
        empirical_risk=0.0,
        risk_upper_bound=1.0,
        alpha=float(alpha),
        delta=float(delta),
        n_cal=int(n),
        n_release=n_release,
        n_review=n_review,
        n_alert=n_alert,
        conditional_miss_rate=0.0,
    )


def select_double_threshold(
    p,
    y,
    alpha: float = 0.01,
    delta: float = 0.1,
    n_low: int = 50,
    n_high: int = 50,
    min_release_rate: float = 0.0,
) -> DoubleThresholdResult:
    """在候选 `(tau_low, tau_high)` 中做 LTT 选择，返回有限样本可行的双阈值。"""
    scores, labels = _check_pair(p, y)
    _check_hyper(alpha, delta, n_low, n_high, min_release_rate)

    n = int(scores.size)
    lows = np.linspace(0.0, 1.0, int(n_low))
    highs = np.linspace(0.0, 1.0, int(n_high))

    # 候选数 m：只保留 tau_low < tau_high
    n_pairs = sum(int(np.count_nonzero(highs > lo)) for lo in lows)
    if n_pairs == 0:  # n_low/n_high>=2 时不会发生，防御性保留
        return _infeasible(alpha, delta, n, scores)

    radius = hoeffding_radius(n, float(delta) / float(n_pairs))
    positives = labels == 1.0

    best_key: tuple[int, float] | None = None
    best: tuple[float, float, float, float] | None = None  # (lo, hi, risk, upper)

    for lo in lows:
        tau_low = float(lo)
        valid = np.nonzero(highs > lo)[0]
        if valid.size == 0:
            continue
        below = scores < lo
        n_release = int(np.count_nonzero(below))
        if n_release / n < float(min_release_rate):
            continue
        risk = float(np.count_nonzero(positives & below) / n)
        upper = risk + radius
        if upper > float(alpha):
            continue
        tau_high = float(highs[int(valid[0])])  # 同分时取更小的 tau_high
        key = (n_release, -tau_high)
        if best_key is None or key > best_key:
            best_key = key
            best = (tau_low, tau_high, risk, upper)

    if best is None:
        return _infeasible(alpha, delta, n, scores)

    tau_low, tau_high, risk, upper = best
    rel = release_mask(scores, tau_low)
    rev = review_mask(scores, tau_low, tau_high)
    alert = ~(rel | rev)  # 与 alert_mask 等价，保证三区互斥且并集为全集
    n_release = int(np.count_nonzero(rel))
    release_rate = n_release / n
    return DoubleThresholdResult(
        tau_low=float(tau_low),
        tau_high=float(tau_high),
        feasible=True,
        empirical_risk=float(risk),
        risk_upper_bound=float(upper),
        alpha=float(alpha),
        delta=float(delta),
        n_cal=n,
        n_release=n_release,
        n_review=int(np.count_nonzero(rev)),
        n_alert=int(np.count_nonzero(alert)),
        conditional_miss_rate=float(risk / max(release_rate, 1e-12)),
    )
