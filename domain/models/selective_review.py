"""H9 · Selective Budgeted Review (SBR).

把“复核谁”从一次性转交升级为**有限时域、预算约束的最优停止**：

    状态 x = (p, k, t)        p=风险分, k=剩余复核次数, t=时刻
    动作 a = 放行 / 升级 / 复核 / 等待
    代价 = 放行漏报 C_FN、升级误报 C_FP、每次复核 c_r
    目标 = 最小化期望代价，s.t. 复核次数 <= K

本模块只做**确定性计算**，不读数据库、不调 LLM；复核者用显式 sensitivity/
specificity 模拟（没有真实医生标签时的离线研究约定）。真实病例轨迹由调用方注入。

与仓库现有模块的关系：
- H4 `utility_calibrate` 给出群体阈值；本模块在其上做病例级、时序的复核决策；
- H5 `sparsity_fusion` 的 p_fuse 可作为本模块的 p；
- H3 人工审计记录可用来估计 reviewer 的 sens/spec（后续）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

RELEASE = 0
ALERT = 1
REVIEW = 2
WAIT = 3

ACTION_NAMES = {RELEASE: "release", ALERT: "alert", REVIEW: "review", WAIT: "wait"}


@dataclass(frozen=True)
class CostModel:
    """临床代价（相对单位）。默认 FN:FP = 4:1，与 H4 的 cost_ratio 一致。"""

    cost_fn: float = 4.0
    cost_fp: float = 1.0
    review_cost: float = 0.2

    def release_cost(self, p):
        return np.asarray(p, dtype=float) * self.cost_fn

    def alert_cost(self, p):
        return (1.0 - np.asarray(p, dtype=float)) * self.cost_fp

    def best_action_cost(self, p):
        return np.minimum(self.release_cost(p), self.alert_cost(p))


@dataclass(frozen=True)
class Reviewer:
    """模拟复核者：P(m=1|y=1)=sensitivity, P(m=0|y=0)=specificity。"""

    sensitivity: float = 0.80
    specificity: float = 0.80

    def posterior(self, p: float, m: int) -> float:
        p = float(np.clip(p, 1e-9, 1 - 1e-9))
        if m == 1:
            num = p * self.sensitivity
            den = num + (1.0 - p) * (1.0 - self.specificity)
        else:
            num = p * (1.0 - self.sensitivity)
            den = num + (1.0 - p) * self.specificity
        return float(np.clip(num / max(den, 1e-12), 0.0, 1.0))

    def positive_rate(self, p):
        p = np.asarray(p, dtype=float)
        return p * self.sensitivity + (1.0 - p) * (1.0 - self.specificity)

    def sample(self, p: float, y: int, rng: np.random.Generator) -> int:
        prob_pos = self.sensitivity if y == 1 else (1.0 - self.specificity)
        return int(rng.random() < prob_pos)

    def expected_post_cost(self, p, cost: CostModel):
        """复核后立即按最小期望代价行动的成本（不含复核费）。"""
        p = np.asarray(p, dtype=float)
        q = self.positive_rate(p)
        post_pos = np.array([self.posterior(pi, 1) for pi in p])
        post_neg = np.array([self.posterior(pi, 0) for pi in p])
        return q * cost.best_action_cost(post_pos) + (1.0 - q) * cost.best_action_cost(post_neg)


def default_transition(grid: Sequence[float], persistence: float = 0.85) -> np.ndarray:
    """构造“粘滞随机游走”转移矩阵：留在原格概率 persistence，左右各一半。"""
    g = np.asarray(grid, dtype=float)
    n = len(g)
    t = np.zeros((n, n), dtype=float)
    for i in range(n):
        t[i, i] = persistence
        if i > 0:
            t[i, i - 1] = (1.0 - persistence) / 2.0
        if i < n - 1:
            t[i, i + 1] = (1.0 - persistence) / 2.0
        if i == 0:
            t[i, 0] += (1.0 - persistence) / 2.0
        if i == n - 1:
            t[i, n - 1] += (1.0 - persistence) / 2.0
    return t / t.sum(axis=1, keepdims=True)


def estimate_transition(trajectories: Sequence[Sequence[float]], grid: Sequence[float]) -> np.ndarray:
    """从病例轨迹估计网格转移矩阵（拉普拉斯平滑）。"""
    g = np.asarray(grid, dtype=float)
    n = len(g)
    counts = np.ones((n, n), dtype=float)  # 平滑
    for traj in trajectories:
        idx = [int(np.argmin(np.abs(g - float(p)))) for p in traj]
        for a, b in zip(idx[:-1], idx[1:]):
            counts[a, b] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


@dataclass
class SBRPolicy:
    """DP 解出的策略表；act() 用最近邻网格查表。"""

    grid: np.ndarray
    action: np.ndarray  # (T+1, n_grid, K+1)
    value: np.ndarray   # (T+1, n_grid, K+1)
    cost: CostModel
    reviewer: Reviewer
    budget: int

    def _idx(self, p: float) -> int:
        return int(np.argmin(np.abs(self.grid - float(p))))

    def act(self, t: int, p: float, k: int) -> int:
        t = max(0, min(int(t), self.action.shape[0] - 1))
        k = max(0, min(int(k), self.action.shape[2] - 1))
        return int(self.action[t, self._idx(p), k])


def solve_sbr(
    grid: Sequence[float],
    transition: np.ndarray,
    cost: CostModel,
    reviewer: Reviewer,
    horizon: int,
    budget: int,
) -> SBRPolicy:
    """有限时域 DP。

    约定：horizon = 决策时刻数 - 1（例如 5 个时刻 -> horizon=4）。
    复核后立即行动，不再等待（v1 的简化；见 docs/H9_SBR.md 的局限）。
    """
    g = np.asarray(grid, dtype=float)
    n = len(g)
    klen = int(budget) + 1
    T = int(horizon) + 1
    V = np.zeros((T, n, klen), dtype=float)
    A = np.full((T, n, klen), WAIT, dtype=int)

    rel = cost.release_cost(g)
    alt = cost.alert_cost(g)
    base = np.minimum(rel, alt)
    rev = cost.review_cost + reviewer.expected_post_cost(g, cost)

    for k in range(klen):
        for i in range(n):
            cand = [(rel[i], RELEASE), (alt[i], ALERT)]
            if k > 0:
                cand.append((rev[i], REVIEW))
            val, act = min(cand, key=lambda x: x[0])
            V[T - 1, i, k], A[T - 1, i, k] = val, act

    for t in range(T - 2, -1, -1):
        wait = transition @ V[t + 1]  # (n, klen)
        for k in range(klen):
            for i in range(n):
                cand = [(rel[i], RELEASE), (alt[i], ALERT), (wait[i, k], WAIT)]
                if k > 0:
                    cand.append((rev[i], REVIEW))
                val, act = min(cand, key=lambda x: x[0])
                V[t, i, k], A[t, i, k] = val, act

    return SBRPolicy(grid=g, action=A, value=V, cost=cost, reviewer=reviewer, budget=int(budget))


class FixedThresholdPolicy:
    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def act(self, t: int, p: float, k: int) -> int:
        return ALERT if p >= self.threshold else RELEASE


class DoubleThresholdPolicy:
    def __init__(self, low: float, high: float):
        self.low, self.high = float(low), float(high)

    def act(self, t: int, p: float, k: int) -> int:
        if p >= self.high:
            return ALERT
        if p < self.low:
            return RELEASE
        return REVIEW


class MyopicReviewPolicy:
    """病例级 Δ 排序的贪心版：只在“复核能降低即时期望代价”时复核。"""

    def __init__(self, cost: CostModel, reviewer: Reviewer, threshold: float = 0.5):
        self.cost, self.reviewer, self.threshold = cost, reviewer, float(threshold)

    def act(self, t: int, p: float, k: int) -> int:
        base = ALERT if p >= self.threshold else RELEASE
        if k <= 0:
            return base
        base_cost = float(min(self.cost.release_cost(p), self.cost.alert_cost(p)))
        review_value = self.cost.review_cost + float(self.reviewer.expected_post_cost(np.array([p]), self.cost)[0])
        return REVIEW if review_value < base_cost else base


def _terminal_action(p: float, cost: CostModel) -> int:
    return ALERT if cost.alert_cost(p) <= cost.release_cost(p) else RELEASE


def simulate_case(
    trajectory: Sequence[float],
    y: int,
    policy,
    cost: CostModel,
    reviewer: Reviewer,
    budget: int,
    rng: np.random.Generator,
) -> dict:
    """跑一条病例轨迹，返回代价与复核统计。"""
    k = int(budget)
    reviews = 0
    changed = 0
    total = 0.0
    for t, p in enumerate(trajectory):
        action = policy.act(t, float(p), k)
        if action == WAIT and t == len(trajectory) - 1:
            action = _terminal_action(float(p), cost)
        if action == WAIT:
            continue
        if action == REVIEW and k > 0:
            k -= 1
            reviews += 1
            total += cost.review_cost
            m = reviewer.sample(float(p), int(y), rng)
            p_post = reviewer.posterior(float(p), m)
            action2 = _terminal_action(p_post, cost)
            base = _terminal_action(float(p), cost)
            changed += int(action2 != base)
            total += float(cost.release_cost(p_post) if action2 == RELEASE else cost.alert_cost(p_post))
            return {"cost": total, "reviews": reviews, "changed": changed, "action": action2,
                    "missed": int(action2 == RELEASE and y == 1)}
        total += float(cost.release_cost(p) if action == RELEASE else cost.alert_cost(p))
        return {"cost": total, "reviews": reviews, "changed": changed, "action": action,
                "missed": int(action == RELEASE and y == 1)}
    # 理论上到不了这里
    total += float(cost.best_action_cost(float(trajectory[-1])))
    return {"cost": total, "reviews": reviews, "changed": changed, "action": RELEASE, "missed": 0}


def evaluate_policy(
    policy,
    trajectories: Sequence[Sequence[float]],
    labels: Sequence[int],
    cost: CostModel | None = None,
    reviewer: Reviewer | None = None,
    budget: int = 3,
    seed: int = 0,
) -> dict:
    """在病例集合上评估策略：代价、漏报、报警、复核率/命中率、oracle gap。"""
    cost = cost or CostModel()
    reviewer = reviewer or Reviewer()
    rng = np.random.default_rng(seed)
    rows = [simulate_case(traj, int(y), policy, cost, reviewer, budget, rng)
            for traj, y in zip(trajectories, labels)]
    n = len(rows)
    total_cost = sum(r["cost"] for r in rows) / max(n, 1)
    reviews = sum(r["reviews"] for r in rows)
    changed = sum(r["changed"] for r in rows)
    oracle = float(np.mean([min(cost.cost_fn * y, cost.cost_fp * (1 - y)) for y in labels]))
    return {
        "n": n,
        "mean_cost": round(total_cost, 6),
        "miss_rate": round(sum(r["missed"] for r in rows) / max(n, 1), 6),
        "alert_rate": round(sum(1 for r in rows if r["action"] == ALERT) / max(n, 1), 6),
        "review_rate": round(reviews / max(n, 1), 6),
        "review_yield": round(changed / max(reviews, 1), 6),
        "oracle_cost": round(oracle, 6),
        "oracle_gap": round(total_cost - oracle, 6),
    }


def make_synthetic_cases(
    n: int = 2000,
    horizon: int = 5,
    seed: int = 0,
    persistence: float = 0.9,
) -> tuple[list[list[float]], list[int]]:
    """合成病例轨迹：低风险起始 + 粘滞随机游走；y 由末时刻风险采样。"""
    rng = np.random.default_rng(seed)
    trajectories: list[list[float]] = []
    labels: list[int] = []
    for _ in range(n):
        p = float(rng.beta(0.4, 6.0))
        traj = [p]
        for _t in range(horizon - 1):
            if rng.random() < persistence:
                p = float(np.clip(p + rng.normal(0, 0.04), 0.0, 1.0))
            else:
                p = float(rng.beta(0.6, 4.0))
            traj.append(p)
        trajectories.append(traj)
        labels.append(int(rng.random() < traj[-1]))
    return trajectories, labels
