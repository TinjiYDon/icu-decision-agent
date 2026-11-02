"""LARE · Label-Ambiguity Review Evaluation（标签时间歧义下的复核评估）。

**问题**：同一批 ICU 病人可以有两套死亡标签——

    y_death : `admissions.deathtime` 精确到时间戳（12h 窗严格）
    y_dod   : `patients.dod` 日期级（把整天视为死亡区间 → 12h 窗粗化）

P1 标签审计的结论是 `y_dod` 是 `y_death` 的**粗化超集**（dod-only 5,401，占 dod 阳性
48.9%；基准率 1.19% → 2.34%，见 `docs/LABEL_AUDIT.md`）。于是任何“复核策略评估”都会碰到
两个陷阱：

1. **口径切换**：同一个策略在粗化标签下 yield 更高（阳性更多、且多出来的阳性分数偏低，
   更容易落进复核带），看起来“更有效”，但那部分是日期级模糊性，不是真实近因死亡；
2. **目标漂移**：按 A 口径选出来的阈值，被 B 口径复核的人并不相同（reviewed-set 不一致）。

本模块只提供**评估口径**，不引入新标签、不改标签定义：

    review_yield(y, review)      = P(y=1 | review)                # 复核命中率
    missed_imminent(y_death, rel)= E[1{y_death=1} · 1{release}]   # 放行区漏检质量
    reviewed_jaccard(A, B)       = |A∩B| / |A∪B|                  # 两策略复核集合一致度
    priority_spearman(s, mask)   = Spearman(rank(s), rank(1{mask}))# 复核是否集中在高分端
    budget_compliance(mask, K)   = mean(mask) <= K                # 复核+升级总量预算

约定与 `domain.models.pac_safety_net` 对齐：风险一律用**放行区漏检质量**（全人群口径，
`E[1{y=1}·1{release}]`），而不是条件漏检率；三区 `release / review / alert` 互斥且并集为全集。

纯函数：只依赖 `numpy`，不读 DB、不调网络、不跑 MIMIC。除 `build_coarse_label`
（显式要求 `rng`）外无随机性。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "budget_compliance",
    "build_coarse_label",
    "double_policy",
    "evaluate_policy",
    "fixed_policy",
    "missed_imminent",
    "priority_spearman",
    "review_yield",
    "reviewed_jaccard",
]

EPS = 1e-12


def _as_array(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含 NaN/inf")
    return arr


def _as_scores(p) -> np.ndarray:
    return np.clip(_as_array(p, "p"), 0.0, 1.0)


def _as_labels(y, n: int, name: str = "y") -> np.ndarray:
    arr = _as_array(y, name)
    if arr.size != n:
        raise ValueError(f"{name} 与 p 长度不一致: {arr.size} vs {n}")
    if not np.all(np.isin(arr, (0.0, 1.0))):
        raise ValueError(f"{name} 只接受 0/1 标签")
    return arr


def _as_mask(mask, n: int, name: str = "mask") -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} 与 y 长度不一致: {arr.size} vs {n}")
    return arr.astype(bool)


def build_coarse_label(y_death, p, *, add_rate: float = 0.5, rng=None) -> np.ndarray:
    """构造日期级粗化标签 `y_dod`（`y_death` 的超集）。

    `y_dod = y_death | Bernoulli(add_rate * p)`：日期级把整天当作死亡区间，因此
    **原本 12h 内没死、但当天晚些时候死亡**的病人会被算成阳性。这里用“随风险分 p
    线性增加”的假阳性率来模拟该机制（分数越高，当天死亡的概率越高）。

    参数
    ----
    y_death : 0/1 精确标签（时间戳口径）
    p : 风险分，截断到 [0,1]；与 `y_death` 等长
    add_rate : 粗化附加率，必须落在 [0,1]；附加概率为 `add_rate * p`
    rng : `np.random.Generator`；None 时用 `default_rng(0)`（可复现）

    返回
    ----
    `float` 0/1 数组，保证 `y_death=1 → y_dod=1`（严格超集）。
    """
    if not 0.0 <= float(add_rate) <= 1.0:
        raise ValueError("add_rate 必须落在 [0, 1]")
    scores = _as_scores(p)
    exact = _as_labels(y_death, scores.size, "y_death")
    generator = np.random.default_rng(0) if rng is None else rng
    flip = generator.random(scores.size) < float(add_rate) * scores
    return np.where(exact == 1.0, 1.0, flip.astype(float))


def review_yield(y, review_mask) -> float:
    """复核命中率 `P(y=1 | review) = E[1{y=1}·1{review}] / E[1{review}]`。

    复核集合为空时返回 `0.0`（无定义，但下游需要可比较的数值；文档中明确说明）。
    """
    labels = _as_labels(y, _as_array(y, "y").size)
    mask = _as_mask(review_mask, labels.size, "review_mask")
    n_review = int(np.count_nonzero(mask))
    if n_review == 0:
        return 0.0
    return float(np.count_nonzero((labels == 1.0) & mask) / n_review)


def missed_imminent(y_death, release_mask) -> float:
    """放行区漏检质量 `E[1{y_death=1}·1{release}]`（全人群口径）。"""
    labels = _as_labels(y_death, _as_array(y_death, "y_death").size, "y_death")
    mask = _as_mask(release_mask, labels.size, "release_mask")
    return float(np.mean((labels == 1.0) & mask))


def reviewed_jaccard(mask_a, mask_b) -> float:
    """复核集合的 Jaccard 一致度；两个空集合视为完全一致（`1.0`）。"""
    a = _as_array(mask_a, "mask_a") != 0.0
    b = _as_array(mask_b, "mask_b") != 0.0
    if a.size != b.size:
        raise ValueError(f"mask_a / mask_b 长度不一致: {a.size} vs {b.size}")
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b) / union)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """平均秩（tie-aware），1-based。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def priority_spearman(scores, mask) -> float:
    """复核指示与风险分的 Spearman 秩相关（衡量复核集合是否集中在高分端）。

    `rho = corr(rank(scores), rank(1{mask}))`，并列用平均秩处理（不引入 scipy）。
    任一侧方差为 0（例如全部复核 / 全部放行）时返回 `0.0`。
    """
    s = _as_scores(scores)
    m = _as_mask(mask, s.size, "mask").astype(float)
    if s.size < 2:
        return 0.0
    rs = _average_ranks(s)
    rm = _average_ranks(m)
    rs = rs - rs.mean()
    rm = rm - rm.mean()
    denom = float(np.sqrt(np.sum(rs * rs) * np.sum(rm * rm)))
    if denom <= EPS:
        return 0.0
    return float(np.sum(rs * rm) / denom)


def budget_compliance(mask_review_or_alert, K: float) -> bool:
    """预算合规：`mean(mask) <= K`（`K` 为复核+升级占全体的比例上限）。"""
    if not 0.0 <= float(K) <= 1.0:
        raise ValueError("K 必须落在 [0, 1]")
    mask = np.asarray(mask_review_or_alert).reshape(-1).astype(bool)
    if mask.size == 0:
        raise ValueError("mask 不能为空")
    return bool(np.mean(mask) <= float(K) + 1e-12)


def fixed_policy(p, tau) -> dict:
    """单阈值策略：`p < tau` 放行，`p >= tau` 报警，复核区为空。"""
    scores = _as_scores(p)
    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau 必须落在 [0, 1]")
    return {
        "release": scores < tau,
        "review": np.zeros(scores.size, dtype=bool),
        "alert": scores >= tau,
    }


def double_policy(p, tau_low, tau_high) -> dict:
    """双阈值策略：`p < tau_low` 放行，`[tau_low, tau_high)` 复核，`p >= tau_high` 报警。"""
    scores = _as_scores(p)
    low, high = float(tau_low), float(tau_high)
    if not 0.0 <= low <= 1.0 or not 0.0 <= high <= 1.0:
        raise ValueError("tau_low / tau_high 必须落在 [0, 1]")
    if high < low:
        raise ValueError("需要 tau_high >= tau_low")
    release = scores < low
    review = (scores >= low) & (scores < high)
    alert = scores >= high
    # 保底：三区互斥且并集为全集（数值边界上仍然成立）
    alert = ~(release | review)
    return {"release": release, "review": review, "alert": alert}


def _policy_masks(policy_masks, n: int) -> dict:
    if not isinstance(policy_masks, dict):
        raise TypeError("policy_masks 必须是 dict，含 release/review/alert 三个掩码")
    missing = {"release", "review", "alert"} - set(policy_masks)
    if missing:
        raise ValueError(f"policy_masks 缺少键: {sorted(missing)}")
    masks = {
        key: _as_mask(policy_masks[key], n, f"policy_masks[{key}]")
        for key in ("release", "review", "alert")
    }
    # 互斥性检查：评估口径依赖三区划分，重叠会让指标重复计数
    overlap = (masks["release"] & masks["review"]).any() or (
        masks["release"] & masks["alert"]
    ).any() or (masks["review"] & masks["alert"]).any()
    if overlap:
        raise ValueError("release / review / alert 必须互斥")
    return masks


def evaluate_policy(
    p,
    y_dod,
    y_death,
    policy_masks,
    *,
    K=None,
    alpha=None,
) -> dict:
    """在**两套标签**下同时评估一个复核/放行策略。

    返回键
    ------
    yield_dod / yield_death : 复核命中率 `P(y=1|review)`（分别用粗化 / 精确标签）
    missed_imminent         : 放行区漏检质量，**只用精确标签** `y_death`
    release_risk_dod / release_risk_death : `E[1{y=1}·1{release}]`（两套标签各一份）
    budget                  : `mean(review | alert)`（复核 + 升级占全体比例）
    compliance              : `K is None or budget <= K`，且 `alpha is None or
                              max(release_risk_dod, release_risk_death) <= alpha`
    """
    scores = _as_scores(p)
    n = scores.size
    dod = _as_labels(y_dod, n, "y_dod")
    death = _as_labels(y_death, n, "y_death")
    masks = _policy_masks(policy_masks, n)
    release, review = masks["release"], masks["review"]
    review_or_alert = review | masks["alert"]
    budget = float(np.mean(review_or_alert))

    risk_dod = float(np.mean((dod == 1.0) & release))
    risk_death = float(np.mean((death == 1.0) & release))

    compliance = True
    if K is not None:
        compliance = compliance and budget_compliance(review_or_alert, float(K))
    if alpha is not None:
        if not float(alpha) > 0.0:
            raise ValueError("alpha 必须 > 0")
        compliance = compliance and bool(max(risk_dod, risk_death) <= float(alpha) + 1e-12)

    return {
        "n": int(n),
        "yield_dod": review_yield(dod, review),
        "yield_death": review_yield(death, review),
        "missed_imminent": missed_imminent(death, release),
        "release_risk_dod": risk_dod,
        "release_risk_death": risk_death,
        "budget": budget,
        "compliance": bool(compliance),
        "n_release": int(np.count_nonzero(release)),
        "n_review": int(np.count_nonzero(review)),
        "n_alert": int(np.count_nonzero(masks["alert"])),
    }
