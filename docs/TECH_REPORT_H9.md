# ICU 预警决策 Agent · H9 复合决策算法技术文档

> 版本：v1.1（2026-10-31）｜状态：研究原型（synthetic-only）｜代码：`domain/models/selective_review.py`、`domain/models/pac_safety_net.py`
> PR：[icu-decision-agent#22](https://github.com/TinjiYDon/icu-decision-agent/pull/22)
> 关联：H4 群体阈值、H5 多时刻融合、H3 解释/HITL；以及论文线 `icu-alert-calibration` 的标签/预算审计。

---

## 0. 文档范围与结论

本文说明 decision 项目的**主要创新点、方法论、公式、函数定义与架构**。核心结论：

1. 项目的可辩护创新不是“又一个模型”，而是**复合决策算法**：把校准风险、预算工作点、序贯复核（SBR）与安全保证组合成一条决策链。
2. 现有 H4/H5/H6 不能作为主创新：H4 净受益仍为负；H5 融合后 PR-AUC 反降；H6 的“最优”来自几乎不报警（4/18,891）。它们只能当基线/负结果。
3. H9 是把“复核谁”从一次性转交升级为**有限时域、预算约束的最优停止**；当前实现为合成/纯函数原型，不含 MIMIC 实验、不构成临床证据。

---

## 1. 系统现状与架构

### 1.1 五层架构

| 层 | 目录 | 职责 |
|---|---|---|
| L1 基础设施 | `infra/` | 配置、DB engine、日志、路径 |
| L2 数据访问 | `data_access/` | `MimicRepository` / `FeatureRepository` / `LabelRepository` / `ModelRepository` |
| L3 领域 | `domain/` | ETL、特征、标签、模型、解释、建议、规划；H9 位于 `domain/models/` |
| L4 应用 | `application/` | `train`、`predict_patient`、`predict_patient_with_explanation`、`care_plan`、`summarize_hitl` |
| L5 展示 | `presentation/` | Streamlit 8 页 + MCP stdio |

### 1.2 数据流（现有）

```text
MIMIC Layer0 (mimic 库)
  │  charttime < intime + h 的因果窗
  ▼
domain/etl → feat.sample_matrix（19 特征） + label.mortality_12h（12h 死亡）
  ▼
LightGBM p(s,h) → TreeSHAP
  ├── 在线：Streamlit/MCP → 原始概率 + 0.2/0.4/0.7 固定分档 + 解释 + HITL
  └── 离线：DCA / H4 阈值 / H5 门控 / H6 RWSG+SUFH → artifacts/*.json
```

**关键事实**：在线推理只跑 LGBM + SHAP；H4/H5/H6 全部离线，阈值 τ* 未接入推理。H9 是研究原型，同样**不改变在线路径**。

### 1.3 在线 / 离线 / 研究三层

```text
┌────────────────────────────── 在线（现状） ──────────────────────────────┐
│  p → 固定分档 0.2/0.4/0.7 → 展示 → 人审（HITL，只记录 accept/reject/edit） │
└──────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── 离线（论文/评估） ─────────────────────────┐
│  p → DCA/NB → H4 约束 max-NB 阈值 → 报告；H5 融合；H6 指标/SUFH           │
└──────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────── 研究（H9，本 PR） ─────────────────────────┐
│  p 轨迹 + 复核预算 K + 模拟复核者 → SBR DP → 策略表 → 合成 pilot 指标     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 问题定义

### 2.1 超早期预警的决策问题

对 stay \(s\)、决策时刻 \(t=\mathrm{intime}+h\)（\(h\in\{0,1,2,4,6\}\)）：

- 风险模型给出 \(p(s,h)=\Pr(y=1\mid x_{<t})\)，\(y\) 为 \([t,t+12h)\) 内死亡；
- 行动：放行（release）、升级（alert）、复核（review）、等待（wait）；
- 复核人力有限：当班最多 \(K\) 次；
- 代价不对称：漏报 \(C_{FN}\) 通常远大于误报 \(C_{FP}\)。

### 2.2 为什么单阈值不够

固定阈值 \(\tau\) 只回答“报不报”，不回答：

1. 这批病例里**先让谁复核**（病例级收益不同）；
2. **什么时候**复核（现在 vs 等下一小时新化验）；
3. 复核预算花完后怎么办；
4. 放行区的漏报风险能否被**保证**。

### 2.3 复合决策的形式化

复合决策链：

```text
p(s,h) ──校准──► p_cal ──预算工作点──► τ* ──SBR──► a∈{release,alert,review,wait}
                                                │
                                                └─► PAC 安全网（已实现）：P(y=1, release) ≤ α
```

目标：

```text
π* = argmin_π  E[ L(a,y) + c_r·1{a=review} + c_e(d)·1{a=review(d)} ]
      s.t.      E[#review] ≤ K,     Risk_release ≤ α
```

其中 \(L\) 为放行漏报/升级误报的分段代价。**复合的不是模型，而是决策层**。

---

## 3. 主要创新点

### I1 复合决策算法（主）

把“校准 → 预算 → 复核 → 安全保证”做成一条可验证的链，而不是单点方法。单看每一层都是成熟组件（isotonic、DCA、LFR、conformal）；创新在**层间耦合**：

- 工作点由预算 \(α\) 与代价比 \(c\) 共同决定，而不是 F1/Youden；
- 复核决策由**病例级期望收益 + 剩余预算 + 时序**共同决定；
- 放行区风险有显式约束（PAC），不是事后解释。

### I2 预算显式工作点（支撑）

在验证集上解：

```text
τ* = argmax_τ NB(τ; c)   s.t.  AlertRate(τ) ≤ α,  τ∈[τ_min,τ_max]
NB(τ;c) = TP/n − (FP/n)·c·τ/(1−τ)
```

论文线的证据（旧 S2/α=1% 五种子）：dod 标签下 max-F1 在每个 seed 都超预算（1.86–3.23%），约束规则回到 0.90–1.02%；α=5% 时预算不 binding，方法无差异——解释了 H1 的负结果。

### I3 选择性复核 SBR（本 PR）

有限时域 DP：状态 \((p,k,t)\)，动作 {release, alert, review, wait}；复核者用 sensitivity/specificity 显式建模；输出策略表与“复核次数–漏报”前沿。

### I4 PAC 双阈值安全网（已实现，PR #22 追加）

规则：`p<τ_low` 放行、`τ_low≤p<τ_high` 复核、`p≥τ_high` 升级。风险定义为**放行区漏检质量**：

```text
R(τ_low) = E[ 1{y=1} · 1{p<τ_low} ]  ≤  α   （以 1−δ 置信）
```

实现：`domain/models/pac_safety_net.py`，用 **LTT + Hoeffding + Bonferroni** 在候选阈值对中选可行且放行人数最多的那个；返回 `DoubleThresholdResult`（含 `feasible/risk_upper_bound/三区计数/conditional_miss_rate`）。

合成证据（n=4000）：α=5% 时 feasible，τ=0.0816/0.1020，放行 74.1%，U=0.0493；α=1% 时**不可行**（Hoeffding+Bonferroni 需 n≈47k），固定阈值 0.2 的经验漏检质量 2.13% 已超 1% 预算——这是该安全网要堵的口子。

### 不主张什么

- 不主张 H4/H5/H6 是创新（证据见 `docs/ARCHITECTURE_MATH.md` 与 artifacts；H6 的 SUFH 报警 4/18,891 属退化）。
- 不主张模型集成（LGBM+GRU+TFT+LLM 堆叠）是创新；SUFH 已证明 4 维 logistic stacking 会退化，且 chartevents 全量缺失。
- 不主张临床效果；当前只有合成/离线证据。

---

## 4. 方法论

### 4.1 决策论框架

以**期望代价**为统一目标，而不是 AUC/F1：

- 漏报代价 \(C_{FN}\)、误报代价 \(C_{FP}\)、复核成本 \(c_r\)；
- 所有策略在同一代价模型下比较；
- 预算 \(K\) 与报警率 \(α\) 是硬约束，不是事后指标。

### 4.2 三层验证（从便宜到贵）

| 阶段 | 数据 | 目的 | 状态 |
|---|---|---|---|
| V1 合成 | `make_synthetic_cases` | 验证 DP/策略/指标正确性 | ✅ 本 PR |
| V2 公开 | PhysioNet/CinC 2012（4,000 records） | 无需 DUA 的可复现 pilot | 待跑 |
| V3 真实 | MIMIC-IV（94,458 stays） | 论文级证据 | 等 F1 数据授权 |

### 4.3 基线与消融

- 基线：release_all、alert_all、fixed_0.5、fixed_0.384（H4 旧工作点）、double_threshold、myopic_review；
- 消融：预算 K∈{0,1,3,5}、复核者 sens/spec∈{0.6,0.8,0.95}、代价比 c∈{2,4,6}、horizon∈{3,5}；
- 对照：同代价模型下的期望代价、漏报率、复核率/命中率、oracle gap。

### 4.4 统计协议

- 病例为独立单位（stay），bootstrap 重采样 stay，不按行；
- 多 seed（≥5）报均值±标准差；
- 合成数据固定 seed，保证 CI 可复现；
- 真实数据阶段用 paired bootstrap 比较策略。

### 4.5 与外部工作的边界

- CoDoC（Nat Med 2023）：单次转交、无预算/时序；SBR 加时序与预算。
- DeCCaF（TMLR 2024）/ OpenL2D（Sci Data 2025）：批次容量/多专家基准；SBR 用其模拟专家思路，但决策是序贯停止。
- Reid 2024 / DeSalvo 2025：预算约束在线转交；SBR 加临床终局代价与解释深度动作（后续）。
- H4：群体阈值；SBR 是病例级、时序扩展。

---

## 5. 公式

### 5.1 风险与校准

\[
p(s,h)=\Pr(y=1\mid x_{<t}),\qquad p_{cal}=\mathrm{Isotonic}(p)
\]

（当前在线未接校准；论文线用 split-conformal/保形预算。）

### 5.2 净受益与预算约束

\[
NB(\tau;c)=\frac{TP}{n}-\frac{FP}{n}\cdot c\cdot\frac{\tau}{1-\tau}
\]

\[
\tau^*=\arg\max_\tau NB(\tau;c)\quad
\text{s.t.}\quad \frac1n\sum_i 1\{p_i\ge\tau\}\le\alpha
\]

### 5.3 复核者模型

灵敏度 \(s=P(m=1\mid y=1)\)，特异度 \(t=P(m=0\mid y=0)\)。后验：

\[
P(y=1\mid m=1)=\frac{p s}{p s+(1-p)(1-t)},\qquad
P(y=1\mid m=0)=\frac{p(1-s)}{p(1-s)+(1-p)t}
\]

复核的正例率：\(q(p)=ps+(1-p)(1-t)\)。

### 5.4 复核的即时期望代价

\[
V_{rev}(p)=c_r+q(p)\,B(p_{+})+(1-q(p))\,B(p_{-}),\qquad
B(p)=\min(pC_{FN},(1-p)C_{FP})
\]

### 5.5 SBR Bellman

终端时刻 \(T\)：

\[
V_T(p,k)=\min\{pC_{FN},\ (1-p)C_{FP},\ \mathbb{1}[k>0]\,V_{rev}(p)\}
\]

中间时刻：

\[
V_t(p,k)=\min\{pC_{FN},\ (1-p)C_{FP},\ \mathbb{1}[k>0]\,V_{rev}(p),\ \mathbb{E}_{p'\mid p}V_{t+1}(p',k)\}
\]

动作取 argmin；`solve_sbr` 在离散网格上反向计算。

### 5.6 PAC 双阈值（Hoeffding + Bonferroni + LTT）

候选阈值对 \((\tau_{low},\tau_{high})\)，候选数 \(m\)；对每个候选：

\[
\hat R=\frac1n\sum_i 1\{y_i=1,\,p_i<\tau_{low}\},\qquad
U=\hat R+\sqrt{\frac{\log(1/\delta_m)}{2n}},\qquad \delta_m=\delta/m
\]

选择：

\[
(\tau_{low}^*,\tau_{high}^*)=\arg\max_{(\tau_{low},\tau_{high}):\,U\le\alpha}
\#\{i:p_i<\tau_{low}\}
\]

无可行候选时 `feasible=False`，回落到 `(0.0, 1.0)`、`U=1.0`。`τ_high` 不进入本风险定义（升级区漏检另行约束）。

### 5.7 评估指标

\[
\text{mean cost}=\frac1n\sum_i \text{cost}_i,\qquad
\text{miss rate}=\frac1n\sum_i 1\{a_i=release,y_i=1\}
\]

\[
\text{review yield}=\frac{\#\{\text{复核后改变行动}\}}{\#\{\text{复核}\}},\qquad
\text{oracle gap}=\text{mean cost}-\text{perfect-oracle cost}
\]

（完美信息下界为 0；gap 等于 mean cost，仅作参考。）

---

## 6. 函数定义

模块：`domain/models/selective_review.py`。所有函数为纯函数/轻量类，不读 DB、不调网络。

### 6.1 常量与数据结构

| 名称 | 定义 |
|---|---|
| `RELEASE=0, ALERT=1, REVIEW=2, WAIT=3` | 动作编码 |
| `CostModel(cost_fn=4.0, cost_fp=1.0, review_cost=0.2)` | 代价模型 |
| `Reviewer(sensitivity=0.80, specificity=0.80)` | 模拟复核者 |
| `SBRPolicy(grid, action, value, cost, reviewer, budget)` | DP 策略表 |

### 6.2 函数表

| 函数 / 方法 | 签名 | 返回 | 说明 |
|---|---|---|---|
| `CostModel.release_cost` | `(p) -> np.ndarray` | \(pC_{FN}\) | 支持标量/数组 |
| `CostModel.alert_cost` | `(p) -> np.ndarray` | \((1-p)C_{FP}\) | 同上 |
| `CostModel.best_action_cost` | `(p) -> np.ndarray` | \(\min\) | 即时最优 |
| `Reviewer.posterior` | `(p: float, m: int) -> float` | 后验 | 贝叶斯 |
| `Reviewer.positive_rate` | `(p) -> np.ndarray` | \(q(p)\) | 复核判阳概率 |
| `Reviewer.sample` | `(p, y, rng) -> int` | 0/1 | 按 sens/spec 采样 |
| `Reviewer.expected_post_cost` | `(p, cost) -> np.ndarray` | \(qB(p_+)+(1-q)B(p_-)\) | 不含 \(c_r\) |
| `default_transition` | `(grid, persistence=0.85) -> np.ndarray` | \(n\times n\) | 粘滞随机游走 |
| `estimate_transition` | `(trajectories, grid) -> np.ndarray` | \(n\times n\) | 病例轨迹 + 拉普拉斯平滑 |
| `SBRPolicy.act` | `(t, p, k) -> int` | 动作 | 最近邻查表 |
| `solve_sbr` | `(grid, transition, cost, reviewer, horizon, budget) -> SBRPolicy` | 策略 | 反向 DP；`horizon=时刻数-1` |
| `FixedThresholdPolicy` | `(threshold)` | policy | `p≥τ` 升级 |
| `DoubleThresholdPolicy` | `(low, high)` | policy | 灰区复核 |
| `MyopicReviewPolicy` | `(cost, reviewer, threshold=0.5)` | policy | Δ 贪心 |
| `simulate_case` | `(trajectory, y, policy, cost, reviewer, budget, rng) -> dict` | cost/reviews/changed/action/missed | 单病例仿真 |
| `evaluate_policy` | `(policy, trajectories, labels, cost=None, reviewer=None, budget=3, seed=0) -> dict` | 指标 | 汇总评估 |
| `make_synthetic_cases` | `(n=2000, horizon=5, seed=0, persistence=0.9) -> (trajs, labels)` | 合成数据 | Beta 起始 + 粘滞游走 |

### 6.3 关键代码

```python
# 反向 DP 的核心（简化）
for t in range(T - 2, -1, -1):
    wait = transition @ V[t + 1]          # (n_grid, K+1)
    for k in range(K + 1):
        for i in range(n_grid):
            cand = [(rel[i], RELEASE), (alt[i], ALERT), (wait[i, k], WAIT)]
            if k > 0:
                cand.append((rev[i], REVIEW))
            val, act = min(cand, key=lambda x: x[0])
            V[t, i, k], A[t, i, k] = val, act
```

```python
# 单病例仿真（简化）
for t, p in enumerate(trajectory):
    a = policy.act(t, p, k)
    if a == WAIT: continue
    if a == REVIEW and k > 0:
        k -= 1; total += cost.review_cost
        m = reviewer.sample(p, y, rng)
        p_post = reviewer.posterior(p, m)
        a2 = _terminal_action(p_post, cost)
        total += cost.release_cost(p_post) if a2 == RELEASE else cost.alert_cost(p_post)
        return {...}
    total += cost.release_cost(p) if a == RELEASE else cost.alert_cost(p)
    return {...}
```

---

### 6.4 PAC 模块函数表（`domain/models/pac_safety_net.py`）

| 函数 / 类 | 签名 | 返回 | 说明 |
|---|---|---|---|
| `release_mask` | `(p, tau_low) -> np.ndarray` | 布尔掩码 | `p < tau_low` |
| `review_mask` | `(p, tau_low, tau_high) -> np.ndarray` | 布尔掩码 | `tau_low <= p < tau_high` |
| `alert_mask` | `(p, tau_high) -> np.ndarray` | 布尔掩码 | `p >= tau_high` |
| `missed_mass` | `(p, y, tau_low) -> float` | \(\hat R\) | 放行区漏检质量 |
| `hoeffding_radius` | `(n, delta_m) -> float` | 半宽 | \(\sqrt{\log(1/\delta_m)/(2n)}\) |
| `select_double_threshold` | `(p, y, alpha=0.01, delta=0.1, n_low=50, n_high=50, min_release_rate=0.0) -> DoubleThresholdResult` | 结果 dataclass | LTT 选择 |
| `DoubleThresholdResult` | dataclass | 字段见下 | `tau_low/tau_high/feasible/empirical_risk/risk_upper_bound/alpha/delta/n_cal/n_release/n_review/n_alert/conditional_miss_rate` |

约定：
- `p` 截断到 `[0,1]`，`y∈{0,1}`；长度不一致、`alpha<=0`、`delta∉(0,1)`、`n_low/n_high<2` 抛 `ValueError`。
- `feasible=False` 时 `tau_low=0.0, tau_high=1.0, risk_upper_bound=1.0`，表示“无法给出保证”，不是“安全”。
- `τ_high` 只影响三区划分，不进入本风险定义；升级区漏检/复核人力约束另行处理。

## 7. 架构展示

### 7.1 系统总架构（现状 + H9 位置）

```text
┌─────────────────────────────────────────────────────────────────────┐
│ L5  Streamlit / MCP                                                 │
│     └─ 风险分 + 固定分档 + SHAP + RAG/LLM + HITL（仅记录）            │
├─────────────────────────────────────────────────────────────────────┤
│ L4  application: train / predict_patient / care_plan / summarize_hitl│
├─────────────────────────────────────────────────────────────────────┤
│ L3  domain                                                          │
│   etl → features → labels → models ┬─ lgbm（骨干）                  │
│                                    ├─ utility_calibrate（H4）       │
│                                    ├─ sparsity_fusion（H5）         │
│                                    ├─ decision_indices/sufh（H6）    │
│                                    ├─ dca（DCA）                     │
│                                    └─ selective_review（H9，研究）   │
│   explain: shap_llm / rag / reference_validator / hitl_audit（H3）   │
├─────────────────────────────────────────────────────────────────────┤
│ L2  data_access: MimicRepository / Feature / Label / Model           │
├─────────────────────────────────────────────────────────────────────┤
│ L1  infra: db / config / logging / paths                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 H9 在决策链中的位置

```text
 p(s,h) ──► [校准] ──► p_cal ──► [预算工作点 τ*] ──► 报警决策
                                   │
                                   └─► [H9 SBR] ──► release / alert / review / wait
                                            │
                                            └─► [PAC 安全网] ──► 放行区漏检 ≤ α
```

### 7.3 H9 类关系

```text
CostModel ◄── SBRPolicy
Reviewer  ◄── SBRPolicy
                ▲
solve_sbr ──────┘
   │
   ├─ FixedThresholdPolicy
   ├─ DoubleThresholdPolicy
   └─ MyopicReviewPolicy
        │
        └─ evaluate_policy ──► metrics dict
```

---

## 8. 算法伪代码

### 8.1 solve_sbr

```text
Input: grid[m], transition[m,m], cost, reviewer, horizon T, budget K
Output: V[T+1,m,K+1], A[T+1,m,K+1]
1. rel[i]=grid[i]*C_FN; alt[i]=(1-grid[i])*C_FP
2. rev[i]=c_r + reviewer.expected_post_cost(grid[i])
3. for k=0..K: at t=T choose argmin(rel, alt, rev if k>0)
4. for t=T-1..0:
     wait = transition @ V[t+1]
     for k=0..K:
       for i=0..m-1:
         cand = rel, alt, wait[i,k], (rev if k>0)
         V[t,i,k], A[t,i,k] = argmin
5. return policy(grid, A, V, cost, reviewer, K)
```

### 8.2 evaluate_policy

```text
Input: policy, trajectories, labels, cost, reviewer, budget, seed
1. rng = default_rng(seed)
2. for each (traj, y): simulate_case(...)
3. aggregate mean cost / miss / alert / review / yield
4. oracle = mean(min(C_FN*y, C_FP*(1-y)))
5. return metrics
```

---

## 9. 实验与证据

### 9.1 合成 pilot（sens/spec=0.8，K=3，n=2000，horizon=5）

| policy | mean_cost | miss_rate | alert_rate | review_rate | review_yield |
|---|---:|---:|---:|---:|---:|
| release_all | 0.2511 | 0.102 | 0.000 | 0.000 | 0.000 |
| alert_all | 0.9372 | 0.000 | 1.000 | 0.000 | 0.000 |
| fixed_0.5 | 0.2442 | 0.101 | 0.004 | 0.000 | 0.000 |
| fixed_0.384 | 0.2338 | 0.098 | 0.013 | 0.000 | 0.000 |
| double_threshold | 0.1999 | 0.086 | 0.029 | 0.082 | 0.671 |
| myopic_review | 0.1993 | 0.078 | 0.054 | 0.157 | 0.462 |
| **sbr_dp** | **0.1630** | **0.063** | 0.087 | 0.166 | 0.482 |

**解读**：在合成设定下，SBR 优于“按 Δ 贪心”和固定阈值；但这是模拟证据，不能外推临床。

### 9.2 PAC 安全网（H9-I4，synthetic）

| 配置（n=4000, seed=0） | τ_low / τ_high | feasible | R̂ | U | release/review/alert |
|---|---|---:|---:|---:|---|
| α=5%, 50×50 网格 | 0.0816 / 0.1020 | True | 0.0150 | 0.0493 | 2963 / 182 / 855 |
| α=1%, 50×50 网格 | 0.0 / 1.0 | **False** | 0 | 1.0 | 0 / 4000 / 0 |
| 基线：固定 τ_low=0.2 | — | — | 0.0213 | U(m=1)=0.0382 | 3681 放行 |

**解读**：固定阈值 0.2 的经验漏检质量 2.13% 本身就超过 1% 预算；α=1% 在 n=4000 下 Hoeffding+Bonferroni 无法给出保证，安全网按设计关闭放行区。这是诚实的负结果，不是实现失败。

### 9.3 测试与 CI

- `pytest tests/test_selective_review.py -q` → **10 passed**；
- `pytest tests/test_pac_safety_net.py -q` → **7 passed**；两者合计 **17 passed**；
- PR #22 无仓库 pytest CI（decision 仓未配置），Sourcery 为外部 bot。

### 9.4 复现命令

```powershell
cd "C:\Users\lenovo\Desktop\decision shcedule\icu-decision-agent"
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_selective_review.py -q
.\.venv\Scripts\python.exe scripts/h9_sbr_pilot.py --n 2000 --horizon 5 --budget 3 --seed 0
.\.venv\Scripts\python.exe -m pytest tests/test_pac_safety_net.py -q
.\.venv\Scripts\python.exe scripts/h9_pac_demo.py --n 4000 --alpha 0.05 --delta 0.1 --seed 0
```

---

## 10. 工程与复现

| 项 | 内容 |
|---|---|
| 依赖 | Python ≥3.11；numpy（已装）；无 DB/LLM |
| 数据 | 合成；真实数据等待 F1 |
| 入口 | `scripts/h9_sbr_pilot.py` |
| 产物 | `artifacts/h9/sbr_pilot_<seed>.json`（gitignored） |
| 测试 | `tests/test_selective_review.py` |
| 文档 | `docs/H9_SBR.md`、`docs/PAC_SAFETY_NET.md`、本文 |
| PR | [icu-decision-agent#22](https://github.com/TinjiYDon/icu-decision-agent/pull/22) |

---

## 11. 局限与下一步

1. **模拟复核者**：sens/spec 是假设；下一步用 HITL 审计估计。
2. **复核后立即行动**：v1 不建模“复核后再等待”；可把 review 接入下一时刻状态。
3. **合成轨迹**：真实 p 轨迹需 F1；可先用已落盘的 5 时刻预测估计转移矩阵（只读，不新跑实验）。
4. **PAC 安全网的代价**：α=1%、n=4000 时不可行（Hoeffding+Bonferroni 需 n≈47k）；实际部署需要更大校准集或更紧的界（如 Clopper-Pearson/经验 Bernstein）。`τ_high` 不进入风险定义，升级区漏检未约束。
5. **不改变在线路径**：H9 是研究模块，上线需另行评审。

---

## 12. 附录

### 12.1 符号表

| 符号 | 含义 |
|---|---|
| \(p\) | 风险概率 |
| \(k\) | 剩余复核次数 |
| \(t\) | 决策时刻 |
| \(C_{FN},C_{FP}\) | 漏报/误报代价 |
| \(c_r\) | 单次复核成本 |
| \(s,t\) | 复核者灵敏度/特异度（\(t\) 也用于时刻，文中按上下文区分） |
| \(K\) | 复核预算 |
| \(α\) | 报警率/漏报率预算 |

### 12.2 默认参数

| 参数 | 默认 | 说明 |
|---|---:|---|
| `cost_fn` | 4.0 | 漏报代价 |
| `cost_fp` | 1.0 | 误报代价 |
| `review_cost` | 0.2 | 单次复核 |
| `sensitivity` | 0.80 | 复核者 |
| `specificity` | 0.80 | 复核者 |
| `persistence` | 0.85/0.9 | 转移粘性 |
| `grid` | 41/21 | 概率网格 |
| `budget` | 3 | 复核次数 |
| `alpha` | 0.01/0.05 | PAC 风险预算 |
| `delta` | 0.1 | PAC 置信参数 |
| `n_low/n_high` | 50/50 | PAC 阈值网格 |

### 12.3 已知不主张

- 不主张 H4/H5/H6 为创新；
- 不主张模型集成；
- 不主张临床效果；
- 不把合成结果写成 MIMIC 结论。
