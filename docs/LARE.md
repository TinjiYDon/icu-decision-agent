# LARE · LAR-Policy（标签时间歧义下的复核评估与鲁棒选阈值）

> 状态：**研究原型 + MIMIC 运行器**。默认 demo 为合成数据；真实模式需 `LARE_MIMIC_ALLOW=1`（F1 已确认），读 `paper_feat/paper_label`，不调 LLM。
> 代码：`domain/models/lare.py` · `domain/models/lar_policy.py`
> 脚本：`scripts/lare_demo.py`（合成 demo） · `scripts/lare_mimic_run.py`（MIMIC 运行器 / `--smoke` 自检）
> 测试：`tests/test_lare.py` · `tests/test_lar_policy.py` · `tests/test_lare_runner.py`
> 关联：[LABEL_AUDIT.md](LABEL_AUDIT.md)（dod 粗化）、[PAC_SAFETY_NET.md](PAC_SAFETY_NET.md)（放行区漏检质量）、
> [H9_SBR.md](H9_SBR.md)（复核策略）、[ARCHITECTURE_MATH.md](ARCHITECTURE_MATH.md)（H4 阈值）。

## 1. 问题：同一批病人，两套死亡标签

| 标签 | 来源 | 12h 死亡窗 | 基准率（本地队列） |
|---|---|---|---|
| `y_death` | `admissions.deathtime`（时间戳） | 严格 | **1.19%** |
| `y_dod` | `patients.dod`（日期级） | 把整天视为死亡区间 → 粗化 | **2.34%** |

P1 标签审计（`docs/LABEL_AUDIT.md`）：461,251 行两者皆阴、5,638 行两者皆阳、5,401 行**仅 dod 阳性**、
0 行仅 deathtime 阳性。即

```
y_dod 是 y_death 的粗化超集：  y_death = 1  ⇒  y_dod = 1
dod-only 阳性 5,401 / dod 阳性 11,039 = 48.9%
```

这带来两个必须显式处理的**评估陷阱**：

1. **口径切换 = 看起来更好**：换到粗化标签后阳性变多，多出来的那部分分数偏低（日期级模糊，
   不一定是真正的 12h 近因死亡），更容易落进复核带 → **复核 yield 变高**、放行区漏检质量也变高；
   若不标注口径，读者会把“标签更松”误读成“策略更有效”。
2. **目标漂移 = 复核的不是同一批人**：按 A 口径选出的阈值，在 B 口径下未必最优；
   两套口径选出的双阈值不同，复核集合的重叠度可能很低（本 demo Jaccard ≈ 0.24）。

**LARE 不发明新标签、不改标签定义**，只提供“同一策略在两套口径下分别怎么读”的评估量；
**LAR-Policy 不追求单口径最优**，而是在两套口径下都满足约束、并让较差的那个口径的代价最小。

## 2. LARE 指标

记 `p ∈ [0,1]` 为风险分（要求 out-of-fold），三区

```
release : p < tau_low
review  : tau_low <= p < tau_high
alert   : p >= tau_high
```

| 指标 | 定义 | 读法 |
|---|---|---|
| `review_yield(y, review)` | `P(y=1 | review) = E[1{y=1}·1{review}] / E[1{review}]` | 复核命中率；分子分母同口径，跨口径比较才能暴露“粗化红利” |
| `missed_imminent(y_death, release)` | `E[1{y_death=1}·1{release}]` | 放行区漏检质量（**只用精确标签**，全人群口径） |
| `reviewed_jaccard(A, B)` | `|A∩B| / |A∪B|` | 两套口径/两个策略的复核集合一致度；空集视为一致（1.0） |
| `priority_spearman(s, mask)` | `corr(rank(s), rank(1{mask}))`（平均秩，手写） | 复核集合是否集中在高分端；方差为 0 时返回 0 |
| `budget_compliance(mask, K)` | `mean(mask) <= K` | 复核 + 升级占全体比例是否在预算内 |
| `release_risk_*` | `E[1{y=1}·1{release}]`（`y` 分别为 dod / death） | 与 `pac_safety_net` 的 `missed_mass` 完全一致 |

粗化标签用 `build_coarse_label(y_death, p, add_rate, rng)` 模拟：

```
y_dod = y_death ∨ Bernoulli(add_rate · p)
```

即“当天晚些时候死亡”的概率随风险分线性增加；`add_rate=0` 时与精确标签完全相同，
且恒有 `y_death=1 ⇒ y_dod=1`。这只是**合成口径**，不是 MIMIC 的真实 dod 生成机制。

`evaluate_policy(p, y_dod, y_death, masks, K=..., alpha=...)` 一次性给出两套口径的
`yield / missed_imminent / release_risk`、`budget` 与 `compliance`
（`compliance = budget<=K` 且 `max(release_risk_dod, release_risk_death) <= alpha`）。

## 3. LAR-Policy：min-max 鲁棒双阈值

在同一候选网格上解

```
min_{tau_low < tau_high}   max( cost_dod(tau), cost_death(tau) )
s.t.  (n_review + n_alert) / n <= K                     # 复核 + 升级预算
      E[1{y_dod=1}   · 1{p < tau_low}] <= alpha          # 粗化口径的放行漏检
      E[1{y_death=1} · 1{p < tau_low}] <= alpha          # 精确口径的放行漏检
```

期望代价（与 H4/H9 `CostModel` 同口径）：

```
cost = mean( cost_fn · 1{y=1, release}
           + cost_fp · 1{y=0, alert}
           + review_cost · 1{review} )
```

- `y=1, release` 是漏报（最贵），`y=0, alert` 是误报，`review` 只计平均人力成本；
- 复核区**不产生**漏报/误报代价（假定复核可纠正），这是与 H9 SBR 一致的简化；
- 默认 `cost_fn=4, cost_fp=1, review_cost=0.2`。

实现细节（`fit_lar_policy`）：

- **网格**：`lows = highs = linspace(0,1,n_grid)`，只保留 `tau_low < tau_high` 的 `n_grid(n_grid-1)/2` 对；
- **解析分解**：因为 `tau_low < tau_high` 时 `release` 只依赖 `tau_low`、`alert` 只依赖 `tau_high`，
  `review` 是两者之差，所以 `cost[i,j] = cost_fn·R_i + cost_fp·A_j + review_cost·(q_i - g_j)`，
  50×50 网格只需两次前缀统计，没有 O(n·m) 掩码开销（`n=4000` 时整个 fit 是毫秒级）；
- **同分**（`robust_cost` 相同）：取**放行人数最多**者；再同分取更小的 `tau_high`（复核带更宽、更保守）；
- **无可行候选**：`feasible=False`，`(tau_low, tau_high) = (0.0, 1.0)`（**全部进复核**），
  `robust_cost = cost_dod = cost_death = inf`，`n_release = 0` —— 宁可放弃放行，也不给一个不自洽的阈值。

`label_specific_policies` 用同一网格/同一约束，但**只用单一标签**的 `release_risk` 与代价选阈值，
用于量化“口径切换代价”；`max_regret = max(cost_dod - opt_dod, cost_death - opt_death)`
即相对标签专属最优的最大后悔（某标签专属最优不可行时该标签按 0 计）。

### 3.1 有限样本上界（`bound="hoeffding"`）

上面的可行性判断用的是**经验**放行风险（同一批样本既选阈值又验约束），因此 `robust_cost`
偏乐观。开 `bound="hoeffding"` 时改用**单侧 Hoeffding 上界**（与
`domain.models.pac_safety_net.hoeffding_radius` 同口径、同一 union bound 写法）：

```
m = n_grid(n_grid - 1) / 2                      # tau_low < tau_high 的候选对数（50 网格 → 1225）
radius = sqrt( log(m / delta) / (2n) )          # 默认 delta = 0.1（即 90% 置信）
可行 ⇔  release_risk_dod   + radius <= alpha
    且  release_risk_death + radius <= alpha
    且  (n_review + n_alert) / n <= K           # 预算约束不变（确定性，无需上界）
目标：仍是  min max( cost_dod , cost_death )   # 代价不加界，只收紧可行域
```

- `m` 个候选共用一次 union bound，故每个候选的失败概率是 `delta/m`；
  `n_grid` 越大 `radius` 越大（候选越多、多重比较越贵），所以**扩大网格不能换来小样本可行性**；
- 选中候选的两个上界写进 `risk_ub_dod / risk_ub_death`（未启用上界时保持 `inf`，向后兼容）；
- 单调性：`hoeffding 可行 ⇒ none 可行`，反向不成立；因此加上界后 `robust_cost` **只会变差或不变**；
- 非法 `bound` / `delta ∉ (0,1)` → `ValueError`；`delta` 默认 `0.1`，调小会同时放大 `radius`（更保守）。

**与 `n≈47k` 边界的关系**：`alpha=1%` 时（§5 的严格口径，`feasible=False`），
`Beta(0.4,6)` 在 80% 放行率处 `E[y_death·1{release}] ≈ 1.7%` 已超 `α`，即使 `radius=0` 也不可行；
要让上界进 `1%` 需要 `radius` 小到把经验风险压回 `α` 以下，与 H9-I4
（`docs/PAC_SAFETY_NET.md` §3：Hoeffding + 1225 候选时 `n >= 47k`）同源。
对 `n=4000` 的 demo：`m=1225, delta=0.1` → `radius ≈ 0.0343`，已吃掉 `α=5%` 的 2/3；
鲁棒策略经验放行风险 `y_dod = 3.13%` 留给 `radius` 的余量只有 `1.87%`，
因此需要 `n ≳ 1.3×10⁴` 才可能可行（若要求 `radius <= 0.5%` 则需 `n ≈ 1.9×10⁵`）。
这就是 §5 里 `bound=hoeffding` 在 `n=4000, α=5%` 下 `feasible=False` 的原因——
**不是策略变差，而是证据量不足**。

**局限**：Hoeffding 对 `[0,1]` 有界量是**无分布**界，通常远松于经验 Bernstein / Clopper-Pearson
（方差已知时 `sqrt(2·R·log(m/δ)/n)` 量级）；这里的半径按**最坏方差 1/4** 上界，
所以“不可行”只能读作“**该样本量下无分布无关保证**”，不能读作“临床上一定不安全”。
真实 `p` 到手后应换 tighter bound 或更大校准集再判。

## 4. API

`domain/models/lare.py`

| 符号 | 说明 |
|---|---|
| `build_coarse_label(y_death, p, *, add_rate=0.5, rng=None)` | 粗化超集；`rng=None` 用 `default_rng(0)` |
| `review_yield(y, review_mask)` | `P(y=1|review)`；复核集为空 → `0.0` |
| `missed_imminent(y_death, release_mask)` | `E[1{y_death=1}·1{release}]` |
| `reviewed_jaccard(mask_a, mask_b)` | 复核集合 Jaccard；空/空 → `1.0` |
| `priority_spearman(scores, mask)` | 复核指示 vs 分数的 Spearman（平均秩，无 scipy） |
| `budget_compliance(mask, K)` | `mean(mask) <= K + 1e-12` |
| `fixed_policy(p, tau)` | `{"release","review","alert"}`；单阈值时 `review` 为空 |
| `double_policy(p, tau_low, tau_high)` | 三区互斥且并集为全集 |
| `evaluate_policy(p, y_dod, y_death, masks, *, K=None, alpha=None)` | 两口径一次评估 |

`domain/models/lar_policy.py`

| 符号 | 说明 |
|---|---|
| `LARPolicy(tau_low, tau_high, feasible, robust_cost, cost_dod, cost_death, n_release, risk_ub_dod=inf, risk_ub_death=inf)` | frozen dataclass；`risk_ub_*` 仅在 `bound="hoeffding"` 且可行时填入 |
| `expected_cost(y, p, tau_low, tau_high, *, cost_fn=4.0, cost_fp=1.0, review_cost=0.2)` | 期望代价 |
| `fit_lar_policy(p, y_dod, y_death, *, K=0.2, alpha=0.01, n_grid=50, ..., bound="none", delta=0.1)` | min-max 鲁棒选择；`bound` ∈ {`none`,`hoeffding`} |
| `label_specific_policies(p, y_dod, y_death, **kw)` | `{"dod": {...}, "death": {...}}` 单口径最优 |
| `max_regret(p, y_dod, y_death, policy, **kw)` | 相对标签专属最优的最大后悔 |

约定：长度不一致 / 标签非 0/1 / `K∉[0,1]` / `alpha<=0` / `n_grid<2` / 负代价 /
`bound∉{none,hoeffding}` / `delta∉(0,1)` → `ValueError`；
`policy_masks` 不是 dict → `TypeError`；三区重叠 → `ValueError`。
纯函数、只依赖 `numpy`（复用 `pac_safety_net` / `lare` 的掩码口径），无 DB、无网络、无全局状态。

## 5. 合成 demo（2026-11-02, seed=0, n=4000, add_rate=0.5, K=0.2, alpha=5%）

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe scripts/lare_demo.py --n 4000 --seed 0
.\.venv\Scripts\python.exe scripts/lare_demo.py --n 4000 --seed 0 --bound hoeffding
```

数据：`p ~ Beta(0.4, 6.0)`，`y_death ~ Bernoulli(0.15p + 0.02)`，`y_dod = build_coarse_label(..., 0.5)`。
基准率：`y_death=0.0260`、`y_dod=0.0565`（+0.0305）。输出终端摘要 + `artifacts/lare/demo.json`。

**(a) 同一固定策略 `tau_low=0.1, tau_high=0.3` 在两套标签下**

| 量 | `y_dod` | `y_death` | 差 |
|---|---:|---:|---:|
| review yield | 0.1158 | 0.0320 | **+0.0839** |
| release risk | 0.0270 | 0.0170 | **+0.0100** |
| budget | 0.2170 | 0.2170 | 0 |
| release/review/alert | 3132 / 751 / 117 | 同 | – |

`compliance = False`：预算 21.7% 已超 `K=20%`。**同一策略的复核命中率在粗化口径下是精确口径的
3.6 倍**——这就是“看起来更好”的来源；放行区漏检质量同时被抬高一倍（1.7% → 2.7%）。

**(b) 按各自标签单独选阈值（同网格、同约束）**

| 标签 | `tau_low` | `tau_high` | cost | n_release |
|---|---:|---:|---:|---:|
| dod | 0.1224 | 0.7551 | 0.15995 | 3301 |
| death | 0.2653 | 0.7551 | 0.09655 | 3829 |

`Δtau_low = -0.1429`（粗化口径明显更激进），`Δtau_high = 0`，
复核集合 **Jaccard = 0.2446** —— 两个“同样合法”的策略复核的只有约四分之一是同一批病人。

**(c) LAR-Policy（min-max 鲁棒，K=20%, alpha=5%）**

| 量 | 值 |
|---|---:|
| `tau_low / tau_high` | 0.1224 / 0.7551 |
| `robust_cost = max(cost_dod, cost_death)` | 0.15995 |
| `cost_dod / cost_death` | 0.15995 / 0.10995 |
| `n_release` | 3301（82.5%） |
| `max_regret` | **0.0134** |

鲁棒策略被**粗化标签**卡住（`cost_dod` 是 max），相对标签专属最优只多付 0.0134 的期望代价，
换来“两套口径下都满足预算与 5% 放行漏检”。

**严格口径 `alpha=1%`：`feasible=False`**（`robust_cost=inf`）。
原因：`K=20%` 要求放行率 ≥80%，而 `Beta(0.4,6)` 的 80% 分位附近
`E[y_death·1{release}] ≈ 1.7% > 1%`；即使粗化附加率为 0，该预算也过不了。
这与 H9-I4 的结论同源——**1% 的放行漏检预算需要更大的校准集 / 更紧的统计界**
（`docs/PAC_SAFETY_NET.md` §3：Hoeffding + 1225 候选时 `n >= 47k`）。

**(d) 有限样本上界对照（`--bound none` vs `--bound hoeffding`，`delta=0.1`）**

| bound | `tau_low` | `tau_high` | feasible | `robust_cost` | `n_release` | `max_regret` |
|---|---:|---:|---|---:|---:|---:|
| none（经验风险） | 0.1224 | 0.7551 | True | 0.15995 | 3301 | 0.0134 |
| hoeffding（上界） | 0.0000 | 1.0000 | **False** | inf | 0 | 0.1035 |

`n=4000, m=1225, delta=0.1` → `radius ≈ 0.0343`；鲁棒策略的经验放行风险
`y_dod=3.13%`，加界后 `6.56% > α=5%`，于是回落到“全部进复核”
（`max_regret` 退化为相对 `none` 策略的 0.1035）。这正说明
**经验可行 ≠ 在 90% 置信下可行**，见 §3.1。

## 6. 局限

1. **合成数据**：`y_dod` 是“随 p 线性附加”的模拟，真实 dod 的粗化机制依赖收治/转出/日期边界，
   附加率未必与分数成正比；本 demo 只能说明**评估口径**的行为，不能定量外推到 MIMIC。
2. **等 F1 数据授权**：真实 `p`（out-of-fold，H5 `p_fuse` / H6 `p_rw`）到手后，需在校准 split 上
   重跑同协议，test split **只做一次性验证**（不做二次调参）。
3. **不构成临床证据**：没有任何真实病人、没有 mortality 端点验证；`cost_fn/cost_fp/review_cost`
   是相对单位，不是金钱或 QALY。
4. **min-max 只覆盖两个口径**：真实标签歧义不止两套（dod/deathtime/院内死亡/30 天），
   多于两套时需把 `max` 推广到集合上，并重新审视“最差口径”是否被单一粗化标签主导。
5. **网格 + 经验风险**：`fit_lar_policy` 直接在评估样本上选阈值（无 PAC 上界），
   因此 `robust_cost` 是**乐观**的；要有限样本保证，需要把 `pac_safety_net` 的 Hoeffding/Bonferroni
   搬到两套标签的联合约束上（未实现）。
6. **复核代价模型是简化的**：假定复核能纠正一切、且成本与病例无关；真实 HITL 有复核者灵敏度/特异度
   （见 `selective_review.Reviewer`），会进一步改变最优阈值。

## 7. MIMIC 运行器（`scripts/lare_mimic_run.py`）

合成 demo 只验证**口径**；要在真实 MIMIC 上跑「两套标签 × 四种策略 × N seeds」的网格，用运行器。
核心管线 `run_lare_pipeline(p, y_dod, y_death, seeds)` 是**纯函数**（可单测、不读 DB），
`--smoke` 与真实模式走**同一条**管线，只在「数据从哪来」上分叉。

### 7.1 命令

```powershell
$env:PYTHONPATH = (Get-Location)
# 1) 自检：合成数据（n=2000），不连 DB、不需要授权
.\.venv\Scripts\python.exe scripts/lare_mimic_run.py --smoke --seeds 2
# 2) 单测（不连 DB、不跑真实数据）
.\.venv\Scripts\python.exe -m pytest tests/test_lare_runner.py -q
# 3) 真实模式：必须先过 F1 门
$env:LARE_MIMIC_ALLOW = "1"
.\.venv\Scripts\python.exe scripts/lare_mimic_run.py --seeds 5 --hours 0,1,2,4,6
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--smoke` | off | 合成自检；**不读 DB**，写 `artifacts/lare/smoke_lare.json` |
| `--seeds` | `5` | bootstrap 重采样次数，seed = `0..N-1` |
| `--hours` | `0,1,2,4,6` | 真实模式的预测时刻（`hour_index`） |
| `--model` | `artifacts/models/lgbm_mortality_12h.txt` | 打分模型；缺失 → 退出码 2 |
| `--out` | `artifacts/lare/mimic_lare.json` | 输出 JSON（`--smoke` 默认 `smoke_lare.json`） |
| `--K` / `--alpha` / `--delta` | `0.2` / `0.05` / `0.1` | 预算 / 放行漏检 / Hoeffding 失败概率 |
| `--dsn` | `postgresql://icu_dev:icu_dev@127.0.0.1:5432/mimic` | 真实模式 DSN |
| `--feat-table` / `--death-table` / `--dod-table` | `paper_feat.sample_matrix` / `paper_label.mortality_12h` / `paper_label.mortality_12h_dod` | 三张表名（见 7.2） |

### 7.2 F1 门与数据契约

- 真实模式**开头**总是打印 `F1-GATE: 需要 DUA/CITI 已确认`；未设置 `LARE_MIMIC_ALLOW=1`
  → **退出码 3**：不建连接、不读 DB。此门由 hermes 任务包（TP-20261108-001）设置。
- 表名对应：五时刻全量矩阵是 **`paper_feat.sample_matrix`**（`feat.sample_matrix` 仅 h=0），
  两套标签是 **`paper_label.mortality_12h`**（精确）与 **`paper_label.mortality_12h_dod`**（日期级粗化，已在库）；
  缺表时运行器以退出码 2 + `artifacts/lare/mimic_lare.error.json`（含 `reason`）结束——
  **不自己造标签**（LARE 的既定原则：不发明、不改标签定义）。
- 打分走 `domain.features.build.FEATURE_COLS` + 现成 LightGBM 模型（与 `domain.models.lgbm` 同口径），
  `p` 截断到 `[0,1]`。真实模式**不在本轮执行**。
- 退出码：`0` 成功 / `2` 输入或数据错误（含模型缺失，原因写 `*.error.json`）/ `3` F1 门未放行。

### 7.3 管线定义（四种策略 + 两个 LAR 口径）

每个 seed 做一次 **bootstrap 重采样**（`rng.integers(0, n, n)`；`bootstrap=False` 可关），
在重采样上拟合/评估，所以 `summary` 是 seed 均值（`inf/nan` 跳过，`feasible_rate` 为可行比例）：

| 策略 | 定义 | 依赖标签 |
|---|---|---|
| `fixed` | 固定双阈值 `(0.1, 0.3)`（label-free 基线） | 否 |
| `double` | `pac_safety_net.select_double_threshold` 在**精确标签**上拟合，两套标签交叉评估 | 是 |
| `sbr` | `selective_review.solve_sbr` 的**终端**动作表按最近邻作用到 `p`（横断面近似：horizon=1、每病例 ≤1 次复核） | 否 |
| `deferral` | cost-aware deferral 的固定阈值近似：top-`K` 风险整体转人工（`tau_low = q_{1-K}`、`tau_high = 1`） | 否 |

每个策略在两套标签下读 `review_yield_dod/death`、`missed_imminent`、`release_risk_dod/death`
（`release_risk = max(两套)`）、`budget`（review+alert 占比）；`reviewed_jaccard` 是
**「用 dod 拟合的复核集合」与「用 death 拟合的复核集合」的 Jaccard**——label-free 策略在两套标签下
完全一致（恒 1.0），只有 `double` 会低于 1，用来量化口径漂移。

另外每个 seed 跑 `fit_lar_policy(bound=none)` 与 `bound=hoeffding` 各一次，记录
`tau_low / tau_high / robust_cost / max_regret / feasible`，并单独给出「标签专属阈值漂移」
（`Δtau_low / Δtau_high / reviewed_jaccard`）。

### 7.4 输出 schema

```jsonc
{
  "config": {"mode": "smoke|mimic", "n": 2000, "seeds": [0, 1], "K": 0.2, "alpha": 0.05,
             "n_grid": 50, "delta": 0.1, "coarse_superset_violations": 0, "hours": [0,1,2,4,6]},
  "seeds": [{"seed": 0, "n": 2000, "base_rate_dod": 0.0682, "base_rate_death": 0.034,
             "strategies": {"fixed|double|sbr|deferral": {"review_yield_dod": 0.0, "budget": 0.0,
                            "reviewed_jaccard": 1.0, "n_release": 0, "tau_low": 0.0}},
             "lar": {"none|hoeffding": {"tau_low": 0.0, "feasible": true, "robust_cost": 0.0,
                                        "max_regret": 0.0, "risk_ub_dod": null}},
             "label_drift": {"tau_low_dod": 0.0, "reviewed_jaccard": 0.0}}],
  "summary": {"n_seeds": 2, "strategies": {"..seed 均值..": {}}, "lar": {}, "label_drift": {}}
}
```

`inf/nan` 一律落成 `null`（严格 JSON，`json.dumps(..., allow_nan=False)` 可过）。

### 7.5 `--smoke` 实测（n=2000, seeds=0..1, K=20%, alpha=5%, delta=0.1）

| 策略 | yield_dod | yield_death | missed | release_risk | Jaccard | budget | compliance_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed | 0.1381 | 0.0560 | 0.0215 | 0.0345 | 1.0000 | 0.2157 | 0.00 |
| double | 0.0260 | 0.0197 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.00 |
| sbr | 0.1757 | 0.0709 | 0.0225 | 0.0382 | 1.0000 | 0.1680 | 1.00 |
| deferral | 0.1562 | 0.0575 | 0.0225 | 0.0370 | 1.0000 | 0.2000 | 1.00 |

LAR：`none` 可行率 1.00（`tau_low≈0.122`、`tau_high≈0.755`、`robust_cost≈0.1846`、`max_regret≈0.0003`）；
`hoeffding` 可行率 0.00（半径 `sqrt(log(1225/0.1)/(2·2000))≈0.0485` 吃掉 5% 的风险预算 → 回落全复核）。
标签专属阈值漂移：`Δtau_low≈-0.0102`、`Δtau_high=0`、Jaccard≈0.903。

**`double` 为何在 smoke 里退化成全复核**：`select_double_threshold` 是 **PAC/LTT** 过程（单标签
Hoeffding + Bonferroni），n=2000 时半径 0.0485 与 `alpha=0.05` 同量级，只剩 0.0015 的风险预算
→ 不可行 → `(0,1)` 全复核。这恰好是它相对 `fit_lar_policy(bound=none)`（经验风险）的**代价**：
同一 alpha 下 PAC 口径需要大得多的校准集（`docs/PAC_SAFETY_NET.md` §3，约 4.7e4 行量级）；
完整 MIMIC（单时刻 ≈9.4e4 行）半径降到 ≈0.007，`double` 才恢复可用。所以读表时把 `double`
当「PAC 严格基线」、`lar.none` 当「经验口径」，两者不可混为一谈。

### 7.6 hermes 使用说明

1. **前置**：F1/DUA 确认后设置 `LARE_MIMIC_ALLOW=1`；确认 `paper_label.mortality_12h_dod` 已落库、
   `artifacts/models/lgbm_mortality_12h.txt` 存在。
2. **跑网格**：`python scripts/lare_mimic_run.py --seeds 5 --hours 0,1,2,4,6`，一条命令产出
   `artifacts/lare/mimic_lare.json`（`seeds` 数组 + `summary`）。
3. **验收判据**：退出码必须为 `0`；`summary.n_seeds == 5`；`summary.lar.none.feasible_rate` 与
   `summary.lar.hoeffding.feasible_rate` 都要记录（翻转本身就是结论）；
   `config.coarse_superset_violations` 应为 `0`（>0 说明 `y_dod` 不是 `y_death` 的超集，
   需回到标签审计 `docs/LABEL_AUDIT.md`）。
4. **失败处理**：`3` → 授权未放行（不得重试绕过）；`2` → 看
   `artifacts/lare/mimic_lare.error.json` 的 `reason`（多为表缺失 / 连接失败 / 模型缺失），
   修数据后重跑，不要改代码绕开。
5. **不要**在 hermes 里调 `--K/--alpha` 来「调到可行」：那等于拿测试集选阈值，见 §6 局限 2。

## 8. 与 conformal deferral / 现有模块的差异

| 对象 | 差异 |
|---|---|
| **Conformal deferral**（如 learning-to-defer / conformal risk control 的 deferral 变体） | 那类工作解决的是**“不确定就转人工”的覆盖率/风险保证**，通常只针对**一个**标签分布。LARE **不是新的 deferral 算法**：复核带（`review`）沿用 H9/PAC 的双阈值形式，本文的新意在于**标签歧义下的评估口径（两套标签同时报告）**与**鲁棒选阈值（min-max）**——即“同一个 deferral 策略在两套标签下是否都成立”。 |
| `pac_safety_net`（H9-I4） | I4 在**单一标签**上给放行漏检的**有限样本上界**（Hoeffding + Bonferroni）。LAR-Policy 在**两套标签**上给**经验**约束 + min-max 目标；二者可组合：把 I4 的上界分别算在两套标签上，再取联合可行域（后续工作）。 |
| `selective_review`（H9 SBR） | SBR 是病例级、时序、预算约束下的**最优停止**（软目标：最小期望代价）。LAR-Policy 是**群体双阈值**（硬约束 + 鲁棒目标），可作为 SBR 的 `DoubleThresholdPolicy` 基线或闸门。 |
| `utility_calibrate`（H4） | H4 用单一标签 + 代价比选净收益最优阈值；LAR-Policy 把“选哪个标签”本身当成不确定性来源。 |
| `LABEL_AUDIT`（P1） | P1 是**标签审计**（口径差异的量化），LARE 是**策略审计**（口径差异如何污染策略评估与选择）。 |

## 9. 下一步

1. **把 PAC 界搬到两套标签**：对每个候选 `(tau_low, tau_high)` 分别算
   `U_dod`、`U_death`（Bonferroni 分摊到 2×候选数），取 `max(U_dod, U_death) <= alpha` 的可行域。
2. **滚动重选**：按周重跑 `fit_lar_policy`，报告 `feasible` 翻转率、`tau_low` 抖动、
   两口径 Jaccard 的时间趋势（漂移检测）。
3. **真实 dod 生成机制**：用 `admissions.deathtime` 与 `patients.dod` 的实际差值估计
   `add_rate(p)`（按风险分层），替换 demo 里的线性假设。
4. **与 P1 alert-budget 联合**：把 `n_alert/n <= B` 加进候选筛选，一次解出
   “放行漏检 × 报警预算 × 标签歧义”的三方约束。

## 10. ε-最优策略集 API（`domain/models/policy_set.py`）

`fit_lar_policy` 返回**唯一**的 `(tau_low, tau_high)`，但有限样本上最优候选与邻域的
`robust_cost` 差异常小于抽样噪声——"唯一最优"不可识别，汇报时容易把噪声当结论。
`policy_set` 把选择结果从**单点**扩展为**ε-最优集合**（Rashomon set 口径）：

    S(ε) = { θ : feasible(θ) 且 robust_cost(θ) <= min_θ robust_cost(θ) + ε }

口径与 `lar_policy` **完全一致**（同网格、同代价公式、同约束、同三区定义），纯函数、只依赖
`numpy`（复用 `lare.double_policy` 重建掩码），不读 DB、不跑 MIMIC、无随机性。

### 10.1 三个函数

| 函数 | 作用 | 关键契约 |
|---|---|---|
| `enumerate_policies(p, y_dod, y_death, *, n_grid=50, K=0.2, alpha=0.05, cost_fn=4.0, cost_fp=1.0, review_cost=0.2)` | 枚举网格 `lows=highs=linspace(0,1,n_grid)` 上 `tau_low < tau_high` 的全部 `n_grid(n_grid-1)/2` 个候选 | 每条记录键固定：`tau_low,tau_high,cost_dod,cost_death,robust_cost(=max),risk_dod,risk_death,budget,n_release,feasible`；顺序 = `tau_low` 外层升序、`tau_high` 内层升序 |
| `epsilon_optimal(records, epsilon)` | 只在 `feasible=True` 中取 `robust_cost <= opt + ε`（`opt = min(robust_cost)`） | 保输入顺序；无可行记录 → `[]`（不可行本身是结论）；`ε<0` 或非有限 → `ValueError`；**不给默认 ε** |
| `summarize_set(p, policies, *, include_rates=True, max_rate_n=10000)` | 集合摘要 | `n_policies, tau_low_min/max, tau_high_min/max, robust_cost_min/max, n_release_min/max`；`include_rates` 且 `len(p) <= max_rate_n` 时额外给 `consensus_rate = mean(∩ review_k)`、`contested_rate = |∪-∩| / |∪|`，否则这两个键为 `None`（不做、不假装做了） |

`budget = mean(review|alert) = P(p >= tau_low)`；`risk_* = mean(y=1 且 release)`
（全人群口径的放行区漏检质量）；`feasible = budget<=K 且 risk_dod<=alpha 且 risk_death<=alpha`（容差 `1e-12`）。

### 10.2 怎么读摘要

- `robust_cost_max - robust_cost_min <= ε`：集合定义的自校验（也写在单测里）。
- `tau_low_max - tau_low_min` **大** → 阈值怎么挪代价都差不多，**别把某条阈值当唯一解**；
  反过来**小**（集合退化到单点）→ 结论对阈值敏感，需要更多样本/更强约束再说。
- `consensus_rate` 高 → 集合内对"谁来复核"共识强，可直接取交集作为**安全复核集**；
  `contested_rate` 高（如示例 seed=0 的 `0.848`）→ 集合内部只对少数人达成一致，
  这部分人属于"口径分歧带"，应作为人审优先项而不是部署依据。
- 两个 rate 需要逐策略重建掩码（`O(len(p)·n_policies)`），大样本时用 `include_rates=False`
  或压低 `max_rate_n`，取值不参与任何约束判定。

### 10.3 用法与坑

```python
from domain.models.policy_set import enumerate_policies, epsilon_optimal, summarize_set

records = enumerate_policies(p, y_dod, y_death, n_grid=50, K=0.2, alpha=0.05)
chosen = epsilon_optimal(records, 0.004)          # ε 由调用方决定，模块不给默认值
summary = summarize_set(p, chosen)                # 阈值/代价/人数区间 + 共识/争议
```

1. **ε 是代价绝对差**（与 `robust_cost` 同量纲），不是相对比例；`0.004` 只是示例口径，
   换数据集必须重议（与 `cost_fn/cost_fp/review_cost` 的量纲绑定）。
2. **不要在测试集上扫 ε 挑好看的集合**：ε 属于评估口径，应在校准集上固定，理由同 §6 局限 2。
3. **候选数**：`n_grid=50` → 1225 条记录；`n_grid=200` → 19900 条，逐候选记录是 Python
   级 dict，注意内存。阈值统计用前缀和（每个阈值 `O(log n)`），不做 `O(n·m)` 掩码。
4. **与 `fit_lar_policy` 的关系**：选中点必然在同一网格上且属于 `S(ε)`（单测
   `test_selected_point_belongs_to_epsilon_set` 守着这条跨模块一致性）；本模块**不改**
   `lar_policy` / `lare`，也不提供 `bound="hoeffding"` 的 ε 集合（后续工作：把 PAC 上界
   并入可行性判定）。
