# H9-I4 · PAC 双阈值安全网（有限样本漏检控制）

> 状态：**研究原型（synthetic-only）**。不读数据库、不调 LLM、不跑 MIMIC。
> 代码：`domain/models/pac_safety_net.py` · 脚本：`scripts/h9_pac_demo.py` · 测试：`tests/test_pac_safety_net.py`
> 关联：[H9_SBR.md](H9_SBR.md)（选择性复核）、H4 群体阈值、P1 alert-budget。

## 1. 问题

H4 用**固定阈值**把风险分切成“预警/不预警”，H9 SBR 在其上加病例级的复核决策。两者都缺一件事：
**“放行”这件事本身没有有限样本保证**——阈值是在同一批校准数据上挑的，报告出来的漏检率是经验值，
换个队列就可能翻倍。论文评审最容易打的就是这一条（“your threshold is tuned on test data”）。

I4 的做法：**只在放行区上加一层可证伪的统计承诺**，用 Learn-then-Test（LTT）在多候选阈值上做
Hoeffding + Bonferroni 的保守选择，使

```
P( R(tau_low) <= alpha ) >= 1 - delta        （对校准集的抽样分布而言）
```

其中 `R` 是**放行区漏检质量**（不是条件漏检率）：

```
R(tau_low) = E[ 1{y=1} · 1{p < tau_low} ]         # 全人群口径：在全体病人上的漏检比例
conditional_miss_rate = R / max(release_rate, eps) # 放行人群口径：仅报告，不做保证
```

选全人群口径的理由：它直接对应“每 100 个 ICU 病人里最多放走 α 个阳性”，不需要对放行比例做假设；
条件口径在 release_rate 很小时方差爆炸，做保证没有意义。

## 2. 三区规则

```
release : p < tau_low              # 不报警、不人工复核
review  : tau_low <= p < tau_high  # 进 HITL 复核队列（H9 SBR / H3 解释）
alert   : p >= tau_high            # 直接报警
```

三区互斥且并集为全集（`n_release + n_review + n_alert == n`），由单测保证。

## 3. 算法：LTT + Hoeffding + Bonferroni

给定校准集 `{(p_i, y_i)}_{i=1..n}`（p 已校准、y 为 0/1）：

1. **候选集**：`tau_low ∈ linspace(0,1,n_low)`（默认 50 点）、`tau_high ∈ linspace(0,1,n_high)`
   （默认 50 点），只保留 `tau_low < tau_high` 的组合；候选数 `m = Σ_i #{j: tau_high,j > tau_low,i}`
   （方形网格 `n_low=n_high=g` 时 `m = g(g-1)/2`，默认 `g=50` → 1225）。
2. **Bonferroni**：每个候选单独用 `delta_m = delta / m`（候选数越多，单候选越保守）。
3. **Hoeffding 上界**（有界变量 ∈ [0,1]，无需方差估计）：

   ```
   R_hat = (1/n) Σ_i 1{y_i=1} · 1{p_i < tau_low}
   U     = R_hat + sqrt( log(1/delta_m) / (2n) )
   ```

   注意 `U` 只依赖 `tau_low` 与 `(n, m, delta)`；`tau_high` **不进入**该风险（见 §6）。
4. **LTT 选择**：在 `U <= alpha` 且 `release_rate >= min_release_rate` 的候选中，
   取**放行人数最多**的那个（临床意图：在满足安全承诺的前提下尽量少打扰）；
   同分时取更小的 `tau_high`（复核带更宽，更保守）。
5. **无可行候选** → `feasible=False`，回落到 `(tau_low, tau_high) = (0.0, 1.0)`、`risk_upper_bound = 1.0`，
   即“全部进复核”。

**样本量下界**（零风险也要能通过上界）：

```
n >= log(m / delta) / (2 * alpha^2)
```

默认 `n_low=n_high=50, delta=0.1` 时：`alpha=1% → n >= 47067`；`alpha=5% → n >= 1883`。
这是 Hoeffding 的代价（不估计方差），换来的是**无需任何分布假设**；若要 `n` 更小，可换 Bernstein/经验
Bernstein 或 `d_OBF`/ betting 边界（后续工作，见 §8）。

## 4. API（`domain/models/pac_safety_net.py`）

| 符号 | 说明 |
|---|---|
| `DoubleThresholdResult` | `tau_low/tau_high/feasible/empirical_risk/risk_upper_bound/alpha/delta/n_cal/n_release/n_review/n_alert/conditional_miss_rate` |
| `release_mask(p, tau_low)` | `p < tau_low` |
| `review_mask(p, tau_low, tau_high)` | `tau_low <= p < tau_high`（`tau_high < tau_low` → `ValueError`） |
| `alert_mask(p, tau_high)` | `p >= tau_high` |
| `missed_mass(p, y, tau_low)` | `R_hat` |
| `hoeffding_radius(n, delta_m)` | `sqrt(log(1/delta_m)/(2n))` |
| `select_double_threshold(p, y, alpha=0.01, delta=0.1, n_low=50, n_high=50, min_release_rate=0.0)` | LTT 选择 |

约定：

- 输入先 `np.asarray(..., dtype=float)`；`p` 截断到 `[0,1]`；`y` 只接受 0/1，否则 `ValueError`；`p/y` 长度必须一致。
- `alpha<=0`、`delta∉(0,1)`、`n_low<2`、`n_high<2`、`min_release_rate∉[0,1]` → `ValueError`。
- 纯函数、无全局状态、无随机数（同输入两次调用结果完全相同）。
- 只依赖 `numpy`；不读 DB、不访问网络、不跑 MIMIC。

> 说明：任务书写明“`alpha<=0` → `ValueError`”，同时又列了一条“`alpha=0.0` 时 `feasible=False`”。
> 两者冲突，这里按**更严格的 API 契约**实现（`alpha=0` 视为调用错误），
> “不可行”用任意小的正数（如 `alpha=1e-12`）表达，单测 `test_alpha_zero_raises_and_tiny_alpha_is_infeasible`
> 同时覆盖这两点。

## 5. 合成 demo

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe scripts/h9_pac_demo.py --n 4000 --alpha 0.01 --delta 0.1 --seed 0
```

数据：`p ~ Beta(0.4, 6.0)`，`y ~ Bernoulli(0.15p + 0.02)`（`np.random.default_rng(seed)`）。输出终端摘要 +
`artifacts/h9/pac_demo.json`。

**(a) 默认配置 `n=4000, alpha=1%, n_low=n_high=50`（2026-10-31, seed=0）— 不可行**

| 量 | 值 |
|---|---:|
| Bonferroni 半径（m=1225） | 0.034302 |
| tau_low / tau_high | 0.0 / 1.0 |
| feasible | **False** |
| risk_upper_bound | 1.0 |
| release / review / alert | 0 / 4000 / 0 |
| 基线 `tau_low=0.2` | release 3681（0.920），`R_hat`=0.0213，`U(m=1)`=0.0382，`U(m=1225)`=0.0556 |

结论：`n=4000` 配 `alpha=1%` 时，连“零风险候选”都过不了 Hoeffding 上界（需 `n>=47067`），
安全网自动**关闭放行区**。这正是想要的行为：**没有把握就全进复核，而不是给一个假阈值**。

**(b) `n=4000, alpha=5%, n_low=n_high=50`（seed=0）— 可行**

| 量 | 值 |
|---|---:|
| tau_low / tau_high | 0.0816 / 0.1020 |
| feasible | True |
| empirical_risk（`R_hat`） | 0.0150 |
| risk_upper_bound | 0.0493 ≤ 0.05 |
| release / review / alert | 2963 / 182 / 855 |
| release_rate | 0.7408 |
| conditional_miss_rate | 0.0203 |

即：在 5% 的放行漏检预算下，74% 的病人可放行（其中 1.5% 全人群口径的阳性被放走，上界 4.93%），
其余 4.6% 进复核、21.4% 直接报警。对比基线固定阈值 0.2 的 `R_hat=0.0213`——**即便只看经验值，
固定阈值也已经超出 1% 预算**，没有上界可用。

> 这不是临床证据：标签是合成的、`p` 不是模型输出、没有任何 MIMIC 病人。

## 6. 局限

1. **交换性假设**：保证只在“校准集与部署队列同分布（可交换）”时成立；ICU 季节、病种、收治策略漂移会破坏它。
   缓解：滚动校准窗 + 漂移检测，或用加权 LTT（协变量移位下界）。
2. **校准集与阈值同源**：LTT 的有效性要求校准数据未参与阈值/模型训练（`p` 需是 out-of-fold 分数）。
3. **只保证放行区**：从定义上 `R` 只看 `p < tau_low`；进入 `alert` 区或 `review` 区的阳性不再计入 `R`，
   因此“升级区的过度报警”和“复核区的人力耗费”都不在本保证的范围内（前者属于 P1 alert-budget）。
4. **两个阈值只共用一个风险量**：`tau_high` 只影响三区划分（选点时取最保守的可行 `tau_high`），
   不参与 `R` 的保证；若要“复核量 ≤ B”这类约束，需在同一候选集上追加 `n_review/n <= B` 的筛选（未实现）。
5. **Hoeffding 保守**：默认网格下 `alpha=1%` 需要 ~4.7 万校准样本；真实 MIMIC 单中心 ICU 队列通常不够，
   届时要么放宽 `alpha`、要么缩网格、要么换更紧的界。
6. **无临床声明**：不写“降低漏诊/死亡率”，只是统计承诺 + 决策规则。

## 7. 与 H9 SBR / P1 alert-budget 的关系

| 模块 | 关系 |
|---|---|
| H9 SBR（`domain/models/selective_review.py`） | SBR 决定“复核谁、复核几次”，是**代价最优**的软目标；I4 是**硬约束**安全网。二者解耦实现，I4 的 `tau_low/tau_high` 可作为 SBR `DoubleThresholdPolicy` 的输入/基线，也可只用于闸门：`p < tau_low` 直接放行，其余交给 SBR。 |
| H4 `utility_calibrate` | 给出代价最优群体阈值 τ*；I4 在其上回答“τ 取这么低，漏检质量的上界是多少”。 |
| H5 `sparsity_fusion` | `p_fuse` 可直接作为 I4 的 `p`（要求 out-of-fold）。 |
| P1 alert-budget | 控的是**升级区总量**（`P(alert) <= B`，防报警疲劳）；I4 控的是**放行区漏检**（防放走阳性）。两者是互补的两侧约束：I4 允许 `tau_high` 往下压（扩大复核带）来省报警预算，但代价是复核人力。 |
| H3 `hitl_audit` | 复核区最终的人工结论可用来评估“复核带是否真的接住了漏检”，进而回填 `review` 的有效性。 |

## 8. 下一步

1. **更紧的界**：Bernstein / 经验 Bernstein / betting（`d_OBF`）替代 Hoeffding，把 `n` 需求从 4.7 万降到数千级；
   同时比较 `n_low × n_high` 的粗网格代价（Bonferroni 的 `log m` 很便宜，主要是 `1/alpha^2`）。
2. **加入 alert-budget 约束**：在候选筛选中追加 `n_alert/n <= B`，把 P1 的预算和 I4 的漏检放进同一个 LTT 选择。
3. **滚动校准**：把 `select_double_threshold` 包成“滑动窗口 + 每周重选”，并报告阈值抖动与 `feasible` 翻转。
4. **真实分数**：F1（数据授权）通过后，用 out-of-fold `p`（H5 `p_fuse` / H6 `p_rw`）在校准 split 上跑同协议，
   在 test split 上只做**一次性**验证（不做二次调参）。
