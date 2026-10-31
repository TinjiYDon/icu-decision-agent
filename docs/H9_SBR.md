# H9 · 选择性复核（SBR）：预算约束下的序贯停止

> 状态：**研究原型（synthetic-only）**。不读数据库、不调 LLM；复核者用 sensitivity/specificity 显式模拟。
> 代码：`domain/models/selective_review.py` · 脚本：`scripts/h9_sbr_pilot.py` · 测试：`tests/test_selective_review.py`
> 关联：H4 群体阈值、H5 多时刻融合、H3 解释/HITL。**本模块不替代 H4/H5，也不改变在线推理路径。**

## 1. 问题

现有转交（deferral）方法多为**一次性**决策：给一个风险分，决定“交不交人”。ICU 的实际决策是**多时刻**的：h=0 看到风险 0.3，可以现在复核，也可以等 h=1 的化验回来再决定；复核人力（当班 K 次）是硬约束。

SBR 把这件事写成**有限时域、预算约束的最优停止**：

```
状态 x = (p, k, t)      p=风险分；k=剩余复核次数；t=决策时刻
动作 a ∈ {放行, 升级, 复核, 等待}
代价：放行漏报 C_FN、升级误报 C_FP、每次复核 c_r
目标：min E[cost]  s.t.  复核次数 ≤ K
```

## 2. 形式化

Bellman（T 为最后时刻）：

```
V_T(p,k) = min{ p·C_FN, (1−p)·C_FP, (k>0 ? c_r + E_m[best(p_m)] : ∞) }
V_t(p,k) = min{ p·C_FN, (1−p)·C_FP,
                c_r + E_m[best(p_m)]      (k>0),
                E_{p'|p}[V_{t+1}(p',k)]   (t<T) }
```

其中 `p_m` 是复核结论 m 的后验：`P(y=1|m=1)=p·sens/(p·sens+(1−p)(1−spec))`，`P(y=1|m=0)=p(1−sens)/(p(1−sens)+(1−p)spec)`；`best(p)=min(p·C_FN,(1−p)·C_FP)`。

v1 简化：复核后立即行动，不再等待（见 §7）。

## 3. 算法与基线

- **SBR-DP**：把 p 离散成网格（默认 41 格），用 `default_transition()` 的粘滞随机游走或 `estimate_transition()` 的病例轨迹估计，反向 DP 求 `V/A`。
- 基线：`release_all`、`alert_all`、`fixed_0.5`、`fixed_0.384`（H4 旧 S2 工作点）、`double_threshold(0.2,0.6)`、`myopic_review`（Δ 贪心，只有在复核能降低即时期望代价时才复核）。

## 4. 运行

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe scripts/h9_sbr_pilot.py --n 2000 --horizon 5 --budget 3 --seed 0
```

输出：终端 Markdown 表 + `artifacts/h9/sbr_pilot_<seed>.json`。

## 5. 合成 pilot（2026-10-31，sens/spec=0.8/0.8，K=3，n=2000）

| policy | mean_cost | miss_rate | alert_rate | review_rate | review_yield |
|---|---:|---:|---:|---:|---:|
| release_all | 0.2511 | 0.102 | 0.000 | 0.000 | 0.000 |
| alert_all | 0.9372 | 0.000 | 1.000 | 0.000 | 0.000 |
| fixed_0.5 | 0.2442 | 0.101 | 0.004 | 0.000 | 0.000 |
| fixed_0.384 | 0.2338 | 0.098 | 0.013 | 0.000 | 0.000 |
| double_threshold | 0.1999 | 0.086 | 0.029 | 0.082 | 0.671 |
| myopic_review | 0.1993 | 0.078 | 0.054 | 0.157 | 0.462 |
| **sbr_dp** | **0.1630** | **0.063** | 0.087 | 0.166 | 0.482 |

**这不是临床证据**：轨迹为合成、复核者为模拟、代价为相对单位。仅用于验证实现与方向；MIMIC 实验须在数据授权（F1）明确后进行。

> `oracle_gap` 在 JSON 中是“完美信息下界（0）”的差，等于 mean_cost，不单独解读。

## 6. 与现有模块的关系

| 模块 | 关系 |
|---|---|
| H4 `utility_calibrate` | 提供群体阈值 τ*；SBR 在其上做病例级、时序复核决策 |
| H5 `sparsity_fusion` | `p_fuse` 可直接作为 SBR 的 p |
| H3 `hitl_audit` | 真实 accept/reject/edit 记录可用于估计 reviewer 的 sens/spec（当前为模拟） |
| PAC 安全网（待做） | 在放行区加“漏报率 ≤ α”的有限样本保证；与 SBR 解耦实现 |

## 7. 局限

1. **模拟复核者**：没有真实医生标签，sens/spec 是假设；只能写“模拟专家下的离线结论”。
2. **复核后立即行动**：v1 不建模“复核后再等待”；后续把 review 动作接到下一时刻状态即可。
3. **合成轨迹**：真实 MIMIC 轨迹需 F1；转移矩阵可从已有 5 时刻预测估计（不改实验，只是读已落盘结果）。
4. **无临床声明**：不做“降低死亡率/漏诊率”的表述。

## 8. 下一步

1. 用 HITL 审计数据估计 reviewer 的 sens/spec（至少给出区间），替换默认 0.8/0.8。
2. 把 SBR 的 p 换成 H5 的 `p_fuse`/H6 的 `p_rw`，比较三种 p 下的复核预算–漏报前沿。
3. 加 PAC 双阈值安全网：`P(y=1, release) ≤ α`，与 SBR 解耦、可单独验证。
4. F1 通过后，在 MIMIC 上跑同协议；否则用 PhysioNet 2012（公开）做 pilot。
