# 唯一指标台账（METRICS LEDGER）

> **更新日期**：2026-11-21（阶段 0 合仓清淤完成）  
> **规则**：本文件是全仓实验数字与口径的**唯一登记处**。任何论文、PPT、结题报告中引用的数字，必须可追溯到本台账中带 `Git Commit + 数据版本 + 分母口径` 的条目；标记为 `[已作废]` 的历史数字禁止再混入新主表。

---

## 1. 四套历史口径冲突对照与作废声明

本项目在 2026 年 7 月至 11 月经历了四次数据与实验口径演进，不同汇报材料之间曾出现数字打架。现正式划清边界：

| 批次 | 状态 | 来源 / Commit | 数据规模与切分 | 关键特征与标签 | 代表指标（仅作溯源） | 使用边界 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **V0 旧 S2 底座** | `[已作废]` | `artifacts/models/metrics_mortality_12h.json`（2026-08-05）· dump `20260802` | 94,458 stays / 472,290 行 · 按 `stay_id` 70/10/20（测试 18,891 stays / 94,455 行） | 19 特征（**体征全空**）· `dod` 阳性率 2.34% · `scale_pos_weight` | ROC `0.779` · PR-AUC `0.139` · Brier `0.040`；旧 H4 NB `-0.053→-0.035`；旧 H6 SUFH Brier `0.022` | **禁止再作主结果**（体征缺失、旧 `dca.py` 符号反了、H6 仅补校准） |
| **V1 旧 S2 + v2 标签** | `[已作废 / 仅历史审计]` | `main@177f2cc`（2026-11-11）· `docs/STATUS.md` / `scripts/lare_mimic_run.py` | 同 472,290 行 · `stay_id` 切分 · 5 次按行重抽 | 同 19 特征（**体征全空**）· `deathtime` 1.19%（5,638 阳）vs `dod` 2.34%（11,039 阳，dod-only 5,401 / 48.9%） | 基线 PR-AUC `0.092` · Brier `0.151`；LARE K=0.35 yield `1.51%` vs `3.17%`；Jaccard `0.2429`（K=0.20）/ `0.2405`（K=0.35）；LAR robust `0.0708` | **仅保留标签审计事实**（dod-only 占比 48.9%）；模型与 LARE 须在 V3 重跑 |
| **V2a 中期受控粗化独立验证** | `[待同步代码]` | 2026-11 中期报告 §3.8 / 附录 B（`scripts/lare_controlled_validation.py` **尚未推送至 origin**） | 93,673 stays · 按 `subject_id` 60/20/20（测试 12,980 人 / 18,515 stays）· 单时点 `h=1` | 6 项入科前/首小时变量 · 同源 `deathtime` 日级粗化（阳性 `213 → 397`） | 10% 预算、代价 `4:1:0.2` 下 Jaccard `1.00`（两套均复核 1,851 例、0 升级）；漏报权重 32 时 Jaccard 降至 `0.081`、代价 `0.2456 → 0.2821` | 待合入 `lare_controlled_validation.py` 并在 V3 多自由度策略空间复现升级 |
| **V2b 补体征 8 模型对照（B0–B7）** | `[待同步代码]` | 2026-11-18 修订版 PPT（**脚本与产物尚未推送至 origin**） | 测试集 94,645 行（**13,073 位病人**，约 20% 患者级切分）· 阳性 1,154 行（1.22%） | 补齐 **3,542 万条体征** · 无 `scale_pos_weight` · 8 模型（B0–B7）同切分 | B4 旧 19 特征 AP=`0.125` · B3 逻辑回归 AP=`0.244` · **B5 全体征时序 AP=`0.323`（Brier `0.0097`）** · B6 等渗校准 ECE=`0.0022` · **B7 五种子集成 AP=`0.334`（Brier `0.0096`）** · 50 条/千时点敏感度 `39.2% → 69.2%`；Fig 6 双口径复核 Jaccard `0.51` | **当前预测底座上限参考**；须先将 B0–B7 代码合入主仓并补齐 `storetime` 与在院过滤审计，再转为 V3 正式数字 |

---

## 2. 仓库代码同步清单（截至 `v2026.11-midterm-sync` = `177f2cc`）

### 2.1 已合入本仓 `main` 的 11 月模块（单元测试 90 passed）
- 受控时间粗化纯函数：`domain/labels/time_coarsening.py` + `tests/test_time_coarsening.py`
- LARE 评估协议：`domain/models/lare.py` + `tests/test_lare.py`
- LAR-Policy 与 Hoeffding 上界：`domain/models/lar_policy.py` + `tests/test_lar_policy.py`
- $\varepsilon$-最优策略集与共识/争议摘要：`domain/models/policy_set.py` + `tests/test_policy_set.py`
- 选择性复核与 PAC 安全网基线：`domain/models/selective_review.py`、`domain/models/pac_safety_net.py`
- MIMIC 运行器与导入工具：`scripts/lare_mimic_run.py`、`scripts/import_mimic_31.py`

### 2.2 仍滞留仓外、待队友合入的 2 组文件（阶段 0.1 跟踪项）
1. **11 月 18 日 B0–B7 全量体征特征提取、8 模型训练与画图脚本**（产出 AP=`0.323`/`0.334`、Brier=`0.0096` 的代码与特征清单）。
2. **中期报告附录 B 的 `scripts/lare_controlled_validation.py`**（产出 93,673 stays 单时点受控粗化汇总表）。

---

## 3. V3 正式封版指标登记表（待阶段 1–4 依次填入）

> 所有 V3 结果必须满足：**化验 `storetime <= t` 审计完成 + 每时点在院存活过滤（`t < outtime & t < deathtime`）+ 按 `subject_id` 60/20/20 切分 + 1,000 次患者级整组 Bootstrap 95% CI**。

### 3.1 预测骨干与目标函数消融（M0 → M1a → M1b → M2a）

| 模型版本 | 数据版本 / SHA256 | 校准方式 | Test PR-AUC (95% CI) | Test AUROC | Test Brier | Test ECE(15) | 产物路径 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **M0**（无权重强基线） | `dataset_v3_causal` (待封版) | 无 / Platt / Isotonic | — | — | — | — | `artifacts/v3/m0/` |
| **M1a**（事件权重归一） | `dataset_v3_causal` (待封版) | Platt / Isotonic | — | — | — | — | `artifacts/v3/m1a/` |
| **M1b**（提前量加权归一） | `dataset_v3_causal` (待封版) | Platt / Isotonic | — | — | — | — | `artifacts/v3/m1b/` |
| **M2a**（事件级 LSE+BCE `fobj`） | `dataset_v3_causal` (待封版) | Platt / Isotonic | — | — | — | — | `artifacts/v3/m2a/` |

### 3.2 动态复核队列事件级主对比（固定实际提醒预算 $B \in \{20, 50, 100\}$ / 千 stay-hour）

| 组合（模型 × 策略） | 预算档位 $B$ | 实际报警负荷 | 独立死亡首次覆盖率 $\mathrm{Cov}_{\mathrm{event}}$ (95% CI) | 首次提前时长中位数 (IQR) | Event-NNR | 重复提醒率 | 漏报事件数 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M0 × D0`（静态直选） | 50 | — | — | — | — | — | — |
| `M0 × D1`（静默排重） | 50 | — | — | — | — | — | — |
| `M0 × D2`（静默+跃升再报） | 50 | — | — | — | — | — | — |
| `M2a × D2`（目标改写+动态队列） | 50 | — | — | — | — | — | — |

### 3.3 LARE 同源粗化与 $\varepsilon$-近优归因主表（基于 D2 / D2-Dual 策略空间）

| 粗化粒度 $\Delta_g$ | 标签翻转率 | 精确测试覆盖差 $\Delta\mathrm{Cov}$ (95% CI) | 住院级 Jaccard | 同标签近优离散度 $D_{\mathrm{within}}(\varepsilon)$ | 跨标签置换度 $D_{\mathrm{cross}}(\varepsilon, g)$ | 共识/争议复核占比 | 同率随机对照 $\Delta\mathrm{Cov}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1h` | — | — | — | — | — | — | — |
| `6h` | — | — | — | — | — | — | — |
| `12h` | — | — | — | — | — | — | — |
| `24h (day)` | — | — | — | — | — | — | — |
