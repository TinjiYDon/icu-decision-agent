# 总纲领 · 一条线串起全仓（2026-11-03）

> SSOT 叙事入口。读完应能回答：项目做什么、主创新是什么、证据在哪、哪些不主张。
> 阶段门禁 → [STAGE.md](STAGE.md)；人话版 → [BRIEF.md](BRIEF.md)；完整技术说明 → [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md)；数字 → [STATUS.md](STATUS.md)。

---

## 1. 总目标（North Star）

> 在时间因果约束下，为 ICU 入科后 0/1/2/4/6 小时提供“可决策、可解释、可人审”的 12 小时恶化预警；**先把标签语义讲清楚，再谈复核策略好坏**。

**当前主创新（S3–S4）**

| ID | 主张 | 一句话 |
|---|---|---|
| D-B | 标签时间精度审计 | `dod` 是 `deathtime` 的粗化超集（dod-only 5,401，48.9%），会改变阈值、排名与预算合规 |
| D-M1 | **LARE** | 同一复核策略在两套标签下评估：yield 差 +0.0839、复核名单 Jaccard 0.2446（合成）/0.2429（MIMIC） |
| D-M2 | LAR-Policy | 对两套标签做 min-max 的鲁棒双阈值：robust_cost 0.15995 / max_regret 0.0134（合成）；0.0708 / 0.0039（MIMIC） |
| D-M3 | Hoeffding 上界（可选） | 数据不够时不允许说“可行”：n=4000/α=5% 直接不可行 |

**明确不主张**：H4 净受益仍为负（−0.0354）；H5 PR-AUC 反降（0.1637→0.1528）；H6 CUGEW 已撤回（三指数共线、SUFH 18,891 行只报 4 行）；SBR/PAC 是工程基线，不是算法创新；证据门控/人补证据已被 EGCL/OpenMed/QCC 占。

---

## 2. 阶段（S0–S4）

| 阶段 | 目标 | 状态 |
|---|---|---|
| S0 问题定义 | 一句话 + 不做什么 | ✅ |
| S1 证据底座 | 标签审计（5,401/48.9%、阈值 0.145→0.165/0.250） | ✅ |
| S2 方法原型 | LARE + LAR-Policy 合成跑通（20 passed） | ✅ |
| S3 真实数据 | MIMIC 472,290 行两标签实验 | ✅ |
| S4 结论成稿 | 四张表 + 负结果 + 限制 | 🔄 |

门禁：`python ~/.dsh/skills/complex-project-harness/scripts/preflight.py --config preflight.yaml`

---

## 3. 一条链（不是并列 Demo）

```text
MIMIC stay → 特征(19, charttime<intime+h) → 标签(12h, deathtime 优先)
  → LightGBM p(s,h)（64 树/depth4/lr0.1, seed42）
  → 动作：放行 / 复核 / 报警（双阈值）
  → LARE：在两套标签下算 yield / 漏检 / 复核名单 Jaccard
  → LAR-Policy：min max(cost_dod, cost_death) + 预算 K + 风险 α
  → 结论：标签歧义是否改变评估；鲁棒策略的 max_regret
```

**核心取舍**：漏报代价 4、误报 1、复核 0.2；复核+报警 ≤ K；放行漏检 ≤ α。

---

## 4. 证据表（当前）

| 结论 | 数字 | 证据 |
|---|---|---|
| dod 是粗化超集 | dod-only 5,401（48.9%）；基准率 1.19%→2.34% | `LABEL_AUDIT.md` |
| 标签改变工作点 | 阈值 0.145→0.165/0.250；ΔPR-AUC −0.0174 | 同上 |
| 同一策略两标签 yield | +0.0839（合成）；Jaccard 0.2429（MIMIC） | `artifacts/lare/` |
| 鲁棒策略 | robust_cost 0.0708、max_regret 0.0039（MIMIC） | `artifacts/lare/mimic_lare.json` |
| 数据不足不可行 | n=4000/α=5% radius≈0.0343 → infeasible | `PAC_SAFETY_NET.md` |

---

## 5. 读者路径

| 你是谁 | 先读 | 再读 |
|---|---|---|
| 非本领域/答辩 | [BRIEF.md](BRIEF.md)（分诊台类比 + 五步链路） | [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md) |
| 算法深挖 | [LARE.md](LARE.md) | [ARCHITECTURE_MATH.md](ARCHITECTURE_MATH.md) |
| 数字/复现 | [STATUS.md](STATUS.md) | [DUMP_READY.md](DUMP_READY.md) |
| 历史汇报稿 | [archive/FULL_REPORT.md](archive/FULL_REPORT.md) | 仅历史参考，口径以本文为准 |

---

## 6. 3 分钟故事线

1. **问题**：ICU 每小时只有少量复核名额，漏报代价远大于误报。
2. **发现**：两套死亡日期记录（精确 vs 日期）会让同一复核策略看起来好坏完全不同。
3. **做法**：LARE 在两套标签下评估；LAR-Policy 选“换标签也不至于太差”的阈值。
4. **证据**：MIMIC 上 Jaccard 0.2429、robust_cost 0.0708、max_regret 0.0039。
5. **边界**：单中心；不主张 SBR/PAC/H4/H5/H6 为创新；不写临床效果。

---

## 7. 文档唯一入口

| 类型 | 入口 |
|---|---|
| 总叙事 | 本文件 |
| 人话版 | BRIEF.md |
| 技术总览 | TECHNICAL_OVERVIEW.md |
| 阶段门禁 | STAGE.md |
| 方法/公式 | LARE.md |
| 数字 | STATUS.md |
| 历史 | archive/（保存不删） |
