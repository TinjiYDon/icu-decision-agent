# RESEARCH PLAN（decision · 2026-11-11）

> 总目标、当前架构、下一步目标与验收。阶段门见 [STAGE.md](STAGE.md)；技术总览见 [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md)。

---

## 1. 总目标（当前版本）

> **在固定风险模型与有限复核预算下，结局时间记录精度是否会改变复核规则的选择，并在独立精确测试集上造成可重复的资源配置损失？**

不再把“新的复核/转交算法”作为主创新。三层结构：

| 层 | 内容 | 定位 |
|---|---|---|
| **L1 评估基础** | 受控粗化（1h/6h/12h/日级）+ 开发集重选规则 + 独立精确测试 + 预算曲线 | 必须完成 |
| **L2 方法增量** | ε-最优策略集 `Π_ε`、共识/争议复核集、`J_total/J_within/J_common` 漂移分解 | 论文核心 |
| **L3 扩展** | 时间与来源感知的事实纠正（模拟反馈、双快照） | 双创/后续，不混入主论文 |

---

## 2. 架构

```text
MIMIC-IV hosp/icu
  → features(19, charttime < intime+h；可用性按 storetime 校验)
  → labels: y_death（deathtime，精确）/ y_dod（日期级）
  → LightGBM p(s,h)（固定，不随标签重训）
  → LARE：release/review/alert 三区指标
  → 规则选择：开发集分别用两套标签重选阈值（同候选族、同预算）
  → 独立精确测试：覆盖率 / 每 100 复核名额目标覆盖 / 漏报 / 成本
  → L2：ε-最优策略集 + 漂移分解
  → L3（可选）：事实纠正 + 双快照
```

代码落点：

| 模块 | 路径 |
|---|---|
| 标签审计 | `docs/LABEL_AUDIT.md` |
| LARE / LAR | `domain/models/lare.py`、`lar_policy.py` |
| MIMIC 运行器 | `scripts/lare_mimic_run.py` |
| 预测 | `domain/models/lgbm.py`、`application/predict_patient.py` |
| 阶段门 | `docs/STAGE.md`、`preflight.yaml` |

---

## 3. 当前证据（截至 2026-11-11）

| 项 | 结果 |
|---|---|
| 标签审计 | dod-only 5,401/48.9%；阳性率 1.19%→2.34%；阈值 0.145→0.165/0.250；ΔPR-AUC −0.0174 |
| MIMIC K=0.20 | fixed yield 0.0313/0.0152；budget 0.3211（超限）；LAR τ=0.2816/0.6327，robust 0.0708，regret 0.0039；Jaccard 0.2429 |
| MIMIC K=0.35 | fixed yield 0.0317/0.0151；budget 0.3214（合规）；LAR τ=0.2449/0.6327，robust 0.0706，regret 0.0040；Jaccard 0.2405 |
| 合成 | yield 差 +0.0839；Jaccard 0.2446；LAR robust 0.15995/regret 0.0134；Hoeffding n=4000 不可行 |
| 代码 | `main@218bc5e`；LARE/LAR 测试 28 passed；preflight 16/16 |

**注意**：以上是“口径会改变评估”的证据，**还不是资源损失证据**；LAR 可能退化（粗化标签逐点支配时 max 恒等于粗化代价），需先核查。

---

## 4. 下一步目标（4 周）

| ID | 任务 | 交付 | 验收 |
|---|---|---|---|
| D1 | 精确死亡时间样本 + 受控粗化 1h/6h/12h/日级 | 粗化标签表 | 每级标签改变率可复核 |
| D2 | storetime 校验 + 固定风险模型 + 开发/验证/测试划分 | 合法快照 + 划分清单 | charttime 不越界；按 subject/stay 分组 |
| D3 | 开发集分别用两套标签重选规则（同候选族/预算） | 两套规则表 | 规则对象不同、预算相同 |
| D4 | 独立精确测试：覆盖率/每 100 名额覆盖/漏报/成本 | 主结果表 | 至少一个预算点出现显著差异 |
| D5 | ε-最优策略集 + 共识/争议 + 三方 Jaccard | 方法表 + 图 | `W_ε/C_ε/D_ε/contested` 可复现 |
| D6 | LAR 退化核查（数学 + 实验） | 核查报告 | 若退化，删除方法主张 |
| D7 | 事实纠正最小原型（可选，L3） | 双快照 demo | 不阻塞主论文 |
| D8 | 四张表 + 限制 + 复现记录 | 成稿材料 | 每行数字有 artifact |

---

## 5. 判定门

| 门 | 条件 | 决策 |
|---|---|---|
| G1 | 至少一个预算点，粗化规则的精确测试覆盖显著更差 | 继续主论文 |
| G2 | 差异只在一种粗化强度/一种代价比 | 降级为条件性发现 |
| G3 | `J_within` 也很低 | 主因是阈值不可识别，不能全归因标签 |
| G4 | LAR 退化或收益全部来自信息量 | 删除 LAR 方法主张，只留评估 |

---

## 6. 论文/投稿计划

- 先投 **ML4H / CHIL workshop 或 short paper**（现有材料 2–4 周可成稿）；
- 主会（IJCAI AI&Health / JAMIA）需补：联合 PAC 界、外部/时间外推、临床代价校准；
- 事实纠正（L3）作为双创系统成果或第二篇，不混入主论文主张。

---

## 7. 明确不主张

- 新 selective prediction / deferral / conformal 算法；
- LAR min-max 是理论创新；
- “首次考虑时间不明确”；
- 临床效果/医生效率提升（无真人实验）；
- 仅凭 Jaccard 断言“标签导致换人”。

---

## 8. 请老师指导（摘要）

1. L1+L2 的组合是否够方法/应用论文；还是只够 workshop？
2. 与 Nagaraj 2025、Wu & Cook 2022、Cohen 2024 的差异表述是否成立？
3. ε-集合应作核心还是附录；`ε=0.004` 的临床锚定是否可接受？
4. LAR 退化检查如何做严谨；是否直接删除方法主张？
5. 主实验“同精确时间受控粗化 + 原生字段仅审计”能否说服审稿人？
6. 资源损失要多大/多稳定才值得主论文；预算曲线选哪些点？
7. 投稿定位：ML4H/CHIL vs JAMIA/医学信息；需要哪些外部验证？
8. 事实纠正作为双创扩展是否影响主论文聚焦？

---

## 9. 关联文档

| 文档 | 用途 |
|---|---|
| [STAGE.md](STAGE.md) | 阶段门禁 |
| [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md) | 技术总览 |
| [LARE.md](LARE.md) | LARE/LAR 算法与参数 |
| [LABEL_AUDIT.md](LABEL_AUDIT.md) | 标签审计 |
| [STATUS.md](STATUS.md) | 当前数字 |
| [MASTER_NARRATIVE.md](MASTER_NARRATIVE.md) | 叙事 SSOT |

> 详细调研与改进版报告在 `analysis/docs-research/`（仓库外的工作区材料）。
