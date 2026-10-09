# 总纲领 · 一条线串起全仓

> **SSOT 叙事入口**（2026-10-09）  
> 读完本文即可回答：总目标是什么、总内容有哪些、每块如何服务总目标。  
> 细节数字 → [STATUS.md](STATUS.md)；假设验收 → [ACADEMIC_THESIS.md](ACADEMIC_THESIS.md)；  
> **架构/公式** → [ARCHITECTURE_MATH.md](ARCHITECTURE_MATH.md)；答辩话术 → [TEACHER_INNOVATION.md](TEACHER_INNOVATION.md)。

---

## 1. 总目标（North Star）

**一句话**

> 在 **时间因果硬约束** 与 **stay 级公平协议** 下，为 ICU 入科后超早期时刻提供「**可决策、可校准效用、可融合、可解释、可人审**」的死亡风险预警—协同系统；创新落在 **评价—阈值—融合—可信交付机制**，而不是再换一个黑箱模型。

**可检验的终点（答辩/结题）**

| # | 终点 | 过线标准 |
|---|------|----------|
| G1 | 预警可用 | 多时刻样本可复现；主报 **PR-AUC / Brier**；ROC 仅对照 |
| G2 | 决策有用 | 工作点由 **净受益（DCA/H4）** 选定，能说清阈值与报警负荷 |
| G3 | 对比诚实 | 表格 / 多时刻（及有 Layer0 时的时序）在同 split 下可比（H2/H5） |
| G4 | 解释可信 | SHAP→RAG→LLM 可引用校验；HITL 可采纳/驳回（H3） |
| G5 | 叙事自立 | 不耦合床位调度；不宣称无外部验证的 SOTA |

---

## 2. 总内容地图（整仓在讲同一件事）

把项目看成一条 **闭环流水线**，而不是并列的 Demo 堆叠：

```text
                         总目标：可决策的早期预警 CDSS
                                         |
     +-----------+-----------+-----------+-----------+-----------+
     v           v           v           v           v           v
  数据因果    风险骨干     决策效用     信息融合     解释协同     交付验证
  (底座)      (LGBM)      (H1/H4)     (H2/H5)     (H3)        (Demo/KPI)
     |           |           |           |           |           |
  MIMIC/dump  多时刻分数   DCA+约束NB   稀疏门控     RAG+HITL    Streamlit
  无泄漏特征   TreeSHAP    工作点       双轨协议     care_plan   STATUS
```

### 2.1 六大内容块（与代码/文档一一对应）

| 块 | 总内容 | 回答的问题 | 主产物 | 绑定假设 |
|----|--------|------------|--------|----------|
| **A 数据与因果** | Layer0/1、dump、特征窗 `charttime < t`、标签 12h 死亡 | 学的东西是否时间合法？ | `feat`/`label`、泄漏测试 | 底座 / 因果纪律 |
| **B 风险骨干** | LightGBM 多时刻预警（默认床旁） | 风险怎么算？ | `artifacts/models/` | 可复现底座（非卖点） |
| **C 决策效用** | DCA、cost_ratio、**约束 max-NB 阈值（H4）** | 该不该响、响多少？ | `artifacts/dca/` · `artifacts/h4/` | **H1 · H4** |
| **D 公平融合** | 同 split 双轨；稀疏门控多时刻融合（H5）；GRU-D/TFT 为对照/消融 | 缺测时信谁？时序何时有用？ | `artifacts/h5/` · D3 | **H2 · H5** |
| **E 解释协同** | TreeSHAP→RAG→LLM、引用校验、HITL、care_plan | 为何高风险、人能否改？ | 解释页 · audit JSONL | **H3** |
| **F 交付验证** | Streamlit、DEMO 口播、STATUS、pytest、整合清单 | 能不能演示、复现、答辩？ | UI · docs · CI 测试 | G1–G5 |

### 2.2 假设如何「串」进流水线（不是五条平行口号）

```text
特征合法(A) → 骨干打分(B) → 用效用选阈值(C/H4)
                              ↓
                    多时刻/双轨融合(D/H5) → 同一工作点下报警
                              ↓
                    解释与人审(E/H3) → 飞轮回流改进知识/阈值
                              ↓
                    用 PR-AUC/Brier/NB 验收(F)，禁止只报 ROC
```

| 顺序 | 假设 | 在流水线中的角色 |
|------|------|------------------|
| 1 | 因果 + 标签诚实（含 H1 的 DCA 评价） | 保证「学对问题」 |
| 2 | H4 效用阈值 | 保证「算完分后决策对齐临床代价」 |
| 3 | H2/H5 公平融合 | 保证「多源信息怎么合是可证伪的」 |
| 4 | H3 解释+HITL | 保证「人能信、能驳、能改」 |
| 5 | 交付指标栈 | 保证「整条链可验收」 |

---

## 3. 总目标 ↔ 现状（一张表看齐）

| 总目标切片 | 现状（旧 S2 机制主表） | 缺口 |
|------------|----------------------|------|
| G1 预警可用 | 472k 五行时刻；PR-AUC≈0.139 / Brier≈0.040 | v2 主表另栏；勿与旧 S2 混 |
| G2 决策有用 | H4：NB −0.053→−0.035，报警≈2% | 床旁默认 thr 仍可切到 utility |
| G3 对比诚实 | H5 门控 + D3 协议；稀疏半 PR-AUC↑ | Layer0 真序列未接 |
| G4 解释可信 | DX-1 评测 + HITL 种子 KPI | 真实周飞轮、引用率进 STATUS |
| G5 叙事自立 | 独立仓、禁调度耦合 | 外部验证永不假装已做 |

---

## 4. 读者路径（按角色只走一条线）

| 你是谁 | 先读 | 再读 | 去跑 |
|--------|------|------|------|
| 老师/答辩 | **本文 §1–2** → [TEACHER_INNOVATION.md](TEACHER_INNOVATION.md) | [STATUS.md](STATUS.md) 数字 | [DEMO_SCRIPT.md](DEMO_SCRIPT.md) |
| 组员开发 | 本文 §2 → [TEAM_DIRECTION.md](TEAM_DIRECTION.md) | [ROADMAP.md](ROADMAP.md) | [INTEGRATION_PREP.md](INTEGRATION_PREP.md) |
| 算法深挖 | 本文 §2.2 → [ACADEMIC_THESIS.md](ACADEMIC_THESIS.md) | [SOTA_SURVEY.md](SOTA_SURVEY.md) | `scripts/h4_*` · `h5_*` |
| 工程复现 | [DUMP_READY.md](DUMP_READY.md) → [PROJECT_GUIDE.md](PROJECT_GUIDE.md) | STATUS | `train --from-existing` · `run_console.ps1` |

---

## 5. 明确「不是什么」（防叙事跑偏）

- ❌ 不是「堆了 LightGBM + GRU + TFT 所以先进」  
- ❌ 不是床位调度 / scheduling 联调项目  
- ❌ 不是无多中心验证的 SOTA 模型论文  
- ✅ 是：**同一总目标下的机制型 CDSS 研究工程**——数据合法 → 分数可靠 → 阈值有用 → 融合可证 → 解释可审

---

## 6. 结题故事线（3 分钟口述骨架）

1. **问题**：入科后数小时内要不要加强监护——稀有、要早、错响代价不对称。  
2. **总目标**：做出可决策的预警协同，而不是刷 AUC。  
3. **做法**：可复现 LGBM 骨干 + **H4 按净受益定阈值** + **H5 按完备度融合多时刻** + **H3 可追溯解释与人审**。  
4. **证据**：旧 S2 上 H4 ΔNB、H5 稀疏子集与 Brier；DCA 曲线；解释评测。  
5. **边界**：单中心；真序列待 Layer0；不宣称外部 SOTA。

---

## 7. 文档与代码索引（避免多 SSOT）

| 类型 | 唯一入口 | 说明 |
|------|----------|------|
| **总叙事** | **本文件** | 总目标 + 总内容串线 |
| **架构与公式** | [ARCHITECTURE_MATH.md](ARCHITECTURE_MATH.md) | 新旧对比 · H4/H5 数学 · 伪代码 |
| 学术假设细节 | [ACADEMIC_THESIS.md](ACADEMIC_THESIS.md) | 痛点/约束/H1–H5 定义与主表 |
| 指标与状态 | [STATUS.md](STATUS.md) | 当前数字 |
| 路线图 | [ROADMAP.md](ROADMAP.md) | 下一拍 |
| 工程结构 | [PLAN.md](../PLAN.md) · [PROJECT_GUIDE.md](PROJECT_GUIDE.md) | 分层与命令 |

**Agent / 组员约定**：叙述冲突时以 **本文件总目标** 为准；数字以 STATUS 为准；假设定义以 ACADEMIC_THESIS 为准。
