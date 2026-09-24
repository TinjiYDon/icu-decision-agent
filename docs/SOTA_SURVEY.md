# SOTA / 文献与市面对标 · icu-decision-agent

> 更新：2026-09-13 · Wave **D-LIT**  
> **规则**：未完成本文档「创新假设」验收前，STATUS/答辩稿 **禁止** 宣称「首创 / SOTA」。  
> 创新点须能一句话回答：**相对谁、改了什么、指标如何更好**。

## 本仓基线（对标锚点）

| 能力 | 现状 |
|------|------|
| 表格预警 | LightGBM · S2 多时刻 · label v2=`deathtime` · 主报 PR-AUC / Brier |
| 时序对照 | PyTorch GRU-D（PR#11）· 同 split manifest · 稀疏门控 |
| 可解释 | TreeSHAP → RAG → LLM · Streamlit 解释页 |
| 人机协同 | HITL accept/reject/edit · 仓内 care_plan（非床位运筹） |

---

## 核心文献与系统条目（≥8）

| # | 条目 | 类型 | 与本仓关系 |
|---|------|------|------------|
| 1 | Che et al., **GRU-D**, Sci Rep 2018 | 不规则 EHR 时序 | GRU-D 对照轨直接对标 |
| 2 | Lim et al., **Temporal Fusion Transformer**, IJF 2021 | 多水平时序 | 远期消融候选，非主路径 |
| 3 | Lundberg & Lee, **SHAP**, NeurIPS 2017；TreeSHAP | 可解释 | 讲解主链因子来源 |
| 4 | Johnson et al., **MIMIC-IV**, Sci Data 2023 | 数据 | 本仓 Layer0/特征底座 |
| 5 | Harutyunyan et al., **Multitask ICU benchmarks**, Sci Data 2019 | 基准任务 | 死亡/失代偿等任务对照叙事 |
| 6 | Futoma / Sendak 等综述：ML clinical deployment pitfalls | 部署 | 强调校准、工作点、人机流程 |
| 7 | NEWS2 / MEWS 等早期预警评分 | 临床产品基线 | 规则评分 vs 学习模型对照 |
| 8 | Epic / Philips 等 CDSS 预警产品公开材料 | 市面 | 人机报警负荷、可解释要求 |
| 9 | RAG + LLM for clinical NLP（Lewis RAG 2020；医学安全综述） | AIGC | SHAP→RAG→LLM 链路定位 |
| 10 | Decision curve analysis（Vickers et al.） | 评价 | 验收台净受益 / 工作点 |

> 答辩引用时请补全规范 bibliographic 字段；上表为对标索引，非完整 BibTeX。

---

## 差距矩阵

| 轴 | 文献/市面常见做法 | 本仓 | 差距 | 拟切口 |
|----|-------------------|------|------|--------|
| 表格预警 | LGBM/XGB + 精细标签 + 外部验证 | LGBM S2 + deathtime v2；单 MIMIC split | 缺时间外推/多中心 | 工作点协议 + 决策曲线 |
| 不规则时序 | GRU-D / TFT 全量训练报告 | GRU-D 已通链路；稀疏序列仍痛 | 真序列覆盖与指标未稳 | 同 split 对照 + 稀疏门控 |
| 可解释 CDSS | SHAP 图或注意热力 | SHAP→RAG→LLM + 引用校验 | 市面少见「可追溯引用+降级」闭环 | HITL + 引用有效性 |
| 人机协同 | 报警抑制、确认流程 | HITL audit + care_plan | 反馈未系统回流改进 | 数据飞轮（见 DATA_FLYWHEEL） |

---

## 三条可验证创新假设（答辩绑定）

| ID | 假设 | 相对谁 | 改什么 | 验收指标 |
|----|------|--------|--------|----------|
| **H1** | 精确死亡时间标签（deathtime v2）+ **DCA/净受益** 优于「只刷 ROC/旧 dod」叙事 | 常用 `dod` + 单报 AUC | 标签 v2 + DCA 工作点 | Brier / **决策曲线**；不以刷高 PR-AUC 退回 dod |
| **H2** | **公平双轨协议**下可量化表格 vs 时序边界 | 随意比两个 AUC | 同 h / split / stay 对齐 + 稀疏门控 | test **PR-AUC / Brier** 主表；ROC 仅对照 |
| **H3** | SHAP→RAG→LLM **解释可信工程** + HITL 优于「仅 SHAP 条」 | 纯可视化 Demo | 分段容差、RAG18、详解 Prompt、audit | 引用 valid 率 + 周 **采纳率**（飞轮） |

### 已采入工程落点（2026-09-24）

| 假设 | 代码/文档落点 | 备注 |
|------|----------------|------|
| H1 | `domain/models/dca.py` · `scripts/d2_dca_full.py` | DX-2 已合 |
| H2 | `scripts/d3_grud_minimal_compare.py` | DX-3；**禁止**仅用 ROC 过线 |
| H3 | `reference_validator` / `rag_retriever` / `prompts` / `quality_eval` | DX-1 供给侧；KPI 待闭合 |

---

## 非宣称

- 未跑外部验证前不写「泛化 SOTA」。
- 不把 care_plan 写成床位 CP-SAT/PPO。
- 不与 `icu-scheduling-agent` 硬耦合作为创新点。

## 相关

- [ROADMAP.md](ROADMAP.md) · [DATA_FLYWHEEL.md](DATA_FLYWHEEL.md) · [STATUS.md](STATUS.md) · [INNOVATION_ROADMAP.md](INNOVATION_ROADMAP.md)
