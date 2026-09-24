# Roadmap · icu-decision-agent

> 更新：2026-09-24（采入组员创新点 + 指标纠偏）  
> 人读：本仓独立 CDSS；决策价值指标优先；不与 scheduling 硬耦合。  
> AI：P0 先于 P1；无 LIT 不宣称 SOTA；D3 主报 **PR-AUC / Brier**（ROC 仅对照）。

## Agent 上下文

```text
repo: icu-decision-agent
adopted: D1-explain-quality; D2-DCA; D3-fair-dual-track-protocol
vnext_p0: D3 metrics on STATUS; utf-8; restore dump; pytest green
vnext_p1: H3 HITL KPI via summarize_hitl; DCA in demo script
vnext_p2: care_plan polish; ADR-001 #12 optional
non_goals: replace TreeSHAP→RAG→LLM; call scheduling optimize
```

## 原则

1. **独立项目**：预警 + AIGC 协同 + 仓内 care_plan；不驱动床位引擎。  
2. **先对标、再创新**：见 [SOTA_SURVEY.md](SOTA_SURVEY.md)。  
3. **决策价值优先**：PR-AUC / Brier / DCA·净受益；ROC 仅对照。  
4. **时间因果**：特征 `charttime < t`。  
5. **讲解主链**：TreeSHAP → RAG → LLM；GRU-D 为公平对照轨。

## 已采入创新点（正式波次）

| 波次 | 创新点 | 状态 | 绑定假设 |
|------|--------|------|----------|
| **DX-1** | 解释可信工程（分段容差 / RAG18 / Prompt 详解） | ✅ PR#18 | H3 供给侧 |
| **DX-2** | 决策曲线 DCA + Bootstrap CI | ✅ PR#18 | H1 决策价值 |
| **DX-3** | 公平双轨协议（同 h / split / stay 对齐） | ✅ PR#18 | H2 协议 |
| D0 | GRU-D 链路 | ✅ PR#11 | — |

## 下一阶段计划

| 优先级 | 项 | 说明 |
|--------|----|------|
| **P0** | D3 主表纠偏 | STATUS / 报告以 **PR-AUC、Brier** 为准；ROC 脚注 |
| **P0** | 整合验收 | restore dump · `pytest` 全绿（含 utf-8） |
| **P1** | H3 闭合 | `summarize_hitl` 写入采纳/驳回/引用 valid 率 |
| **P1** | Demo 口述 | DCA 曲线进验收脚本 |
| **P2** | care_plan 打磨 | 非床位结构化建议 |
| **P2** | ADR-001 #12 | UI 分层 Draft，有空再合 |

## 非目标

- 用 GRU-D 砍掉 TreeSHAP→RAG→LLM。  
- 调用 scheduling `optimize_beds`。  
- 以 ROC「不劣于」作为唯一答辩卖点。  
- 无 LIT 宣称首创/SOTA。

## 相关

- [TEAM_DIRECTION.md](TEAM_DIRECTION.md) · [SOTA_SURVEY.md](SOTA_SURVEY.md) · [INTEGRATION_PREP.md](INTEGRATION_PREP.md)  
- [STATUS.md](STATUS.md) · [DATA_FLYWHEEL.md](DATA_FLYWHEEL.md) · [TOP_TIER_NEXT.md](TOP_TIER_NEXT.md)
