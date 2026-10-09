# Roadmap · icu-decision-agent

> 更新：2026-10-09（采入 H4/H5 机制创新 + 旧 S2 真跑）  
> 人读：本仓独立 CDSS；决策价值指标优先；不与 scheduling 硬耦合。  
> AI：P0 先于 P1；无 LIT 不宣称 SOTA；D3 主报 **PR-AUC / Brier**（ROC 仅对照）。

## Agent 上下文

```text
repo: icu-decision-agent
adopted: DX-1; DX-2; DX-3; H4-UCEW; H5-SGDF
vnext_p0: Layer0 mimic for true temporal track
vnext_p1: v2 dump main table; care_plan polish
non_goals: replace TreeSHAP→RAG→LLM; call scheduling optimize; claim SOTA net
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
| **H4** | 效用校准工作点 UCEW（约束 max-NB） | ✅ 旧 S2 真跑 | H4 |
| **H5** | 稀疏门控多时刻融合 SGDF | ✅ 旧 S2（轨迹代理） | H5 |
| D0 | GRU-D 链路 | ✅ PR#11 | — |

## 下一阶段计划

| 优先级 | 项 | 说明 |
|--------|----|------|
| **P0** | Layer0 `mimic` | 真序列替换 H5 轨迹代理；因果窗三方重跑 |
| **P1** | v2 dump 主表 | 与旧 S2 机制表分栏，勿混 |
| **P1** | Demo | ✅ DCA + [TEACHER_INNOVATION.md](TEACHER_INNOVATION.md) H4/H5 |
| **P2** | care_plan 打磨 | 非床位结构化建议 |

## 非目标

- 用 GRU-D 砍掉 TreeSHAP→RAG→LLM。  
- 调用 scheduling `optimize_beds`。  
- 以 ROC「不劣于」作为唯一答辩卖点。  
- 无 LIT 宣称首创/SOTA。

## 相关

- **[MASTER_NARRATIVE.md](MASTER_NARRATIVE.md)**（总目标/总内容）· [TEAM_DIRECTION.md](TEAM_DIRECTION.md) · [SOTA_SURVEY.md](SOTA_SURVEY.md) · [INTEGRATION_PREP.md](INTEGRATION_PREP.md)  
- [STATUS.md](STATUS.md) · [DATA_FLYWHEEL.md](DATA_FLYWHEEL.md) · [TOP_TIER_NEXT.md](TOP_TIER_NEXT.md)
