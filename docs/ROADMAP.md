# Roadmap · icu-decision-agent

> 更新：2026-09-13  
> 人读：本仓独立出成果——AIGC 人机协同 CDSS；不与 scheduling 硬耦合。  
> AI：P0 先于 P1；非目标勿实现、勿在 STATUS 宣称。

## Agent 上下文

```text
repo: icu-decision-agent
vnext_p0: D-LIT SOTA_SURVEY; keep independent from scheduling
vnext_p1: dual-track LGBM vs GRU-D metrics; DATA_FLYWHEEL KPI
vnext_p2: HITL flywheel summarize; care_plan polish
non_goals: replace TreeSHAP→RAG→LLM; call scheduling optimize; claim SOTA without LIT
independent: zero hard couple to icu-scheduling-agent
```

## 原则

1. **独立项目**：本仓交付预警 + AIGC 协同决策 + 仓内处置规划建议；不驱动床位运筹引擎。
2. **先对标、再创新**：见 [SOTA_SURVEY.md](SOTA_SURVEY.md)；无 LIT 不宣称首创/SOTA。
3. 决策价值指标（PR-AUC / Brier / 工作点）优先于单报 ROC。
4. 时间因果：特征 `charttime < t`；标签窗口自预测时刻起。
5. 可讲解主路径保持表格模型 + TreeSHAP → RAG → LLM；深度时序作对照。

## vNext（Wave）

| 波次 | 项 | 说明 |
|------|----|------|
| **D-LIT** | 文献/市面对标 | [SOTA_SURVEY.md](SOTA_SURVEY.md) · 假设 H1–H3 |
| D0 | PR#11 GRU-D / 热力图 | **已合 main**；label v2 dump 文档对齐 |
| D1 | 真序列 + 双轨对照 | Layer0→序列 ETL；同 split；主报 PR-AUC/Brier |
| D2 | AIGC 人机闭环 | 采纳/驳回/编辑 + audit log |
| D3 | 仓内 care_plan | 结构化处置建议（非床位） |
| **D-FLY** | 数据飞轮 | [DATA_FLYWHEEL.md](DATA_FLYWHEEL.md) · HITL 汇总 KPI |
## 非目标

- 用 GRU-D 砍掉 TreeSHAP → RAG → LLM 讲解链。
- 调用 `icu-scheduling-agent` 的 `optimize_beds` / 写入 `sched.assignments`。
- 本仓实现床位 PPO。
- 宣称模型已临床部署上线。
- **无** [SOTA_SURVEY.md](SOTA_SURVEY.md) 就宣称首创/SOTA。

## 相关

- 变更史：[CHANGELOG.md](CHANGELOG.md)
- 现状：[PROGRESS.md](PROGRESS.md) · [STATUS.md](STATUS.md)
- 对标：[SOTA_SURVEY.md](SOTA_SURVEY.md) · 飞轮：[DATA_FLYWHEEL.md](DATA_FLYWHEEL.md)
- 方法建议：[TOP_TIER_NEXT.md](TOP_TIER_NEXT.md) · [INNOVATION_ROADMAP.md](INNOVATION_ROADMAP.md)
