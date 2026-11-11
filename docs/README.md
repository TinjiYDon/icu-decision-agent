# 文档索引（icu-decision-agent）

> 第一入口：[TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md) —— 意图与责任分离的技术总览（架构 / 创新点详解 / 调研 / 选型 / 实现路径 / 术语 / API / 索引）。
> 旧版/被替代文档在 `docs/archive/`（保存不删）。

## 1. 总览与叙事

| 文档 | 内容 |
|---|---|
| [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md) | **技术总览（主入口）** |
| [STAGE.md](STAGE.md) | 阶段门禁 S0–S4（当前 S4） |
| [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | **总目标 / 架构 / 下一步目标与验收（2026-11-11）** |
| [BRIEF.md](BRIEF.md) | 人话版解释（分诊台类比 + 五步链路） |
| [MASTER_NARRATIVE.md](MASTER_NARRATIVE.md) | 总目标 · 总内容（SSOT） |
| [STATUS.md](STATUS.md) | 进度与指标 |
| [ROADMAP.md](ROADMAP.md) / [CHANGELOG.md](CHANGELOG.md) | 计划 / 变更记录 |

## 2. 创新与调研

| 文档 | 内容 |
|---|---|
| [LARE.md](LARE.md) | **LARE / LAR-Policy / Hoeffding 上界 / MIMIC 运行器** |
| [INNOVATION_PORTFOLIO.md](INNOVATION_PORTFOLIO.md) | 创新点集合与可主张性 |
| [LABEL_AUDIT.md](LABEL_AUDIT.md) | P1 标签时间精度审计（证据底座） |
| [H9_SBR.md](H9_SBR.md) / [PAC_SAFETY_NET.md](PAC_SAFETY_NET.md) | 工程基线（不主张算法创新） |
| [SOTA_SURVEY.md](SOTA_SURVEY.md) / [ACADEMIC_THESIS.md](ACADEMIC_THESIS.md) | 对标 / 学术纲领 |
| [TEACHER_INNOVATION.md](TEACHER_INNOVATION.md) | 答辩口径 |

## 3. 架构与公式

| 文档 | 内容 |
|---|---|
| [ARCHITECTURE_MATH.md](ARCHITECTURE_MATH.md) | 新旧架构与公式 |
| [TECH_DESIGN_SHAP_LLM.md](TECH_DESIGN_SHAP_LLM.md) | 解释链设计 |
| [PARAM_STORY.md](PARAM_STORY.md) / [EXPLAIN_QUALITY_IMPROVEMENT_PLAN.md](EXPLAIN_QUALITY_IMPROVEMENT_PLAN.md) | 参数 / 解释质量改造 |

## 4. 实验与数据

| 文档 | 内容 |
|---|---|
| [DATA_LOCAL.md](DATA_LOCAL.md) / [DUMP_READY.md](DUMP_READY.md) | 数据约定 / dump 恢复 |
| [BASELINE_EXPERIMENT.md](BASELINE_EXPERIMENT.md) / [D1_OPTIMIZATION_SUMMARY.md](D1_OPTIMIZATION_SUMMARY.md) | 基线与解释评测 |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) / [TUNING_LOCAL.md](TUNING_LOCAL.md) | 演示与启动 |

## 5. 协作与工程

| 文档 | 内容 |
|---|---|
| [COLLABORATION.md](COLLABORATION.md) | 3 人协作主手册 |
| [PROJECT_GUIDE.md](PROJECT_GUIDE.md) | 架构、流程、命令 |
| [TEAM_DIRECTION.md](TEAM_DIRECTION.md) / [INTEGRATION_PREP.md](INTEGRATION_PREP.md) / [PROGRESS.md](PROGRESS.md) | 分工与里程碑 |
| [MERGE_GATE.md](MERGE_GATE.md) / [BUGBOT.md](BUGBOT.md) / [CONTRIBUTING.md](../CONTRIBUTING.md) | 工程规范 |
| ADR | [001-layer-boundaries.md](adr/001-layer-boundaries.md) |

## 6. 归档（保存不删）

`docs/archive/`：报告稿旧版、日期戳进度报告等被替代文档。

## 角色速查：A 数据/基础设施 · B 算法 · C 应用/集成负责人

## 数据检查点

```powershell
$env:PYTHONPATH = (Get-Location)
.\scripts\run_data_pipeline.ps1
```
