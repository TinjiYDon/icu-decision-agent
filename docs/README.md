# 文档索引（icu-decision-agent）

> **当前唯一主入口（2026-11-21 阶段 0 更新）**：  
> 1. **[OPTIMIZATION_SPEC.md](OPTIMIZATION_SPEC.md)** —— V3 优化技术规范（现状痛点 + 导师三问公式与算法设计）  
> 2. **[ACTION_CHECKLIST.md](ACTION_CHECKLIST.md)** —— 阶段 0–5 全流程执行行动清单与分工门控  
> 3. **[METRICS_LEDGER.md](METRICS_LEDGER.md)** —— 唯一实验指标台账（区分已作废口径与 V3 正式口径）  
>
> 被替代的旧版文档（旧 H4/H5/H6 CUGEW 及 7–10 月历史稿）均已无损归档至 `docs/archive/`（共 30 份），**请勿再将其作为当前设计依据**。

---

## 1. 当前核心规范（SSOT · 必读）

| 文档 | 内容 |
| :--- | :--- |
| [OPTIMIZATION_SPEC.md](OPTIMIZATION_SPEC.md) | **V3 项目优化说明（三大支柱与完整数学公式）** |
| [ACTION_CHECKLIST.md](ACTION_CHECKLIST.md) | **阶段 0–5 行动清单、三人分工与 Gate 验收** |
| [METRICS_LEDGER.md](METRICS_LEDGER.md) | **唯一指标台账（四套历史口径对照 + V3 槽位）** |
| [STAGE.md](STAGE.md) | 阶段门禁与变更日志 |
| [MASTER_NARRATIVE.md](MASTER_NARRATIVE.md) | 一页总纲领与答辩故事线（供 UI 与速览引用） |
| [adr/001-layer-boundaries.md](adr/001-layer-boundaries.md) | 分层架构硬边界（ADR-001） |

## 2. 现有 LARE / 标签 / 基线技术参考（11 月基线，持续沿用与扩展）

| 文档 | 内容 |
| :--- | :--- |
| [LARE.md](LARE.md) | LARE / LAR-Policy / MIMIC 运行器细节（`preflight.yaml` 检查项） |
| [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md) | 11 月初技术总览与接口索引（`preflight.yaml` 检查项） |
| [BRIEF.md](BRIEF.md) | 面向非专业读者的分诊台类比说明（`preflight.yaml` 检查项） |
| [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | 2026-11-11 中期阶段研究计划（D0–D9 与 `double` 退化分析） |
| [LABEL_AUDIT.md](LABEL_AUDIT.md) | 死亡标签时间精度审计底座 |
| [STATUS.md](STATUS.md) | 历史状态快照（最新数字以 [METRICS_LEDGER.md](METRICS_LEDGER.md) 为准） |
| [H9_SBR.md](H9_SBR.md) / [PAC_SAFETY_NET.md](PAC_SAFETY_NET.md) | 选择性复核与 PAC 安全网工程基线参考 |

## 3. 解释链、部署与工程协作指南

| 文档 | 内容 |
| :--- | :--- |
| [TECH_DESIGN_SHAP_LLM.md](TECH_DESIGN_SHAP_LLM.md) | SHAP→RAG→LLM 解释链与引用校验设计 |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | Streamlit 演示脚本参考 |
| [TUNING_LOCAL.md](TUNING_LOCAL.md) | 本地控制台启动与常见问题 |
| [DATA_LOCAL.md](DATA_LOCAL.md) / [DUMP_READY.md](DUMP_READY.md) | MIMIC 数据约定与 dump 恢复 |
| [PROJECT_GUIDE.md](PROJECT_GUIDE.md) / [COLLABORATION.md](COLLABORATION.md) | 工程结构、命令与 3 人协作手册 |
| [BUGBOT.md](BUGBOT.md) / [CHANGELOG.md](CHANGELOG.md) / [CONTRIBUTING.md](../CONTRIBUTING.md) | 质量门禁、变更记录与 PR 规范 |

## 4. 历史归档（`docs/archive/` · 保存不删，勿作当前依据）

包含已淘汰的 **H4/H5/H6 (CUGEW)** 文档（`ARCHITECTURE_MATH.md`、`ACADEMIC_THESIS.md`、`TEACHER_INNOVATION.md`、`SOTA_SURVEY.md` 等）、7–9 月进度/计划文档以及早期汇报讲稿，共 30 份。
