# Changelog · icu-decision-agent

> 更新：2026-08-14  
> 人读：本仓变更与 Release。权威进度见 [STATUS.md](STATUS.md)。  
> AI：仅本仓；勿引用其他仓库路径；Release tag `decision-s2-console`。

## Agent 上下文

```text
repo: icu-decision-agent
changelog_scope: this repo only
open_pr: https://github.com/TinjiYDon/icu-decision-agent/pull/9
release: decision-s2-console
```

## [Unreleased]

- **Merged** PR [#11](https://github.com/TinjiYDon/icu-decision-agent/pull/11)：PyTorch GRU-D + 归因热力图 + 知识库。
- 独立深挖叙事：AIGC 人机协同 CDSS；仓内 care_plan；不耦合 scheduling。
- **D-LIT / D-FLY**：[`SOTA_SURVEY.md`](SOTA_SURVEY.md)、[`DATA_FLYWHEEL.md`](DATA_FLYWHEEL.md)、`application.summarize_hitl`。
- D1：序列稀疏门控、同 split manifest、`train_grud --real`、`compare_dual_track`。
- D2–D3：解释页 HITL audit；Streamlit「规划」页。
- 标签路径 v2：优先 `admissions.deathtime`；dump 文档标明磁盘/库内版本差。

## 2026-08

- **Merged** PR [#9](https://github.com/TinjiYDon/icu-decision-agent/pull/9)：SHAP → RAG → LLM 解释页。
- **Merged** [#8](https://github.com/TinjiYDon/icu-decision-agent/pull/8) S2 多时刻；[#7](https://github.com/TinjiYDon/icu-decision-agent/pull/7) S1；[#6](https://github.com/TinjiYDon/icu-decision-agent/pull/6) 分层 baseline。
- GRU-D 研究骨架（numpy smoke）与 deathtime 重训指标入 STATUS。
- Release：[decision-s2-console](https://github.com/TinjiYDon/icu-decision-agent/releases/tag/decision-s2-console)。

## 更早

- Issues #1–#5 关闭（MCP / Streamlit / ETL dump 等骨架）。

## 相关文档

| 文档 | 用途 |
|------|------|
| [ROADMAP.md](ROADMAP.md) | 下一版本 |
| [STATUS.md](STATUS.md) | 指标、dump 与里程碑 |
