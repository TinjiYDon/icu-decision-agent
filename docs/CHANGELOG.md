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

- **Merged** PR [#9](https://github.com/TinjiYDon/icu-decision-agent/pull/9)：SHAP → RAG → LLM 解释页。
- 标签路径 v2：优先 `admissions.deathtime`；**已在 Layer0 重算 label 并 `train --from-existing`**（test PR-AUC≈0.092）。
- GRU-D 研究骨架：`domain/models/temporal` + `application.train_grud` smoke。

## 2026-08

- **Merged** [#8](https://github.com/TinjiYDon/icu-decision-agent/pull/8)：S2 多时刻 `h∈{0,1,2,4,6}`，约 472k 行，stay 级同折。
- **Merged** [#7](https://github.com/TinjiYDon/icu-decision-agent/pull/7)：S1 早期预警 `t=intime+1h` + 多指标 STATUS。
- **Merged** [#6](https://github.com/TinjiYDon/icu-decision-agent/pull/6)：分层 baseline（PR-AUC / Brier）。
- main：Plotly 监测台 v4、验收净受益、S2 dump 说明、MCP `predict_risk(hour_index)`。
- Release：[decision-s2-console](https://github.com/TinjiYDon/icu-decision-agent/releases/tag/decision-s2-console)。

## 更早

- Issues #1–#5 关闭（MCP / Streamlit / ETL dump 等骨架）。

## 相关文档

| 文档 | 用途 |
|------|------|
| [PROGRESS.md](PROGRESS.md) | 里程碑完成度 |
| [ROADMAP.md](ROADMAP.md) | 下一版本 |
| [STATUS.md](STATUS.md) | 指标与 dump |
