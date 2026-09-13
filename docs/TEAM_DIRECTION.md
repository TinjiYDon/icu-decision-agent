# 方向同步（组员必读）· 2026-09-13

> Owner：`TinjiYDon` · 已合入：PR [#14](https://github.com/TinjiYDon/icu-decision-agent/pull/14) → `main`  
> 请 `git pull origin main` 后按本文调整本周工作；有异议在 Issue / 群内回复。

## 人读摘要

| 项 | 内容 |
|----|------|
| 定位 | **独立** ICU 早期恶化预警 + AIGC 人机协同 CDSS |
| 禁止 | 调用 / 依赖 `icu-scheduling-agent` 床位 API；用 risk 分驱动分床 |
| 主指标 | **PR-AUC / Brier / 工作点净受益**；ROC 仅对照 |
| 标签 | 坚持 `deathtime` v2；**不以刷高 PR-AUC 退回 dod** |
| 本周请做 | Review `main` 变更；按角色推进 H1–H3 / HITL 飞轮 KPI |

## 本周方向调整

1. 本仓按**独立成果**答辩，不再讲「三仓 MCP 协同系统」主叙事。  
2. 深挖：双轨对照（LGBM vs GRU-D）、HITL 采纳/驳回、仓内 `care_plan`（非床位）。  
3. 文献与诚实边界见 [`SOTA_SURVEY.md`](SOTA_SURVEY.md)；飞轮见 [`DATA_FLYWHEEL.md`](DATA_FLYWHEEL.md)。

## 已合入改动（请 Review）

- SOTA / 飞轮 / ROADMAP LIT 文档  
- HITL audit · `summarize_hitl` · Streamlit「规划」页  
- 工作点 **net benefit** 进入 metrics / `compare_dual_track`  
- GRU-D 同 split / 稀疏门控（此前 PR#11 已在主线）

## 验收命令

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_net_benefit.py tests/test_hitl_care_plan.py tests/test_summarize_hitl.py -q
.\.venv\Scripts\python.exe -m application.compare_dual_track
```

## Draft PR（暂不合，供知情）

| PR | 内容 | 建议 |
|----|------|------|
| [#12](https://github.com/TinjiYDon/icu-decision-agent/pull/12) Draft | 恢复 ADR-001：UI 去掉直连 SQL，改走 `application/ui_queries.py` | 有空 review 后可合；与主线不冲突 |

## Agent 上下文

```text
SSOT: docs/STATUS.md · docs/TEAM_DIRECTION.md · docs/SOTA_SURVEY.md
禁区: dumps/ artifacts/ · 禁止接 scheduling optimize_beds
Closes: direction sync 2026-09-13 (docs only follow-up on main)
```
