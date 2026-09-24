# 方向同步（组员必读）· 2026-09-24

> Owner：`TinjiYDon` · 已合入：PR [#18](https://github.com/TinjiYDon/icu-decision-agent/pull/18) → `main`  
> **创新点已正式采入**：[ROADMAP.md](ROADMAP.md) DX-1/2/3 · [SOTA_SURVEY.md](SOTA_SURVEY.md) H1–H3 落点  
> 请 `git pull origin main`；整合步骤见 [INTEGRATION_PREP.md](INTEGRATION_PREP.md)。

## 人读摘要

| 项 | 内容 |
|----|------|
| 定位 | **独立** ICU 早期恶化预警 + AIGC 人机协同 CDSS |
| 禁止 | 调用 scheduling 床位 API |
| 已采入 | **DX-1** 解释可信 · **DX-2** DCA · **DX-3** 公平双轨（主报 PR-AUC/Brier） |
| 本周请做 | pull → 按 INTEGRATION_PREP 跑 pytest；D3 报告看 PR-AUC 主表 |

## 命名对照

| 组员编号 | 正式波次 | 内容 |
|----------|----------|------|
| D1 | **DX-1** | 引用容差 / RAG18 / Prompt |
| D2 | **DX-2** | DCA + CI |
| D3 | **DX-3** | 公平双轨；**主指标 PR-AUC/Brier** |

## 验收命令

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_dca.py tests/test_d3_grud_compare.py tests/test_reference_validator.py -q
```

## 下一阶段

见 [ROADMAP.md](ROADMAP.md) · [TOP_TIER_NEXT.md](TOP_TIER_NEXT.md) · [INTEGRATION_PREP.md](INTEGRATION_PREP.md)

## Agent 上下文

```text
SSOT: docs/STATUS.md · docs/TEAM_DIRECTION.md · docs/ROADMAP.md
adopted: DX-1 DX-2 DX-3
禁区: dumps/ artifacts/ · 禁止接 scheduling
```
