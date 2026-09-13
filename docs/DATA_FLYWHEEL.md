# 数据飞轮 · icu-decision-agent

> 更新：2026-09-13 · Wave **D-FLY**  
> **独立闭环**：本仓 HITL / 解释质量 → 改进知识或阈值 → 再测量。  
> **禁止**：跨仓写 scheduling；**禁止**自动把驳回当金标全量重训。

## 闭环图

```text
解释页生成 → 用户 accept/reject/edit → artifacts/hitl/explain_audit.jsonl
        → 周汇总 KPI → ① RAG 知识修订 ② recommend 阈值 ③ 难例人工标注
        → 再部署 / 再测采纳率
```

## 数据源

| 源 | 路径 | 入 Git？ |
|----|------|----------|
| HITL 审计 | `artifacts/hitl/explain_audit.jsonl` | **否** |
| 周汇总 | `artifacts/hitl/summary_*.json` | **否** |

## 回流优先级

1. **修订 RAG 知识块**（事实错误、过时指南）——最高优先级、最低风险  
2. **调 `configs/labels.yaml` recommend 阈值**（报警负荷）——需记录前后工作点  
3. **难例进人工标注集**——仅人工审核后进入训练/评测集  
4. **不自动**用 reject 翻转标签重训 LightGBM/GRU-D

## KPI（写入 STATUS「飞轮」节）

| KPI | 定义 |
|-----|------|
| 周解释次数 | 汇总窗口内 audit 条数 |
| 采纳率 | accept / (accept+reject+edit) |
| 驳回率 | reject / total |
| 编辑率 | edit / total |

降级率：若解释结果带 `fallback` 元数据可后续扩展；当前以 HITL 决策为主。

## 命令

```powershell
cd d:\project\icu-decision-agent
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m application.summarize_hitl
.\.venv\Scripts\python.exe -m application.summarize_hitl --days 7 --out artifacts/hitl/summary_latest.json
```

## 相关

- [SOTA_SURVEY.md](SOTA_SURVEY.md) 假设 H3 · [hitl_audit.py](../domain/explain/hitl_audit.py)
