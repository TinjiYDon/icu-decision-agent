# Progress · icu-decision-agent

> 更新：2026-08-14  
> 人读：本仓做到哪、卡在哪。  
> AI：状态符号 ✅ 完成 · 进行中 · 规划；PR#9 未合并则不算交付。

## Agent 上下文

```text
repo: icu-decision-agent
product: ICU early-warning CDSS at t=intime+h
primary_metrics: PR-AUC, Brier, operating_point (ROC对照 only)
baseline_model: LightGBM + TreeSHAP
open_pr: #9 liujiawei SHAP+RAG+LLM
```

## 里程碑

| 里程碑 | 状态 | 证据 |
|--------|------|------|
| S1 早期预警 `intime+1h` | 完成 | PR #7 |
| S2 多时刻网格 | 完成 | PR #8 · STATUS PR-AUC≈0.14 |
| 演示台 v4（监测/调参/验收） | 完成 | main |
| SHAP + RAG + LLM 解释链 | 进行中 | **open PR #9**（未交付） |
| `deathtime` 精确标签 | 规划 | 现用日期级 `dod` |
| GRU-D 同 split 对照 | 规划 | 需序列 + mask + Δt 数据 |

## Issue / PR

- Issues #1–#5：全部关闭。
- Open PR：**#9** `liujiawei`。
- 风险：dump 中 chart vitals 多为 null；序列管线未齐 → 不可宣称 GRU-D 已训。

## 研究已定（工程未全落）

| 项 | 决定 |
|----|------|
| 可讲解交付 | LightGBM + TreeSHAP → RAG → LLM（主路径） |
| 时序对照 | GRU-D 与 LGBM **同 stay split**；Transformer 后议 |
| 时间窗 | 现扩展窗 `charttime < t`；后滑动窗 + 近端加权 |
| 与 scheduling | **本仓不向 scheduling 硬推风险分**（两仓独立交付） |

## 下一冲刺

见 [ROADMAP.md](ROADMAP.md)。指标细节见 [STATUS.md](STATUS.md)。
