# Progress · icu-decision-agent

> 更新：2026-08-14（执行推进）  
> 人读：本仓做到哪、卡在哪。

## Agent 上下文

```text
repo: icu-decision-agent
product: ICU early-warning CDSS at t=intime+h
primary_metrics: PR-AUC, Brier, operating_point
baseline_model: LightGBM + TreeSHAP
explain: SHAP+RAG+LLM merged (PR #9)
label: prefer admissions.deathtime (v2)
temporal: GRU-D smoke/skeleton (needs sequence dump to train)
```

## 里程碑

| 里程碑 | 状态 | 证据 |
|--------|------|------|
| S1 / S2 / 演示台 v4 | 完成 | PR #7/#8 · main |
| SHAP + RAG + LLM 解释链 | **已合入** | **PR #9 merged** · 监测台「解释」页 |
| `deathtime` 标签路径 | **代码完成** | `mortality_12h` prefer deathtime；需 Layer0 重跑 label+train |
| GRU-D 同 split 对照 | **骨架完成** | `domain/models/temporal` · `train_grud` smoke；缺真实序列 dump |

## 下一冲刺（剩余）

1. 在 Layer0 上 `build_labels` + `application.train`，打新 dump 版本号。  
2. 从 chart/labs 建真实 (T×F) 序列后训练 PyTorch GRU-D 并对照 LGBM。  
3. 滑动窗 `[t−L,t)` 接入特征构建（配置已在 `temporal.yaml`）。
