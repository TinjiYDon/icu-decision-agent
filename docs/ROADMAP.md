# Roadmap · icu-decision-agent

> 更新：2026-08-14  
> 人读：下一版本只做本仓事项。  
> AI：P0 先于 P1；非目标勿实现、勿在 STATUS 宣称。

## Agent 上下文

```text
repo: icu-decision-agent
vnext_p0: merge_or_split PR#9; deathtime labels + new dump
vnext_p1: GRU-D data pipeline; sliding window
vnext_p2: optional Transformer ablation
non_goals: replace TreeSHAP path with GRU-D-only; claim clinical production
```

## 原则

1. 决策价值指标（PR-AUC / Brier / 工作点）优先于单报 ROC。
2. 时间因果：特征 `charttime < t`；标签窗口自预测时刻起。
3. 可讲解主路径保持表格模型 + TreeSHAP；深度时序作对照。

## vNext

| 优先级 | 项 | 说明 |
|--------|----|------|
| P0 | 收敛 PR #9 | SHAP+RAG+LLM 合入或拆分；LGBM 讲解链不断 |
| P0 | `deathtime` 标签 | 替换日期级 `dod`；新 dump 版本号 |
| P1 | GRU-D 数据前置 | 序列 + mask + Δt；与 LGBM 同 split 对照 |
| P1 | 滑动窗演进 | `[t−L, t)` + 近端加权 |
| P2 | Transformer 变体 | 可选消融，非主叙事 |

## 非目标

- 用 GRU-D 砍掉 TreeSHAP → RAG → LLM 讲解链。
- 宣称模型已临床部署上线。

## 相关

- 变更史：[CHANGELOG.md](CHANGELOG.md)
- 现状：[PROGRESS.md](PROGRESS.md) · [STATUS.md](STATUS.md)
- 方法建议：[TOP_TIER_NEXT.md](TOP_TIER_NEXT.md)
