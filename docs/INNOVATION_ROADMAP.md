# 创新路线

> 本仓库**独立演进、独立出成果**；MCP 仅作本仓对外接口，不与 scheduling 硬耦合。

## 目标（产品叙事）

**基于 AIGC 的智慧人机协同临床决策支持（CDSS）**：

1. 感知：床旁可得 vitals/labs → LightGBM 12h 死亡风险（GRU-D 同 split 对照）
2. 解释：TreeSHAP → RAG → LLM
3. 协同：人确认 / 驳回 / 编辑（HITL 审计）
4. 规划：仓内结构化处置建议（监护升级、复查时点等）——**不是** ICU 床位运筹

入科瞬间 6 特征弱基线仅作对照，不再作为产品主叙事。

## 里程碑

| 阶段 | 目标 | 交付物 |
|------|------|--------|
| **P0** ✓ | Demo 跑通 | ETL + LightGBM + dump |
| **P1** ✓ | 可解释 Demo | Streamlit + SHAP |
| **P2** ✓ | 标准接口 | MCP `predict_risk` |
| **S1** ✓ | 实时早期预警切片 | `hour_index=1` + 扩展特征 + 多指标 |
| **S2** ✓ | 多时刻流式样本 | `prediction_hours=[0,1,2,4,6]` · 472k 行 |
| **S2b** ✓ | 标签精度 | `deathtime` v2 + 重训 |
| **P3** | 时序对照 | GRU-D 真训 + 同 split 指标 |
| **P4** | AIGC 闭环 | HITL audit + care_plan |

## 评测（投刊/答辩）

主报：**PR-AUC、Brier、工作点 Precision/Recall**；ROC-AUC 仅对照。禁止单报 ROC。

## 当前重点

1. D0：收敛 GRU-D（PR#11）+ v2 dump 文档
2. D1：真序列 ETL 与双轨对照
3. D2–D3：人机协同与仓内规划建议
