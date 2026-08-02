# 创新路线

> 本仓库独立演进；MCP 仅作为对外标准接口，不与其他 ICU 项目耦合。

## 目标（实时落地）

**ICU 早期恶化预警**：在预测时刻 `t`（S1：`intime+1h`）使用床旁可得 vitals/labs → LightGBM 风险分 → SHAP → 人机建议。

入科瞬间 6 特征弱基线仅作**对照**，不再作为产品主叙事。

## 里程碑

| 阶段 | 目标 | 交付物 |
|------|------|--------|
| **P0** ✓ | Demo 跑通 | ETL + LightGBM + dump |
| **P1** ✓ | 可解释 Demo | Streamlit + SHAP |
| **P2** ✓ 骨架 | 标准接口 | MCP `predict_risk` |
| **S1** | 实时早期预警切片 | `hour_index=1` + 扩展特征 + **多指标** |
| **S2** | 多时刻流式样本 | 同一 `prediction_time` API，h 网格 |
| **P3** | 时序升级 | GRU-D / TFT（未做） |

## 评测（投刊/答辩）

主报：**PR-AUC、Brier、工作点 Precision/Recall**；ROC-AUC 仅对照。禁止单报 ROC。

## 当前重点

1. S1 契约与 Layer0 重建训练（本机 5432 `mimic`）
2. 对照 Wave2 6-feat 基线写入 STATUS
3. S2：参数化 offset 已预留（`prediction_offset_hours`）
