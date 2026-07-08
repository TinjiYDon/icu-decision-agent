# 创新路线

> 本仓库独立演进；MCP 仅作为对外标准接口，不与其他 ICU 项目耦合。

## 目标

**可解释临床恶化预警**：MIMIC 时序 ETL → LightGBM 风险分 → SHAP 归因 → 人机协同建议。

## 里程碑

| 阶段 | 目标 | 交付物 |
|------|------|--------|
| **P0** ✓ | Demo 跑通 | ETL + LightGBM + dump |
| **P1** | 可解释 Demo | Streamlit 单患者页 + SHAP 图 |
| **P2** | 标准接口 | MCP Tool `predict_risk(stay_id)` |
| **P3** | 时序升级 | GRU-D / TFT 与表格模型融合 |
| **P4** | 互操作（可选） | FHIR Observation 映射 |

## 扩展方向

- **MCP**：对外暴露 `predict_risk`，供任意 Agent 客户端调用
- **推理服务**：模型 artifact 容器化部署（生产阶段）

## 当前重点

1. Streamlit 单患者演示
2. MCP Tool schema（输入 stay_id，输出 score + shap_factors）
