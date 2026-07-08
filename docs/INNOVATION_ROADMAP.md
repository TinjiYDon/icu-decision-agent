# 创新路线 — ICU 决策智能体

> 与调度、多模态项目**独立仓库、独立数据库**；后期通过 MCP 标准接口扩展，无需改库结构。

## 总叙事

**可解释临床恶化预警**：MIMIC 时序 ETL → LightGBM 风险分 → SHAP 归因 → 人机协同建议。

## 分步里程碑（可独立交付、可答辩）

| 阶段 | 目标 | 交付物 | 依赖 |
|------|------|--------|------|
| **P0** ✓ | Demo 跑通 | ETL + LightGBM + dump | mimic_iv_demo |
| **P1** | 可解释 Demo | Streamlit 单患者页 + Top-K SHAP 图 | P0 |
| **P2** | 标准接口 | MCP Tool `predict_risk(stay_id)` 返回 JSON | P1 |
| **P3** | 时序升级 | GRU-D 表征 + 与 LightGBM 融合 | Waveform P1 数据 |
| **P4** | 互操作（可选） | FHIR Observation 映射层 | 医院数据源 |

## 后期扩展空间

- **MCP 层**：与 GitHub 2025–2026 医疗 Agent（MCP-FHIR）趋势对齐，调度侧可**调用**本工具而不读 `icu_decision` 库
- **CDN**：模型 artifact（`artifacts/*.pkl`）可走对象存储 + CDN 分发推理服务（生产阶段）
- **与调度协同**：P2 完成后，调度项目将风险分作为**软约束输入**（事件/MCP，非 DB 耦合）

## 当前 Next

1. 完成 Streamlit 单患者演示
2. 设计 MCP Tool schema（输入 stay_id，输出 score + shap_factors + recommendation）

## 关联仓库

- 调度：[`icu-scheduling-agent`](https://github.com/TinjiYDon/icu-scheduling-agent) — `optimize_beds` MCP（P2 并行）
- 多模态：[`zhixue-multimodal`](https://github.com/TinjiYDon/zhixue-multimodal) — 教育域 RAG，无代码依赖
