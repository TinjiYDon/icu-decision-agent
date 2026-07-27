# 创新路线

> 本仓库独立演进；MCP 仅作为对外标准接口，不与其他 ICU 项目耦合。

## 目标

**可解释临床恶化预警**：MIMIC 时序 ETL → LightGBM 风险分 → SHAP 归因 → 人机协同建议。

## 里程碑

| 阶段 | 目标 | 交付物 |
|------|------|--------|
| **P0** ✓ | Demo 跑通 | ETL + LightGBM + dump |
| **P1** ✓ | 可解释 Demo | Streamlit + SHAP |
| **P2** ✓ 骨架 | 标准接口 | MCP `predict_risk` |
| **P3** | 时序升级 | GRU-D / TFT（未做） |
| **P4** | 互操作（可选） | FHIR |

## 当前重点（2026-07-27）

1. ✅ Wave1 无泄漏特征 + 真三集；Owner 全量训 auc_val≈0.711 / auc_test≈0.682
2. ✅ **P0-full dump 可训**：见 [`DUMP_READY.md`](DUMP_READY.md)（线下分发）
3. ✅ 本地调参：Streamlit + MLflow · [`TUNING_LOCAL.md`](TUNING_LOCAL.md)
4. B：复核 AUC / 实验笔记（#3）；A：restore 验证（#4）
5. **不做** online PPO；时序见 P3

## 扩展方向

- MCP / 推理服务容器化（生产阶段）
