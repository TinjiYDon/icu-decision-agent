# 项目状态

> 更新：2026-07-26 · Owner 代备 Wave2 训练底座

## 数据检查点（已完成）

| 项 | 状态 |
|----|------|
| Layer0 `mimic` | ✓ icustays/patients/admissions（labevents 导入中） |
| ETL staging | ✓ 94,458 stays |
| dump | ✓ **新** `dumps/icu_decision_P0-full_mimic_94458stays_20260726.dump`（旧 20260708 可删） |
| 冒烟测试 | ✓ |

## 模型 / 特征

| 项 | 状态 |
|----|------|
| 无泄漏 FEATURE_COLS | ✅ Wave1 |
| 真三集 0.7/0.1/0.2 stay_id | ✅ Wave1 |
| mortality_12h + LightGBM | ✅ Owner 代训 2026-07-26 · 见指标 |
| L4 `predict_patient` + recommend | ✅ C |
| Streamlit + SHAP | ✅ C |
| MCP `predict_risk` | ✅ C 骨架 |
| PPO / RL | ❌ 不做（本仓） |
| Bugbot | ✅ 已开 |
| 路线图 | [`ROADMAP_EXEC.md`](ROADMAP_EXEC.md) |

## Wave2 指标（Owner 本机可复现）

| 指标 | 值 |
|------|-----|
| feat/label 行数 | 94,458 |
| 阳性数 / 阳性率 | 2,099 / ~2.22% |
| split n | train 66,121 · val 9,446 · test 18,891 |
| **auc_val** | **0.711** |
| **auc_test** | **0.682** |
| artifact | `artifacts/models/lgbm_mortality_12h.txt`（不入 Git） |

复现：`python -m application.train`（PYTHONPATH=.）

## 成员 C / Owner 本阶段交付

- Wave1：特征消毒 + split
- Wave2 底座：重建无泄漏 feat + 全量 label + LightGBM artifact + 20260726 dump
- B 仍可复核指标 / 写入自己的实验笔记

## 成员 C 本阶段交付

- Wave1：特征消毒 + split + PARAM_STORY / ROADMAP_EXEC
- L4 / Streamlit / MCP 骨架

## 验证

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_features_leak.py tests/test_predict.py tests/test_mcp_predict.py -q
```
