# 项目状态

> 更新：2026-07-27 · dump 可训说明 · TUNING/MLflow

## 数据检查点

| 项 | 状态 |
|----|------|
| Layer0 `mimic` | ✅ icustays/patients/admissions + **labevents 1.58e8**（Layer0，不在 dump 内） |
| ETL staging | ✅ 94,458 |
| **dump（训练用）** | ✅ `dumps/icu_decision_P0-full_mimic_94458stays_20260726.dump` · `schemas_only=false` · 含 feat/label |
| 冒烟 | ✅ |
| 交付说明 | [`DUMP_READY.md`](DUMP_READY.md) |

## 模型 / 特征

| 项 | 状态 |
|----|------|
| 无泄漏 FEATURE_COLS | ✅ |
| 真三集 0.7/0.1/0.2 | ✅ |
| LightGBM | ✅ Owner 代训 |
| L4 / Streamlit / MCP | ✅ |
| 调参台 | [`TUNING_LOCAL.md`](TUNING_LOCAL.md) · MLflow `sqlite:///./mlflow.db` |
| PPO | ❌ 本仓不做 |

## Wave2 指标

| 指标 | 值 |
|------|-----|
| feat/label | 94,458 |
| 阳性 | 2,099（~2.22%） |
| **auc_val** | **0.711** |
| **auc_test** | **0.682** |
| artifact | `artifacts/models/lgbm_mortality_12h.txt`（不入 Git） |

**结论：当前 dump + 本机 DB 已可支撑监督训练/复核；dump 须线下分发。**

## GitHub PR（2026-07-27 查）

- 无 open PR；Issue #3/#4 仍开（B/A 复核）

## 验证

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_features_leak.py tests/test_predict.py -q
```
