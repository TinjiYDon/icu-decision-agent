# 项目状态

> 更新：2026-07-13

## 数据检查点（已完成）

| 项 | 状态 |
|----|------|
| Layer0 `mimic` | ✓ P0 核心表 |
| ETL staging | ✓ 94,458 stays |
| dump | ✓ `dumps/icu_decision_P0-etl_mimic_94458stays_20260708.dump` |
| 冒烟测试 | ✓ `run_data_pipeline.ps1` |

## 模型阶段

| 项 | 状态 |
|----|------|
| mortality_12h + LightGBM | `application.train`（B 主责跑通 + 写 registry） |
| L4 `predict_patient` | ✅ C 已完成 |
| Streamlit 选 stay + SHAP | ✅ C 已完成 |
| MCP Tool | P2 规划 |

## 成员 C 本阶段交付

- `application/predict_patient.py`（含 `get_label_config`、缓存 `list_stays`）
- `presentation/streamlit_app.py`（概率/raw 分数展示）
- `domain/models/lgbm.py` · `predict_stay` + SHAP 缓存（与 B 联调）

## 验证

```powershell
$env:PYTHONPATH = (Get-Location)
pytest tests/test_predict.py tests/test_smoke.py -q
streamlit run presentation/streamlit_app.py
```
