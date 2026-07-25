# 项目状态

> 更新：2026-07-25 · Wave1

## 数据检查点（已完成）

| 项 | 状态 |
|----|------|
| Layer0 `mimic` | ✓ P0 核心表 |
| ETL staging | ✓ 94,458 stays |
| dump | ✓ 曾导出；本地若仅 schemas_only 见 `PARAM_STORY.md` |
| 冒烟测试 | ✓ `run_data_pipeline.ps1` |

## 模型 / 特征

| 项 | 状态 |
|----|------|
| 无泄漏 FEATURE_COLS | ✅ Wave1 |
| 真三集 0.7/0.1/0.2 stay_id | ✅ Wave1 |
| mortality_12h + LightGBM | ⏳ Wave2 · B 全量 train + AUC |
| L4 `predict_patient` + recommend | ✅ C |
| Streamlit + SHAP | ✅ C |
| MCP `predict_risk` | ✅ C 骨架 |
| PPO / RL | ❌ 不做 |
| Bugbot | ✅ 已开 |
| 路线图 | [`ROADMAP_EXEC.md`](ROADMAP_EXEC.md) |

## 成员 C 本阶段交付

- Wave1：特征消毒 + split + PARAM_STORY / ROADMAP_EXEC
- L4 / Streamlit / MCP 骨架

## 验证

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_features_leak.py tests/test_predict.py tests/test_mcp_predict.py -q
```
