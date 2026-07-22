# 项目状态

> 更新：2026-07-22

## 数据检查点（已完成）

| 项 | 状态 |
|----|------|
| Layer0 `mimic` | ✓ P0 核心表 |
| ETL staging | ✓ 94,458 stays |
| dump | ✓ 曾导出；本地若仅 schemas_only 见 `PARAM_STORY.md` |
| 冒烟测试 | ✓ `run_data_pipeline.ps1` |

## 模型阶段

| 项 | 状态 |
|----|------|
| mortality_12h + LightGBM | `application.train`（B · Issue #3） |
| L4 `predict_patient` + `recommend` 档位 | ✅ C |
| Streamlit 选 stay + SHAP + 建议 | ✅ C |
| MCP Tool | P2 规划 |
| PPO / RL | ❌ 不做（方案 C） |

## 成员 C 本阶段交付

- `predict_patient` / SHAP 缓存 / recommend 档位
- `docs/PARAM_STORY.md` 参数故事
- Streamlit 风险建议展示

## 验证

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_predict.py tests/test_smoke.py -q
streamlit run presentation/streamlit_app.py
```
