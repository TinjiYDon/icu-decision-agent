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
| MCP Tool `predict_risk` | ✅ C 骨架（`presentation/mcp_server.py`） |
| PPO / RL | ❌ 不做（方案 C） |
| Bugbot | ✅ 已开（2026-07-22 · Dashboard Enable 三仓） |

## 成员 C 本阶段交付

- `predict_patient` / SHAP 缓存 / recommend 档位
- `docs/PARAM_STORY.md` 参数故事
- Streamlit 风险建议展示
- MCP `predict_risk` 工具骨架（`presentation/mcp_tools.py` + `mcp_server.py`）

## 验证

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_predict.py tests/test_smoke.py tests/test_mcp_predict.py -q
streamlit run presentation/streamlit_app.py
# MCP（可选）
.\.venv\Scripts\pip.exe install "mcp>=1.0"
.\.venv\Scripts\python.exe -m presentation.mcp_server
```
