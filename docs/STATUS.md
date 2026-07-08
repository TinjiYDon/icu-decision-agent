# 项目状态

> 更新：2026-07-08

## 数据检查点（已完成）

| 项 | 状态 |
|----|------|
| Layer0 `mimic` | ✓ P0 核心表 |
| ETL staging | ✓ 94,458 stays |
| dump | ✓ `dumps/icu_decision_P0-etl_mimic_94458stays_20260708.dump` |
| 冒烟测试 | ✓ `run_data_pipeline.ps1` |

## 模型阶段（下一步）

| 项 | 状态 |
|----|------|
| mortality_12h + LightGBM | 待 `application.train` |
| Streamlit | 占位 |
| MCP Tool | P2 规划 |

```powershell
$env:PYTHONPATH = (Get-Location)
.\scripts\run_data_pipeline.ps1
python -m application.train
```
