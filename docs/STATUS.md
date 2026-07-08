# 项目二 · 状态摘要

> 最后更新：2026-07-08

## 数据检查点（已完成）

| 项 | 状态 |
|----|------|
| Layer0 `mimic` 库 | ✓ P0 导入，94,458 icustays |
| ETL staging | ✓ 94,458 |
| dump 生产 | ✓ `icu_decision_P0-etl_mimic_94458stays_20260708.dump` |
| 连通性 + 冒烟 | ✓ `run_data_pipeline.ps1` |

## 模型阶段（下一步）

| 项 | 状态 |
|----|------|
| mortality_12h + LightGBM 全量 | 待跑 `application.train` |
| Streamlit | 占位 |
| MCP Tool | P2 规划 |

## 命令

```powershell
$env:PYTHONPATH = (Get-Location)
.\scripts\run_data_pipeline.ps1    # 数据阶段
python -m application.train        # 模型阶段
```
