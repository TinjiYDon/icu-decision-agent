# 文档索引 · icu-decision-agent

| 文档 | 用途 |
|------|------|
| [PROJECT_GUIDE.md](PROJECT_GUIDE.md) | **主入口**：架构、命令、检查点、待办 |
| [DATA_LOCAL.md](DATA_LOCAL.md) | 本地 MIMIC / dump / ETL（不入 GitHub） |
| [STATUS.md](STATUS.md) | 当前进度 |
| [INNOVATION_ROADMAP.md](INNOVATION_ROADMAP.md) | 创新路线 P0–P4 |

## 数据检查点（当前阶段）

```powershell
$env:PYTHONPATH = (Get-Location)
.\scripts\run_data_pipeline.ps1
```

**停止线：** ETL + dump + 冒烟 ✓ · 模型训练 `run_p0` 为下一步。

## dump 命名

`{db}_P0-etl_{layer0}_{N}stays_{yyyymmdd}.dump`  
示例：`icu_decision_P0-etl_mimic_94458stays_20260708.dump`（网盘路径见 DATA_LOCAL.md）
