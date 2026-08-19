# 文档索引

| 文档 | 内容 |
|------|------|
| [COLLABORATION.md](COLLABORATION.md) | **3 人协作主手册** |
| [BACKLOG.md](BACKLOG.md) | 垂直切片任务（可开 Issue） |
| [PROJECT_GUIDE.md](PROJECT_GUIDE.md) | 架构、流程、命令 |
| [DATA_LOCAL.md](DATA_LOCAL.md) | MIMIC / dump |
| [STATUS.md](STATUS.md) | 当前进度与指标 |
| [PROGRESS.md](PROGRESS.md) | 里程碑完成度（人/AI） |
| [CHANGELOG.md](CHANGELOG.md) | 本仓变更与 Release |
| [ROADMAP.md](ROADMAP.md) | 下一版本目标 |
| [DUMP_READY.md](DUMP_READY.md) | **线下 dump 单发 / restore**（含特征口径） |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | **答辩演示口播** |
| [TUNING_LOCAL.md](TUNING_LOCAL.md) | Streamlit 监测台启动 / 常见问题 |
| [TOP_TIER_NEXT.md](TOP_TIER_NEXT.md) | 下一步优化建议 |
| [INNOVATION_ROADMAP.md](INNOVATION_ROADMAP.md) | 里程碑 |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | PR 规范 |

## ADR

- [001-layer-boundaries.md](adr/001-layer-boundaries.md)

## 角色速查

| 成员 | 角色 |
|------|------|
| A | 数据/基础设施 |
| B | 算法 |
| C | 应用/集成负责人 |

## 数据检查点

```powershell
$env:PYTHONPATH = (Get-Location)
.\scripts\run_data_pipeline.ps1
```
