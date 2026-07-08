# 文档索引

| 文档 | 内容 |
|------|------|
| [COLLABORATION.md](COLLABORATION.md) | **3 人协作主手册** |
| [BACKLOG.md](BACKLOG.md) | 垂直切片任务（可开 Issue） |
| [PROJECT_GUIDE.md](PROJECT_GUIDE.md) | 架构、流程、命令 |
| [DATA_LOCAL.md](DATA_LOCAL.md) | MIMIC / dump |
| [STATUS.md](STATUS.md) | 当前进度 |
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
