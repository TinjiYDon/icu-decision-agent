# ICU 临床恶化预警决策智能体

独立开源项目 · 3 人协作 · 仓库 [icu-decision-agent](https://github.com/TinjiYDon/icu-decision-agent)

**协作入口**：[`docs/COLLABORATION.md`](docs/COLLABORATION.md) · [`CONTRIBUTING.md`](CONTRIBUTING.md) · [`docs/BACKLOG.md`](docs/BACKLOG.md)

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
copy configs\database.yaml.example configs\database.yaml
copy configs\data.yaml.example configs\data.yaml
.\scripts\apply_migrations.ps1
$env:PYTHONPATH = (Get-Location)
.\scripts\run_data_pipeline.ps1   # 数据检查点：ETL + dump + 冒烟
```

## 文档

| 文档 | 说明 |
|------|------|
| [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) | 架构、流程、命令 |
| [docs/DATA_LOCAL.md](docs/DATA_LOCAL.md) | MIMIC / dump 本地配置 |
| [docs/STATUS.md](docs/STATUS.md) | 当前进度 |
| [docs/README.md](docs/README.md) | 文档索引 |

## 架构

```
MIMIC (Layer0) → ETL → staging/feat → LightGBM + SHAP → Streamlit
```

## Docker（可选）

```powershell
docker compose up -d   # PostgreSQL 端口 5433
```

## 模型阶段（数据检查点之后）

```powershell
python -m application.train
python -m application.run_p0
```
