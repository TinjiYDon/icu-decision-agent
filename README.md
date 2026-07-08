# ICU 临床恶化预警决策智能体

仓库：[icu-decision-agent](https://github.com/TinjiYDon/icu-decision-agent) · 数据库：`icu_decision`

## 文档

**从 [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md) 开始。**

| 文档 | 说明 |
|------|------|
| [docs/README.md](docs/README.md) | 文档索引 |
| [docs/DATA_LOCAL.md](docs/DATA_LOCAL.md) | dump / MIMIC 本地约定 |
| [docs/STATUS.md](docs/STATUS.md) | 当前进度 |

## 当前阶段

**数据检查点已完成**（ETL 94,458 stays + dump + 冒烟）。模型训练为下一步。

```powershell
$env:PYTHONPATH = (Get-Location)
.\scripts\run_data_pipeline.ps1   # 重现数据检查点
python -m application.train       # 模型阶段
```

## 架构

```
Layer0 mimic → ETL → staging/feat → LightGBM + SHAP → Streamlit
L5 → L4 application → L3 domain → L2 data_access → L1 infra → PostgreSQL
```

## Docker（队友）

```powershell
docker compose up -d   # PG 端口 5433
```
