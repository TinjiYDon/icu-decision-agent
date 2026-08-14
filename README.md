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
| [docs/DUMP_READY.md](docs/DUMP_READY.md) | **线下 dump 单发** |
| [docs/TUNING_LOCAL.md](docs/TUNING_LOCAL.md) | Plotly 监测台 |
| [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | **答辩演示口播** |
| [docs/STATUS.md](docs/STATUS.md) | 当前进度与指标 |
| [docs/PROGRESS.md](docs/PROGRESS.md) | 里程碑完成度 |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | 本仓变更 / Release |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 下一版本 |
| [docs/README.md](docs/README.md) | 文档索引 |

**进行中**：PR [#9](https://github.com/TinjiYDon/icu-decision-agent/pull/9) SHAP+RAG+LLM（未计入已交付）。

## 答辩演示

```powershell
.\scripts\run_console.ps1
```

口播步骤见 [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md)：项目总览 → 监测高低风险 → 验收校准/净受益。

演示前请 restore S2 dump（见 [`docs/DUMP_READY.md`](docs/DUMP_READY.md)）。**勿**在 restore 后再跑会清空 `feat` 的 P0 ETL。监测台主信号为年龄+化验；chart 生命体征在 dump 中多为缺测。

## 架构

```
MIMIC (Layer0) → ETL → staging/feat → LightGBM + SHAP → Streamlit
```

## Docker（可选）

```powershell
docker compose up -d   # PostgreSQL 端口 5433
```

## 模型与演示

```powershell
$env:PYTHONPATH = (Get-Location)
# 有 S2 dump 时（推荐）:
.\.venv\Scripts\python.exe -m application.train --from-existing
.\scripts\run_console.ps1
```

线下 dump：见 [`docs/DUMP_READY.md`](docs/DUMP_READY.md)（**不入 Git**）。监测台页面：项目 / 监测 / 调参 / 验收。

