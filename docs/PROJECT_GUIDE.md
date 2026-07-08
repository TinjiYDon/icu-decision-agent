# ICU 决策智能体 · 项目指南

面向所有克隆本仓库的开发者：按下列步骤即可复现数据检查点，无需其他 ICU 项目。

## 架构

```
Layer0 (MIMIC) → ETL → staging / feat / label → LightGBM + SHAP → Streamlit
PostgreSQL: icu_decision
```

## 前置条件

| 项 | 说明 |
|----|------|
| PostgreSQL 16 | 本机 `localhost:5432` 或 Docker `5433` |
| Python 3.11+ | 建议 venv + `pip install -e ".[dev]"` |
| Layer0 数据 | `mimic_iv_demo`（小样本）或自建 `mimic` 全量库 |
| 本地配置 | 从 `configs/*.example` 复制，**勿提交** |

## 推荐流程

### 1. 初始化

```powershell
copy configs\database.yaml.example configs\database.yaml
copy configs\data.yaml.example configs\data.yaml
.\scripts\apply_migrations.ps1
```

### 2. 数据检查点（当前阶段终点）

```powershell
$env:PYTHONPATH = (Get-Location)
.\scripts\run_data_pipeline.ps1
```

包含：环境检查 → ETL → 导出 `dumps/` → 冒烟测试。

### 3. 模型阶段（下一步）

```powershell
python -m application.train
python -m application.run_p0
```

### 从已有 dump 恢复（跳过 ETL）

```powershell
.\scripts\restore_layer1.ps1 -DumpFile .\dumps\icu_decision_P0-etl_mimic_94458stays_20260708.dump
# 或自动选 dumps/ 下最新文件：
.\scripts\bootstrap_from_dump.ps1 -SkipEtl
```

## 检查点一览

| 阶段 | 命令 | 说明 |
|------|------|------|
| 连通性 | `scripts/check_env.py` | 含于 `run_data_pipeline` |
| ETL | `application/run_etl_stage.py` | **数据阶段终点** |
| 导出 | `scripts/export_layer1.ps1` | 输出到 `dumps/` |
| 冒烟 | `pytest tests/test_{smoke,db,etl}.py` | 不含训练 |
| 模型 | `application/train` | 下一步 |

## dump 命名

`{db}_P0-etl_{layer0}_{N}stays_{yyyymmdd}.dump`

示例：`icu_decision_P0-etl_mimic_94458stays_20260708.dump`（放在 `dumps/`，不入 Git）

可选镜像目录：`-DumpMirrorDir <path>` 或环境变量 `LOCAL_DATA_MIRROR`。

## Layer0 切换

`configs/database.yaml` → `mimic_source: demo | full | mimic`

## 待完成

| 项 | 位置 |
|----|------|
| Streamlit 演示 | `presentation/streamlit_app.py` |
| MCP Tool | `docs/INNOVATION_ROADMAP.md` P2 |
| 时序模型 | P1+ `domain/models/` |

## 注意

- 勿提交 `dumps/`、`artifacts/`、`.env`、`configs/database.yaml`
- 本仓库与其他 ICU 项目**无代码或数据依赖**
