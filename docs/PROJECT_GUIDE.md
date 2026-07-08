# ICU 决策智能体 · 项目指南

## 架构

```
Layer0 mimic(_demo) → ETL → staging/feat → LightGBM → SHAP → Streamlit
PostgreSQL: icu_decision（Layer1）
```

## 数据与 dump（不入 GitHub）

| 资源 | 路径 |
|------|------|
| Layer1 dump | `d:\project\_local-data\mimic\icu_decision_P0-etl_mimic_94458stays_20260708.dump` |
| MIMIC 全量 | `d:\project\_local-data\mimic\mimic-iv-3.1.zip` |
| 本地配置 | `configs/database.yaml` `configs/data.yaml` |

## 常用命令

```powershell
$env:PYTHONPATH = (Get-Location)

# ✅ 数据阶段检查点（ETL + dump + 冒烟，不含模型）
.\scripts\run_data_pipeline.ps1

# ⏸ 模型阶段（下一步，本阶段不跑）
python -m application.run_p0          # ETL + train
python -m application.train           # 仅训练
```

## 检查点说明

| 阶段 | 命令 | 状态 |
|------|------|------|
| 连通性 | `scripts/check_env.py` | 数据管道含 |
| ETL | `application/run_etl_stage.py` | **本阶段终点** |
| dump 生产 | `scripts/export_layer1.ps1` | 输出到 `dumps/` 并镜像 `_local-data/mimic/` |
| 冒烟 | `pytest tests/test_{smoke,db,etl}.py` | 数据管道含 |
| 模型 | `application/run_p0` · `train` | **下一步** |

## 待完成

| 项 | 文件 |
|----|------|
| Streamlit 演示 | `presentation/streamlit_app.py` |
| MCP Tool | 见 `docs/INNOVATION_ROADMAP.md` P2 |
| GRU-D/TFT | P1+ `domain/models/` |

## GitHub

`TinjiYDon/icu-decision-agent`

## 注意

- 勿提交 dump、artifacts、`.env`
- Layer0 切换：`database.yaml` → `mimic_source: demo|full|mimic`
