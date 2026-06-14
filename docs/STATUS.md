# 项目二 · 状态摘要

> 团队总览：[`../PROJECT_STATUS.md`](../PROJECT_STATUS.md)

## P0 进度（Demo）

| 项 | 状态 |
|----|------|
| ETL staging | ✓ 140 stays |
| mortality_12h 标签 | ✓ 2 阳性 / 140 |
| LightGBM | ✓ `artifacts/models/lgbm_mortality_12h.txt` |
| Layer1 dump | ✓ `../dumps/icu_decision_layer1_schemas_*.dump` |

## 一键跑通

```powershell
.\scripts\apply_migrations.ps1
python -m application.run_p0
```

## 分步

```powershell
python -m application.etl_pipeline
python -m application.train
```
