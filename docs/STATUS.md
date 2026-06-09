# 项目二 · 状态摘要

> 完整团队日志见上级目录 `../PROJECT_STATUS.md`（ monorepo 工作区）  
> **MIMIC**：下载中 · **阶段**：P0 · **阻塞**：Layer0 导入后实现完整 ETL

## 仓库

- 本项目：`icu-decision-agent`
- 调度项目：`icu-scheduling-agent`（独立仓库）

## P0 目标

ETL → 12h 死亡标签 → LightGBM → SHAP → Streamlit → Layer1 dump

## 本地命令

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
copy .env.example .env
copy configs\database.yaml.example configs\database.yaml
copy configs\data.yaml.example configs\data.yaml
.\scripts\apply_migrations.ps1
pytest tests/ -q
python -m application.etl_pipeline
streamlit run presentation/streamlit_app.py
```

## 队友 restore dump

见仓库内 `docs/TEAM_RESTORE.md`（从 project-code 同步）

## 代码待办（MIMIC 后）

- [ ] `domain/etl` 完整 pipeline
- [ ] `domain/labels/mortality_12h.py`
- [ ] `domain/models/lgbm.py`
- [ ] `domain/explain/shap.py`
- [ ] Streamlit 联调
