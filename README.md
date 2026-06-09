# ICU 临床恶化预警决策智能体

项目二 · 仓库 `icu-decision-agent` · PostgreSQL `icu_decision`

## 方案与状态

- 架构与数据层：见团队文档 `docs/ARCHITECTURE.md`（或向负责人索取 `SPEC.md`）
- **当前进度**：`docs/STATUS.md`
- **队友恢复 dump**：`docs/TEAM_RESTORE.md`

## 模型路线

- **P0**：LightGBM + Isotonic + SHAP（表格特征，标签 12h 内死亡）
- **P1+**：GRU-D → TFT（时序/波形）

## 快速开始

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

## 架构

L5 Streamlit → L4 application → L3 domain → L2 data_access → L1 infra → PostgreSQL
