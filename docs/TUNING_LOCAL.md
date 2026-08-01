# 本地学习 / 调参环境

> 更新：2026-07-27 · Streamlit + 可选 MLflow

| 工具 | 入口 |
|------|------|
| Streamlit | `streamlit run presentation/streamlit_app.py` |
| MLflow | `python -m mlflow ui --backend-store-uri sqlite:///./mlflow.db` |
| full dump 训练 | `python -m application.train --from-existing` |
| 进度 | `d:\project\_local-data\mimic\PROGRESS.md` |

```powershell
cd d:\project\icu-decision-agent
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\pip.exe install mlflow streamlit
.\.venv\Scripts\python.exe -m application.train --from-existing
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///./mlflow.db
```

`mlflow.db` 已 gitignore。特征/划分契约见 `configs/features.yaml`、`configs/labels.yaml`。
