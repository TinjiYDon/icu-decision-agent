# 本地学习 / 调参环境

> 更新：2026-08-02 · Streamlit 交互台（预测 / 细调 / 验收）+ MLflow

| 工具 | 入口 |
|------|------|
| Streamlit 交互台 | `streamlit run presentation/streamlit_app.py` |
| MLflow | `python -m mlflow ui --backend-store-uri sqlite:///./mlflow.db` |
| S2 dump 训练 | `python -m application.train --from-existing` |
| dump | `dumps/icu_decision_S2-full_mimic_94458stays_20260802.dump` · [`DUMP_READY.md`](DUMP_READY.md) |

```powershell
cd d:\project\icu-decision-agent
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\pip.exe install mlflow streamlit
# 若 Layer1 已是 S2（或刚 restore S2 dump）:
.\.venv\Scripts\python.exe -m application.train --from-existing
streamlit run presentation/streamlit_app.py
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///./mlflow.db
```

交互台 Tab：**多时刻预测** · **细调运行**（写 yaml + from-existing）· **验收门禁**（472k 行数 + PR-AUC/Brier/按小时）· STATUS。

`mlflow.db` 已 gitignore。顶尖下一步见工作区 `docs/TOP_TIER_NEXT.md`。
