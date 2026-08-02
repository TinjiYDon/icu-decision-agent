# 本地学习 / 调参环境

> 更新：2026-08-02 · **Plotly 临床监测台**（Monitor / Tune / Accept）

| 工具 | 入口 |
|------|------|
| Streamlit 监测台 | `.\.venv\Scripts\python.exe -m streamlit run presentation/streamlit_app.py` |
| MLflow | `python -m mlflow ui --backend-store-uri sqlite:///./mlflow.db` |
| S2 dump 训练 | `python -m application.train --from-existing` |
| dump | `dumps/icu_decision_S2-full_mimic_94458stays_20260802.dump` · [`DUMP_READY.md`](DUMP_READY.md) |

```powershell
cd d:\project\icu-decision-agent
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m streamlit run presentation/streamlit_app.py --server.port 8501
```

页面：**Monitor**（选 stay 自动风险徽章 + 多时刻曲线 + SHAP）· **Tune** · **Accept**。  
勿用裸 `streamlit`（需 venv）。顶尖下一步见 [`TOP_TIER_NEXT.md`](TOP_TIER_NEXT.md)。
