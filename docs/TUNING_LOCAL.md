# 本地学习 / 调参环境

> 更新：2026-08-02 · **演示台 v4**（项目 / 监测 / 调参 / 验收）

| 工具 | 入口 |
|------|------|
| Streamlit 监测台 | `.\scripts\run_console.ps1`（推荐，清 env + venv） |
| MLflow | `python -m mlflow ui --backend-store-uri sqlite:///./mlflow.db` |
| S2 dump 训练 | `python -m application.train --from-existing` |
| dump | `dumps/icu_decision_S2-full_mimic_94458stays_20260802.dump` · [`DUMP_READY.md`](DUMP_READY.md) |
| 答辩口播 | [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) |

```powershell
cd d:\project\icu-decision-agent
.\scripts\run_console.ps1
# 默认 http://localhost:8501
```

页面：**项目**（开场叙事）· **监测**（风险徽章 / 多时刻 / SHAP / 建议动作）· **调参** · **验收**（校准 + 净受益）。  
勿用裸 `streamlit`（需 venv）。顶尖下一步见 [`TOP_TIER_NEXT.md`](TOP_TIER_NEXT.md)。
