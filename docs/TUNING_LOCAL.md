# 本地学习 / 调参环境

> 更新：2026-08-02 · **演示台 v4.1**（项目 / 监测 / 调参 / 验收）

| 工具 | 入口 |
|------|------|
| Streamlit 监测台 | `.\scripts\run_console.ps1`（推荐，清 env + 杀端口 + venv） |
| MLflow | `python -m mlflow ui --backend-store-uri sqlite:///./mlflow.db` |
| S2 dump 训练 | `python -m application.train --from-existing` |
| dump | `dumps/icu_decision_S2-full_mimic_94458stays_20260802.dump` · [`DUMP_READY.md`](DUMP_READY.md) |
| 答辩口播 | [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) |

```powershell
cd d:\project\icu-decision-agent
.\scripts\restore_layer1.ps1 -DumpFile .\dumps\icu_decision_S2-full_mimic_94458stays_20260802.dump
.\scripts\run_console.ps1
# 默认 http://localhost:8501
```

页面：**项目** · **监测**（可用 stay 列表 / 缺测 `—` / SHAP 有值优先）· **调参** · **验收**（校准 + 净受益）。

### 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| 体征/化验全 0 或仅 los | 跑了 P0 ETL，feat 被占位覆盖 | 重新 restore S2 dump |
| 换 stay 像没刷新 | 旧版 widget key；或全是占位零向量 | 用 v4.1 + 确认 restore |
| 连错库 | 环境变量 `DATABASE_URL` 指向 zhixue 等 | 只用 `run_console.ps1` |
| 心率等长期 `—` | dump 本身 vitals 多为 null | 正常；看年龄/化验与 SHAP |

勿用裸 `streamlit`（需 venv）。顶尖下一步见 [`TOP_TIER_NEXT.md`](TOP_TIER_NEXT.md)。
