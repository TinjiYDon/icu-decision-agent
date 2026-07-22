# icu-decision-agent — AGENTS.md

> 临床恶化预警 · LightGBM + SHAP · 成员 C=集成

## 一句话

MIMIC stay → 特征/标签 → LightGBM 12h 死亡风险 → SHAP + 建议档位 → Streamlit。

## 角色

| 成员 | 职责 | 目录 |
|------|------|------|
| A | ETL / migrations / dump | `domain/etl/` `scripts/` |
| B | 特征/标签/训练 | `domain/features/` `domain/models/` |
| C | L4 + UI | `application/` `presentation/` `data_access/` |

## 命令

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_predict.py tests/test_smoke.py -q
.\.venv\Scripts\python.exe -m application.train
streamlit run presentation/streamlit_app.py
```

## 关键契约

- L4：`application.predict_patient.predict_patient(stay_id)` → `risk_score` / `recommend` / `top_factors`
- 参数故事：`docs/PARAM_STORY.md`
- **不做 PPO**；P3 是时序模型（GRU-D/TFT），见 `docs/INNOVATION_ROADMAP.md`

## 分层

Streamlit 只调 L4；禁止页面内 SQL / 直接 import domain（见 `docs/adr/001-layer-boundaries.md`）。

## 数据

dump/artifacts **不入 Git**；schemas_only dump 不足做 RL。
