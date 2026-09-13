# icu-decision-agent — AGENTS.md

> AIGC 人机协同 CDSS · LightGBM + SHAP→RAG→LLM · 成员 C=集成  
> **独立项目**：不与 `icu-scheduling-agent` 硬耦合。

## 一句话

MIMIC stay → 特征/标签 → LightGBM 12h 风险 → SHAP+RAG+LLM → 人机确认 → 仓内处置规划建议 → Streamlit。

## 角色

| 成员 | 职责 | 目录 |
|------|------|------|
| A | ETL / migrations / dump | `domain/etl/` `scripts/` |
| B | 特征/标签/训练 / GRU-D | `domain/features/` `domain/models/` |
| C | L4 + UI + HITL / care_plan | `application/` `presentation/` `data_access/` |

## 先读

1. `docs/ROADMAP.md`（Wave D-LIT / D0–D3 / D-FLY）
2. `docs/SOTA_SURVEY.md`（**先对标再创新**；无 LIT 不宣称 SOTA）
3. `docs/DATA_FLYWHEEL.md`
4. `docs/ROADMAP_EXEC.md`
5. `docs/PARAM_STORY.md`
6. `docs/STATUS.md`
7. `docs/DUMP_READY.md` · `docs/TUNING_LOCAL.md`

## 命令

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_features_leak.py tests/test_predict.py tests/test_smoke.py tests/test_mcp_predict.py -q
.\.venv\Scripts\python.exe -m application.train
streamlit run presentation/streamlit_app.py
# MCP（可选依赖）
.\.venv\Scripts\pip.exe install "mcp>=1.0"
.\.venv\Scripts\python.exe -m presentation.mcp_server
```

## 关键契约

- L4：`application.predict_patient.predict_patient(stay_id)` → `risk_score` / `recommend` / `top_factors`
- 解释：`predict_patient_with_explanation` → SHAP + RAG + LLM
- 规划：`application.care_plan` / `domain.planning` → 结构化处置建议（非床位分配）
- 特征：`configs/features.yaml` · `domain.features.build.FEATURE_COLS`（无泄漏）
- 划分：`domain.models.split` · 0.7/0.1/0.2 stay_id seed=42
- MCP：`predict_risk(stay_id)` → 同上（`presentation/mcp_tools.py`）；可选 `explain_risk`
- Bugbot：`docs/BUGBOT.md`
- **不做床位 PPO**；时序对照为 GRU-D/TFT，见 `docs/INNOVATION_ROADMAP.md`
- **禁止**调用 scheduling 的 `optimize_beds` / 读写 `sched.*`

## 分层

Streamlit 只调 L4；禁止页面内 SQL / 直接 import domain（见 `docs/adr/001-layer-boundaries.md`）。

## 数据

dump/artifacts **不入 Git**。
