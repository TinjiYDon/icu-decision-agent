# icu-decision-agent — AGENTS.md

> 智慧 ICU 早期恶化预警与人机协同决策 CDSS · 3 人协作（A=数据/标签 · B=模型/算法 · C=应用/UI/集成）  
> **独立项目**：不与 `icu-scheduling-agent` 硬耦合，禁止调用 scheduling 的 `optimize_beds` 或读写 `sched.*`。

## 一句话（2026-11-21 · 中期后 V3 执行主线）

**总目标**：在严格时间因果（化验 `storetime <= t` · 在院存活过滤 · `subject_id` 患者级切分）下，构建**事件级可优化、动态队列可决策、标签时间精度可检验、解释可人审留痕**的 ICU 12h 死亡早期预警决策系统。  
**核心流水线**：MIMIC-IV 因果底座 → LightGBM 事件级目标消融（**M0 → M1 → M2a**）+ 概率校准 → **状态依赖动态复核队列（D0/D1/D2/D2-Dual）** → **同源粗化 LARE + 多自由度 $\varepsilon$-近优名单稳定性** → SHAP+RAG+LLM + 事实纠正原型（Streamlit）。

> **重要**：旧 H4 效用阈值 / H5 稀疏融合 / H6 CUGEW（DEI/ANB/UEI/RWSG/SUFH）已正式裁撤，相关历史文档已移入 `docs/archive/`，**禁止再作为当前设计依据**。

## 角色

| 成员 | 职责 | 主要目录 |
|------|------|----------|
| A | ETL / `storetime` 与因果清洗 / 粗化算子 | `domain/etl/` `domain/labels/` `scripts/` |
| B | 特征 / 患者级切分 / LightGBM M0–M2a / 队列与统计验证 | `domain/features/` `domain/models/` `domain/decision/` |
| C | L4 编排 / Streamlit 原型 / 事实纠正与 HITL / 软著与材料 | `application/` `presentation/` `data_access/` |

## 先读（当前唯一 SSOT）

1. **`docs/OPTIMIZATION_SPEC.md`（V3 项目优化说明 · 痛点与三大支柱完整公式 · SSOT）**
2. **`docs/ACTION_CHECKLIST.md`（阶段 0–5 全流程行动清单、分工与 Gate 门控）**
3. **`docs/METRICS_LEDGER.md`（唯一指标台账 · 区分作废口径与 V3 正式槽位）**
4. `docs/STAGE.md` · `docs/MASTER_NARRATIVE.md`
5. `docs/LARE.md` · `docs/TECHNICAL_OVERVIEW.md` · `docs/LABEL_AUDIT.md`
6. `docs/adr/001-layer-boundaries.md`（分层硬约束）

## 常用命令

```powershell
$env:PYTHONPATH = (Get-Location)
# 核心无 DB 单元测试
.\.venv\Scripts\python.exe -m pytest tests/test_lare.py tests/test_lar_policy.py tests/test_lare_runner.py tests/test_policy_set.py tests/test_time_coarsening.py tests/test_selective_review.py tests/test_pac_safety_net.py tests/test_features_leak.py tests/test_smoke.py tests/test_dca.py -q
# 训练与控制台
.\.venv\Scripts\python.exe -m application.train
streamlit run presentation/streamlit_app.py
# MCP（可选依赖）
.\.venv\Scripts\pip.exe install "mcp>=1.0"
.\.venv\Scripts\python.exe -m presentation.mcp_server
```

## 关键契约与红线

- **L4 预测与解释**：`application.predict_patient.predict_patient(stay_id)` / `predict_patient_with_explanation`
- **处置规划**：`application.care_plan` / `domain.planning`（仅结构化处置建议，**非床位分配**）
- **因果红线（V3）**：严禁使用 `diagnoses_icd`（出院结算码）、`los_hours`、`outtime`、`hospital_expire_flag`；化验须校验 `storetime`；每时点须过滤 `t < outtime` 且 `t < deathtime`；禁止推理缺失填 `0`；移除 `scale_pos_weight`
- **切分与统计红线**：按 `subject_id` 患者级切分（60/20/20，seed=42），不确定性统一采用 **1,000 次按 `subject_id` 整组 Bootstrap**
- **分层边界**：Streamlit 只调 `application/`（L4），禁止页面内 SQL 或直接 import `domain/`（见 `docs/adr/001-layer-boundaries.md`）
- **数据边界**：`dumps/`、`artifacts/` **不入 Git**
- **工程协议**：多阶段任务先读 `docs/STAGE.md` 与 `docs/ACTION_CHECKLIST.md`；遵循先单测/红测 → smoke → 全量验证。
