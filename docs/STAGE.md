# STAGE（icu-decision-agent 阶段门禁）

> 项目：icu-decision-agent｜当前阶段：**阶段 0 收尾（待合入仓外 B0–B7 与 `lare_controlled_validation.py` 脚本）→ 即将进入阶段 1（V3 因果数据封版）**｜更新时间：2026-11-21  
> 规则：不跨阶段开工；旧阶段文档归档至 `docs/archive/`，保存不删；实验数字唯一登记于 [METRICS_LEDGER.md](METRICS_LEDGER.md)。

---

## 1. 中期后结题执行阶段（2026.11 – 2027.04 · 当前生效）

| 阶段 | 目标 | 核心规范与产出 | Exit 门禁条件 | 状态 |
| :--- | :--- | :--- | :--- | :---: |
| **阶段 0 仓库治理与合仓** | 同步 `origin/main`、归档淘汰文档、统一口径台账、合入仓外 11 月脚本 | [OPTIMIZATION_SPEC.md](OPTIMIZATION_SPEC.md)、[ACTION_CHECKLIST.md](ACTION_CHECKLIST.md)、[METRICS_LEDGER.md](METRICS_LEDGER.md)、`docs/archive/`（30 份） | `main` 同步至 `177f2cc`（✅）、无 DB 单测 90 passed（✅）、旧文档归档（✅）；**待队友合入 B0–B7 与 `lare_controlled_validation.py`** | 🔄 |
| **阶段 1 因果数据封版** | 审计化验 `storetime`、修复预入科化验锁死与在院存活过滤、统一 `subject_id` 60/20/20 切分 | `dataset_v3_causal`、`split_manifest_subject_v3.json`、`tests/test_causal_data_v3.py` | 泄漏红线测试全绿；在 V3 底座复现 B5/B7 并冻结 SHA256 | ⏳ |
| **阶段 2 算法构建与单测** | 构建动态复核队列（D0/D1/D2/D2-Dual）、LightGBM M0/M1/M2a 自定义 `fobj`、扩展多自由度 LARE/$\varepsilon$-近优 | `domain/models/event_objective.py`、`domain/decision/review_queue.py`、`domain/models/policy_set.py` 扩展 | M2a 有限差分梯度误差 $<10^{-6}$；队列与近优单测全绿 | ⏳ |
| **阶段 3 全量训练与对照** | 跑完 M0–M2a × D0–D2 × 多预算交叉消融，以及 1h/6h/12h/日级 LARE 与 $\varepsilon$-近优归因 | `artifacts/v3/experiments/` 全量产物与日志 | 完整实验矩阵落盘；按止损门控诚实定性 M2a 增益 | ⏳ |
| **阶段 4 统计验证与终审** | 1,000 次患者级 Bootstrap CI、50 例原始时间戳人工溯源、亚组检验与一键出图 | `scripts/audit_bootstrap_v3.py`、`scripts/render_paper_artifacts.py` | 零泄漏交叉复核通过；[METRICS_LEDGER.md](METRICS_LEDGER.md) 全表填齐 | ⏳ |
| **阶段 5 原型、软著与投稿** | 修复推理零填充并迁移 Streamlit 至 V3、接入事实纠正/VoI 与队列演示、软著与结题定稿 | 新版 Streamlit 原型、软著材料、结题报告与投稿论文 | 原型与实验同源一致；参考文献逐条核实；打 `v2027.04-final-release` 标签 | ⏳ |

---

## 2. 中期前历史阶段记录（S0–S4，2026-10-31 – 2026-11-11）

| 阶段 | 目标 | 关键证据 | 状态 |
| :--- | :--- | :--- | :---: |
| S0 问题定义 | 明确“标签语义决定复核评估” | `docs/BRIEF.md` | ✅ |
| S1 证据底座 | 标签审计（dod-only 5,401 / 48.9%） | `docs/LABEL_AUDIT.md` | ✅ |
| S2 方法原型 | LARE + LAR-Policy 合成跑通 | `domain/models/lare.py` / `lar_policy.py` | ✅ |
| S3 真实数据 | 旧 S2 MIMIC 两标签离线对照 | `scripts/lare_mimic_run.py`（已入台账 V1） | ✅ |
| S4 中期收口 | 受控粗化（D1）与 $\varepsilon$-策略集初版入库 | `time_coarsening.py`、`policy_set.py`（`177f2cc`） | ✅（合并转入阶段 0–5） |

---

## 3. 阶段变更日志

| 日期 | 变更 | 依据 |
| :--- | :--- | :--- |
| 2026-11-03 | S3 → S4 | F1 确认 + MIMIC LARE 跑通 |
| 2026-11-11 | S4 口径修订 | 补齐 `time_coarsening.py`（#33）与 `policy_set.py` |
| 2026-11-21 | 转入中期后阶段 0–5 | 同步 `origin/main@177f2cc`，整合 11-18 B0–B7 进展与导师三项核心意见（动态队列决策 + LightGBM M0→M1→M2a + 多自由度 LARE/$\varepsilon$-近优），归档 25 份淘汰旧文档 |
