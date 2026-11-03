# STAGE（decision 项目阶段门禁）

> 项目：icu-decision-agent｜当前阶段：S4（结论与成稿）｜更新时间：2026-11-03
> 规则：不跨阶段开工；旧阶段文档进 `docs/archive/`，不删。

| 阶段 | 目标 | 输入 | 产出 | Exit 条件 | 状态 |
|---|---|---|---|---|---|
| S0 问题定义 | 明确“标签语义决定复核评估” | 需求 | `docs/BRIEF.md` L0/L1 | 3 个反例写清 | ✅ |
| S1 证据底座 | 标签审计 | MIMIC | `docs/LABEL_AUDIT.md` + 数字 | 阳性/阈值/CI 可复核 | ✅ |
| S2 方法原型 | LARE + LAR-Policy 合成跑通 | S1 | `lare.py`/`lar_policy.py` + 单测 | 单测绿 + demo | ✅ |
| S3 真实数据 | MIMIC 两标签实验 | S2 + F1 | `artifacts/lare/mimic_lare.json` | F1 确认 + 预检通过 | ✅ |
| S4 结论成稿 | 四张表 + 负结果 | S3 | 报告/论文 | 证据完整率 100% | 🔄 |

## 门禁

```powershell
.\.venv\Scripts\python.exe C:\Users\lenovo\.dsh\skills\complex-project-harness\scripts\preflight.py --config preflight.yaml
```

## 当前门禁证据

- S1：dod-only 5,401/48.9%；阈值 0.145→0.165/0.250；ΔPR-AUC −0.0174
- S2：`pytest tests/test_lare.py tests/test_lar_policy.py` → 20 passed
- S3：`artifacts/lare/mimic_lare.json`（Jaccard 0.2429、robust_cost 0.0708）
- S4 未完成项：四张表成稿、旧叙事（CUGEW）清理、坏链修复

## 阶段变更

| 日期 | 从 | 到 | 依据 |
|---|---|---|---|
| 2026-11-03 | S3 | S4 | F1 确认 + MIMIC LARE 跑通 |
