# 项目状态

> 更新：2026-07-31 · B 严格分层复核 · 完整不平衡指标

## 数据检查点

| 项 | 状态 |
|----|------|
| Layer0 `mimic` | ✅ icustays/patients/admissions + **labevents 1.58e8**（Layer0，不在 dump 内） |
| ETL staging | ✅ 94,458 |
| **dump（训练用）** | ✅ `dumps/icu_decision_P0-full_mimic_94458stays_20260726.dump` · `schemas_only=false` · 含 feat/label |
| 冒烟 | ✅ |
| 交付说明 | [`DUMP_READY.md`](DUMP_READY.md) |

## 模型 / 特征

| 项 | 状态 |
|----|------|
| 无泄漏 FEATURE_COLS | ✅ |
| 真三集 0.7/0.1/0.2 | ✅ stay_id 严格分层 · seed=42 |
| LightGBM | ✅ B 独立复核 · [`BASELINE_EXPERIMENT.md`](BASELINE_EXPERIMENT.md) |
| 标签审计 | ⚠️ 日期级 `dod` 不足以精确判断 12h · [`LABEL_AUDIT.md`](LABEL_AUDIT.md) |
| L4 / Streamlit / MCP | ✅ |
| 调参台 | [`TUNING_LOCAL.md`](TUNING_LOCAL.md) · MLflow `sqlite:///./mlflow.db` |
| PPO | ❌ 本仓不做 |

## Wave2 指标

| 指标 | 值 |
|------|-----|
| feat/label | 94,458 |
| 阳性 | 2,099（~2.22%） |
| split n | train 66,121 · val 9,446 · test 18,891 |
| split 阳性 | train 1,469 · val 210 · test 420 |
| **ROC-AUC val / test** | **0.673 / 0.693** |
| **PR-AUC val / test** | **0.039 / 0.041** |
| Brier val / test | 0.048 / 0.048 |
| validation 选阈值 | 0.3458（最大 F1） |
| test Precision / Recall | 5.34% / 18.33% |
| artifact | `artifacts/models/lgbm_mortality_12h.txt`（不入 Git） |

**结论：full dump 已能稳定复现弱基线；当前特征和标签精度不足以支持临床部署。**

## GitHub（2026-08-02）

- ✅ [PR #6](https://github.com/TinjiYDon/icu-decision-agent/pull/6) mj 已合
- 🔔 Issue #4 A restore 仍开
- ⏸️ 分支 `liujiawei` **不合**（待泄漏审计）

## 验证

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_features_leak.py tests/test_predict.py -q
```
