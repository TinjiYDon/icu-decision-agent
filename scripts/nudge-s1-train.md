## 催促 · 成员 B · #S1-1 训练闭环

请尽快完成全量训练并写入 STATUS。

### 人读摘要

| 项 | 内容 |
|----|------|
| Owner | B 算法 |
| 任务 | `python -m application.train` 全量 + AUC/阳性率写入 `docs/STATUS.md` |
| 依赖 | A 确认 feat/label 对齐 |
| C 已就绪 | L4 `predict_patient` + Streamlit + recommend 档位 |

### Agent 上下文

```text
入口：python -m application.train
产物：artifacts/models/lgbm_mortality_12h.txt + model.registry
注意：hospital_expire_flag 可能标签泄漏，评估是否移出 FEATURE_COLS
验收：pytest tests/test_predict.py -q；STATUS 写明 auc / pos_rate
RL：本仓不做 PPO，走监督学习 + SHAP 故事（方案 C）
```
