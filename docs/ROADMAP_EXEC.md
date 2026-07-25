# 执行路线图 ROADMAP_EXEC（人机双可读）

> 更新：2026-07-25 · Wave1 已落地代码 · Wave2 等队友 · Wave3 并行骨架

## 人读摘要

| Wave | 含义 | 状态 | 主责 |
|------|------|------|------|
| **1** | 去泄漏特征 + 真三集划分 | ✅ 代码已合 | C |
| **2** | 全量 `train` + STATUS 写 auc_* | ⏳ 等 B/A + Layer1 | B |
| **3** | （智学并行，见 zhixue ROADMAP） | — | — |

| 划分 | 比例 | 规则 |
|------|------|------|
| train/val/test | 0.7/0.1/0.2 | stay_id · seed=42 · 禁止合并 val+test |

## Agent 上下文

```text
特征契约：configs/features.yaml（denied: hospital_expire_flag, los_hours）
划分：domain/models/split.py → artifacts/models/split_manifest_mortality_12h.json
训练：python -m application.train（Wave2：须先 ETL+build_features）
验收：pytest tests/test_features_leak.py tests/test_predict.py tests/test_mcp_predict.py -q
禁止：把结局列加回 FEATURE_COLS；用 test 调参
Issue：S1-3 特征消毒；S1-1 重训 AUC（B）
```

## Wave2 等待清单（队友）

- [ ] A：feat/label 与非 schemas_only dump（#4）
- [ ] B：`application.train` 全量 + STATUS 填 `auc_val`/`auc_test`/`pos_rate`（#3）
