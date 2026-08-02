# 执行路线图 ROADMAP_EXEC（人机双可读）

> 更新：2026-08-02 · PR #6（mj）已合 · liujiawei 暂不合

## 人读摘要

| Wave | 含义 | 状态 | 主责 |
|------|------|------|------|
| **1** | 去泄漏特征 + 真三集划分 | ✅ | C |
| **2** | 全量 train + 分层指标 | ✅ Owner 底座 + **B PR #6** | B/C |
| **待审计** | `liujiawei` 特征扩展（AUC≈0.89） | ⏸️ 不合 main | B/A |

| 划分 | 比例 | 规则 |
|------|------|------|
| train/val/test | 0.7/0.1/0.2 | stay_id · seed=42 · 禁止合并 val+test |

## Agent 上下文

```text
特征契约：configs/features.yaml（denied: hospital_expire_flag, los_hours）
划分：domain/models/split.py
训练：python -m application.train
验收：pytest tests/test_features_leak.py tests/test_predict.py tests/test_evaluation.py tests/test_train.py -q
禁止：把结局列加回 FEATURE_COLS；用 test 调参；未审计合入 liujiawei
Issue：#4 A restore；#3 可关（PR#6）；liujiawei 待审计
```

## Wave2 清单

- [x] Owner：feat/label + artifact + P0-full dump
- [x] B：分层复核 · PR #6 merged 2026-08-02
- [ ] A：dump restore 验收（#4）
- [ ] `liujiawei`：泄漏/时刻审计通过前 **禁止合 main**

## Owner 可代备（Wave2 数据底座 · 非 PPO）

前置：Docker/本机 PG + Layer0 `mimic`（或 demo）已导入；`configs/database.yaml` 指向正确 DSN。

```powershell
$env:PYTHONPATH = (Get-Location)
.\scripts\bootstrap_from_dump.ps1   # 或 restore 后跳过
.\.venv\Scripts\python.exe -m application.run_etl_stage
.\.venv\Scripts\python.exe -m application.train   # 内含 build_features + labels + lgbm
.\scripts\export_layer1.ps1 -MimicSource mimic    # 导出含数据的 Layer1；勿只依赖误导性文件名
```

产物：`feat.sample_matrix` · `label.mortality_12h` · `artifacts/models/lgbm_*` · 新 `dumps/` + `DATA_VERSION_*.json`  
**不做**：代 B 改 STATUS 叙事而不跑可复现命令；把总 LOS / expire 加回特征。
