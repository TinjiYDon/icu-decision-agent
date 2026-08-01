# 执行路线图 ROADMAP_EXEC（人机双可读）

> 更新：2026-07-27 · Wave1/2 底座已交 · dump 可训见 DUMP_READY

## 人读摘要

| Wave | 含义 | 状态 | 主责 |
|------|------|------|------|
| **1** | 去泄漏特征 + 真三集划分 | ✅ 代码已合 | C |
| **2** | 全量 `train` + STATUS 写 auc_* | ✅ Owner 代训已填 · B 可复核 | Owner→B |
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

- [x] Owner：feat/label + train artifact + `P0-full` dump（2026-07-26）
- [x] B：严格分层复核 + 完整不平衡指标 + 实验笔记（2026-07-31）
- [ ] A：确认可从新 dump restore（#4）

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
