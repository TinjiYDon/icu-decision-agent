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

- [ ] A：feat/label 与非 schemas_only dump（#4）— **或 Owner 代备后发新 dump**
- [ ] B：`application.train` 全量 + STATUS 填 `auc_val`/`auc_test`/`pos_rate`（#3）— 若 Owner 已训出 artifact，B 可只复核指标与文档

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
