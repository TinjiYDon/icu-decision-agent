# Label audit — mortality_12h

## 当前实现（2026-08-14）

`domain/labels/mortality_12h.py`：

1. **优先** `admissions.deathtime`（时间戳）  
2. 缺失时回退 `patients.dod`（日期级）  
3. 配置：`configs/labels.yaml` → `death_time_prefer: deathtime`，`label_version: mortality_12h_v2_deathtime`

`data_access.mimic_repo.fetch_cohort` 已 SELECT `a.deathtime`。

## 仍需人工/离线步骤

- 在 Layer0 上重跑 `build_labels` + `application.train`  
- 导出带版本号的新 dump（旧 dump 无原始 deathtime，无法在 dump 内单独重算）  
- 对比 v1(dod) vs v2(deathtime) 阳性率与 PR-AUC 漂移

## 历史问题（仍成立于仅 dod 时）

日期级 `dod` 会把整天视为死亡区间，污染 12h 窗。v2 路径用于消除该偏差。
