# Dump 交付说明（队友 restore / 训练）

> 更新：2026-08-02 · **S2 full dump** · dump 不入 GitHub · 线下分发

## 人读摘要

| 仓 | 文件 | schemas_only | 可用于 |
|----|------|--------------|--------|
| **decision（主）** | `icu_decision_S2-full_mimic_94458stays_20260802.dump` | **false** | ✅ **S2 LightGBM**（feat/label≈**472,290**，`h∈{0,1,2,4,6}`） |
| decision（旧） | `icu_decision_P0-full_mimic_94458stays_20260726.dump` | false | ⚠️ Wave2/单时刻对照；**勿作 S2 底座** |
| scheduling | `icu_scheduling_P0-full_mimic_94458stays_20260802.dump` | **false** | ✅ SOFA/CP-SAT 仿真 |
| Layer0 `mimic.labevents` | 不在 Layer1 dump 内 | — | 重算特征需本机 Layer0 |
| online PPO 轨迹 | — | — | ❌ 不在 dump 内 |

**SHA-256（decision S2）**：`5b71b752cb17bb8513c73682230329b1223a0baa7e52a791d3cd83e9409c1605`  
**元数据**：`dumps/DATA_VERSION_20260802_1758.json`

## Agent 上下文

```text
decision restore: .\scripts\restore_layer1.ps1 -DumpFile .\dumps\icu_decision_S2-full_mimic_94458stays_20260802.dump
然后: PYTHONPATH=. python -m application.train --from-existing
验收: SELECT COUNT(*) FROM feat.sample_matrix;  -- ≈ 472290
      SELECT hour_index, COUNT(*) FROM feat.sample_matrix GROUP BY 1;  -- 五档各 ≈ 94458
禁止: 把 20260726 / schemas_only dump 当 S2 训练底座
禁止: full dump 恢复后在无 Layer0 时运行默认 application.train（会重建 feat/label）
禁止: 宣称 Layer1 dump 含 labevents 或 PPO 轨迹
```

## 恢复命令

```powershell
cd d:\project\icu-decision-agent
.\scripts\restore_layer1.ps1 -DumpFile .\dumps\icu_decision_S2-full_mimic_94458stays_20260802.dump
# 脚本会 GRANT icu_dev；验收 feat≈472290
# Docker Layer1 时加 -PgPort 5433，并设:
# $env:DATABASE_URL = "postgresql+psycopg://icu_dev:icu_dev@localhost:5433/icu_decision"
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m application.train --from-existing
streamlit run presentation/streamlit_app.py
```

**注意**：不要在 restore 后无 Layer0 时跑默认 `application.train`（会重建并可能冲掉 S2 网格）。
