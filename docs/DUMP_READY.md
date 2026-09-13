# Dump 交付说明（队友 restore / 训练）

> 更新：2026-09-13 · **S2 full dump** · **不入 GitHub** · 线下单发  
> **标签版本注意**：库内已重算为 `mortality_12h_v2_deathtime`；下表「磁盘 dump」若仍为 20260802 文件，则 **restore 后需再跑 label 重算 + train**，或等待 Owner 另发 v2 dump。

## 单发文件（Owner → 队友）

| 文件 | 绝对路径 | 说明 |
|------|----------|------|
| **主 dump（磁盘，可能仍为旧标签）** | `d:\project\icu-decision-agent\dumps\icu_decision_S2-full_mimic_94458stays_20260802.dump` | S2 · feat≈**472,290** · `h∈{0,1,2,4,6}` |
| 元数据（可选） | `d:\project\icu-decision-agent\dumps\DATA_VERSION_20260802_1758.json` | SHA / 行数说明 |
| **库内权威** | PostgreSQL Layer1 `label.mortality_12h` | **v2 deathtime**（阳性约 1.19%）；见 `docs/STATUS.md` |

**SHA-256（20260802 文件）**：`5b71b752cb17bb8513c73682230329b1223a0baa7e52a791d3cd83e9409c1605`

### v2 对齐（restore 旧 dump 后）

```powershell
cd d:\project\icu-decision-agent
$env:PYTHONPATH = (Get-Location)
# 需 Layer0 admissions.deathtime；仅 schemas dump 不够
.\.venv\Scripts\python.exe -m application.train   # 或项目内 label rebuild 入口
# 若特征已在库： 
.\.venv\Scripts\python.exe -m application.train --from-existing
```

待 Owner 导出新文件后，命名建议：`icu_decision_S2-v2_deathtime_<stays>_<YYYYMMDD>.dump`，并更新本表 SHA。

## 特征口径（答辩口径）

| 项 | 说明 |
|----|------|
| 行数 | `feat.sample_matrix` ≈ **472,290**（94,458 stays × 5 时刻） |
| 主信号 | **年龄 + 化验（BUN/肌酐/Hct/钠/乳酸等）+ 科室等** |
| 生命体征 | dump 内 chart 心率/血压/体温等 **多为 null**（导出时未写入有效 vitals）；监测台以化验面板为主 |
| 标签 v2 | 优先 `admissions.deathtime`；否则回退 `dod`；见 `domain/labels/mortality_12h.py` |

## 恢复 + 训练 + 监测台

```powershell
cd d:\project\icu-decision-agent
.\scripts\restore_layer1.ps1 -DumpFile .\dumps\icu_decision_S2-full_mimic_94458stays_20260802.dump
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m application.train --from-existing
.\scripts\run_console.ps1
```

验收：

```sql
SELECT COUNT(*) FROM feat.sample_matrix;           -- ≈ 472290
SELECT hour_index, COUNT(*) FROM feat.sample_matrix GROUP BY 1 ORDER BY 1;
-- 期望 0/1/2/4/6 各约 94458
SELECT SUM(y)::float / COUNT(*) FROM label.mortality_12h;  -- v2 约 0.012
```

## 禁止

- 把 schemas_only / 旧单时刻 dump 当 S2 底座
- restore 后再跑默认会 TRUNCATE feat 的 P0 ETL
- 把 dump 推进 GitHub
- 用旧 dod 阳性率与 v2 指标混比而不注明版本
