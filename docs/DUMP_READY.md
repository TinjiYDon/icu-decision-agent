# Dump 交付说明（队友 restore / 训练）

> 更新：2026-08-02 · **仅保留 S2 full dump** · **不入 GitHub** · 线下单发

## 单发文件（Owner → 队友）

| 文件 | 绝对路径 | 说明 |
|------|----------|------|
| **主 dump** | `d:\project\icu-decision-agent\dumps\icu_decision_S2-full_mimic_94458stays_20260802.dump` | S2 · feat/label≈**472,290** · `h∈{0,1,2,4,6}` |
| 元数据（可选） | `d:\project\icu-decision-agent\dumps\DATA_VERSION_20260802_1758.json` | SHA / 行数说明 |

**SHA-256**：`5b71b752cb17bb8513c73682230329b1223a0baa7e52a791d3cd83e9409c1605`  
旧 Wave2（20260726）dump 已从本机清理；勿再索要旧文件作 S2 底座。

## 特征口径（答辩口径）

| 项 | 说明 |
|----|------|
| 行数 | `feat.sample_matrix` ≈ **472,290**（94,458 stays × 5 时刻） |
| 主信号 | **年龄 + 化验（BUN/肌酐/Hct/钠/乳酸等）+ 科室等** |
| 生命体征 | dump 内 chart 心率/血压/体温等 **多为 null**（导出时未写入有效 vitals）；监测台以化验面板为主，缺测显示 `—` |
| 标签 | `label.mortality_12h` 同步 ≈ 472,290 |

## 恢复 + 训练 + 监测台

```powershell
cd d:\project\icu-decision-agent
.\scripts\restore_layer1.ps1 -DumpFile .\dumps\icu_decision_S2-full_mimic_94458stays_20260802.dump
$env:PYTHONPATH = (Get-Location)
# 已有模型产物时可跳过训练
.\.venv\Scripts\python.exe -m application.train --from-existing
.\scripts\run_console.ps1
```

验收：

```sql
SELECT COUNT(*) FROM feat.sample_matrix;           -- ≈ 472290
SELECT hour_index, COUNT(*) FROM feat.sample_matrix GROUP BY 1 ORDER BY 1;
-- 期望 0/1/2/4/6 各约 94458
```

监测台应能看到年龄/化验非空；若体征全 `—` 且年龄/化验也空，说明 feat 被 P0 ETL 冲掉，请重新 restore。

## 禁止

- 把 schemas_only / 旧单时刻 dump 当 S2 底座  
- restore 后再跑默认 `application.train`（无 `--from-existing`）或 P0 `run_pipeline`（会 **TRUNCATE** 并写成占位特征）  
- restore 后无 Layer0 时宣称可重算 vitals  
- 宣称 dump 含完整 labevents 原始表 / 把 dump 推进 GitHub  
