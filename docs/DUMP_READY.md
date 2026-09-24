# Dump 交付说明（队友 restore / 训练）

> 更新：2026-09-22 · **S2 v2 dump（含 DCA 完整协议）** · **不入 GitHub** · 线下单发
> **标签版本**：`mortality_12h_v2_deathtime`，阳性率 ≈ 2.20%（全库 ~1.19%，test split ~2.20%）

## 单发文件（Owner → 队友）

| 文件 | 绝对路径 | 说明 |
|------|----------|------|
| **v2 dump（当前权威）** | `dumps\icu_decision_v2-full_mimic_94458stays_20260922.dump` | v2 · feat≈**472,290** · `h∈{0,1,2,4,6}` · Phase=v2 |
| 元数据 | `dumps\DATA_VERSION_20260922_2013.json` | SHA / 行数说明 |
| DCA 报告 | `artifacts\dca\dca_lgbm_*.json` | Bootstrap CI + 工作点 + 关键阈值净受益 |
| **库内权威** | PostgreSQL Layer1 `label.mortality_12h` | **v2 deathtime**；见 `docs/STATUS.md` |

**SHA-256（v2 dump 20260922 20:13）**：`D9DEB0453470C6D59F0D922F57A4689B52EB335CC8219F900BC4F910F8664DC0`
**SHA-256（旧 dump 20260802）**：`5b71b752cb17bb8513c73682230329b1223a0baa7e52a791d3cd83e9409c1605`

### v2 dump 直接恢复（无需再跑 label 重算）

```powershell
cd d:\project\icu-decision-agent
$env:PYTHONPATH = (Get-Location)
# v2 dump 已含 v2 deathtime 标签 + h=0/1/2/4/6 特征
.\scripts\restore_layer1.ps1 -DumpFile .\dumps\icu_decision_v2-full_mimic_94458stays_20260922.dump
.\.venv\Scripts\python.exe -m application.train --from-existing    # 重训 LGBM
.\.venv\Scripts\python.exe -m scripts.d2_dca_full --model lgbm     # 生成 DCA 报告
.\scripts\run_console.ps1                                          # 启动前端
```

## 特征口径（答辩口径）

| 项 | 说明 |
|----|------|
| 行数 | `feat.sample_matrix` ≈ **472,290**（94,458 stays × 5 时刻） |
| 主信号 | **年龄 + 化验（BUN/肌酐/Hct/钠/乳酸等）+ 科室等** |
| 生命体征 | dump 内 chart 心率/血压/体温等 **多为 null**（导出时未写入有效 vitals）；监测台以化验面板为主 |
| 标签 v2 | 优先 `admissions.deathtime`；否则回退 `dod`；见 `domain/labels/mortality_12h.py` |

## D2 验收协议（决策曲线分析）

### 运行

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m scripts.d2_dca_full --model lgbm --cost-ratio 4 --bootstrap 1000
```

### 产出（`artifacts/dca/`）

| 文件 | 格式 | 内容 |
|------|------|------|
| `dca_lgbm_<timestamp>.json` | JSON | 完整 DCA 数据：曲线、CI、工作点、关键阈值 |
| `dca_lgbm_<timestamp>.md` | Markdown | 可读报告 |

### 验收条件

1. **优于全部干预**：在阈值区间 `[0.01, 0.50]` 内，模型 NB > treat_all NB（比"对所有患者干预"好）
2. **CI 有意义**：95% CI 范围合理（不过宽），与均值趋势一致
3. **工作点清晰**：F1 或临床成本比寻优有唯一解，指标合理
4. **代价比敏感性**：`cost_ratio ∈ {2, 4, 6}` 均给出合理曲线（不过度敏感）

> 注意：在极低阳性率（<3%）场景下，NB 可能始终为负，但只要 NB > treat_all 且有 CI 支撑，即视为通过。

### 代价比说明

- **默认 cost_ratio=4.0**：ICU 场景下漏诊（FN）代价约为误报（FP）的 4 倍
- 对应阈值约 **t ≈ 0.05~0.15**（低阈值区间，符合早期预警场景）
- 公式：`NB = TP/n - FP/n × cost_ratio × odds(t)`，`odds(t) = t/(1-t)`

## 恢复 + 训练 + 监测台

```powershell
cd d:\project\icu-decision-agent
.\scripts\restore_layer1.ps1 -DumpFile .\dumps\icu_decision_v2-full_mimic_94458stays_20260922.dump
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m application.train --from-existing
.\.venv\Scripts\python.exe -m scripts.d2_dca_full
.\scripts\run_console.ps1
```

验收：

```sql
SELECT COUNT(*) FROM feat.sample_matrix;           -- ≈ 472290
SELECT hour_index, COUNT(*) FROM feat.sample_matrix GROUP BY 1 ORDER BY 1;
-- 期望 0/1/2/4/6 各约 94458
SELECT SUM(y)::float / COUNT(*) FROM label.mortality_12h;  -- v2 约 0.022
```

## 禁止

- 把 schemas_only / 旧单时刻 dump 当 S2 底座
- restore 后再跑默认会 TRUNCATE feat 的 P0 ETL
- 把 dump 推进 GitHub
- 用旧 dod 阳性率与 v2 指标混比而不注明版本
- DCA 报告中的 cost_ratio 与阈值需一致标注，不可混用不同协议的数字
- 用 `predict_stay` 逐条调用代替批量 Booster 预测（会耗尽 DB 连接池）
