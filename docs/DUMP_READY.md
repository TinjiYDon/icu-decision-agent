# Dump 交付说明（队友 restore / 训练）

> 更新：2026-09-15 · **S2 v2 deathtime dump** · **不入 GitHub** · 线下单发  
> **标签版本注意**：库内已重算为 `mortality_12h_v2_deathtime`（阳性约 1.19%）。
> D2 新增决策曲线分析（DCA + Bootstrap CI），见 `scripts/d2_dca_full.py`。

## 单发文件（Owner → 队友）

| 文件 | 绝对路径 | 说明 |
|------|----------|------|
| **主 dump（磁盘，可能仍为旧标签）** | `d:\project\icu-decision-agent\dumps\icu_decision_S2-full_mimic_94458stays_20260802.dump` | S2 · feat≈**472,290** · `h∈{0,1,2,4,6}` |
| **v2 dump（推荐，含新标签）** | `d:\project\icu-decision-agent\dumps\icu_decision_S2-v2_deathtime_94458stays_<YYYYMMDD>.dump` | 同上 + 正确 mortality_12h_v2 标签 |
| 元数据（可选） | `d:\project\icu-decision-agent\dumps\DATA_VERSION_<YYYYMMDD>_<HHMM>.json` | SHA / 行数说明 |
| **库内权威** | PostgreSQL Layer1 `label.mortality_12h` | **v2 deathtime**（阳性约 1.19%）；见 `docs/STATUS.md` |
| **DCA 报告** | `artifacts/dca/dca_report_<YYYYMMDD>.json` | Bootstrap CI DCA 数值结果 |
| **DCA 摘要** | `artifacts/dca/dca_summary_<YYYYMMDD>.txt` | 人类可读结论 |

### v2 对齐（restore 旧 dump 后）

```powershell
cd d:\project\icu-decision-agent
$env:PYTHONPATH = (Get-Location)
# 需 Layer0 admissions.deathtime；仅 schemas dump 不够
.\.venv\Scripts\python.exe -m application.train   # 或项目内 label rebuild 入口
# 若特征已在库： 
.\.venv\Scripts\python.exe -m application.train --from-existing
# 生成 DCA 报告
.\.venv\Scripts\python.exe scripts\d2_dca_full.py --n_boot 500
```

## D2 决策曲线分析（DCA）

**用途**：在阳性率 ~1.2% 的稀有事件场景下，用净受益（Net Benefit）替代 ROC-AUC 评估临床价值。

**CBR（代价比）约定**：`CBR = 1/9` → 工作阈值 `p_t = 0.1`，含义：临床医生愿意为 1 个真阳性接受 9 个假阳性。这是脓毒症/ICU 死亡率研究的常用标准。

**运行**：
```powershell
# CLI 全量报告
python scripts/d2_dca_full.py --cost_benefit_ratio 0.111 --n_boot 500
# Streamlit 验收页自动调用（带 Bootstrap CI）
streamlit run presentation/streamlit_app.py → 验收门禁
```

**输出字段**：
| 字段 | 含义 |
|------|------|
| `working_nb` | 模型在 CBR=1/9 下的净受益 |
| `treat_all_nb` | 全部干预的净受益（参考线） |
| `ci_lower / ci_upper` | Bootstrap 95% CI（500 次重采样） |
| `conclusion` | "✅ 模型净受益 > 全干预" 或 "⚠️ 阈值需调整" |

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
.\scripts\restore_layer1.ps1 -DumpFile .\dumps\icu_decision_S2-v2_deathtime_94458stays_<YYYYMMDD>.dump
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
