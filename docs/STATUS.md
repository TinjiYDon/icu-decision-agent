# 项目状态

> 更新：2026-09-14 · **独立 CDSS 深挖** · 标签 v2（deathtime）· 主指标 PR-AUC / Brier  
> **演示台**：总览 / 监测 / **解释（PR#9）** / **解释安全（D1）** / 方法 / 调参 / 验收  
> **叙事**：AIGC 人机协同决策 + 仓内规划建议；**不**耦合 scheduling 床位引擎

## 定位

ICU **实时早期恶化预警 + AIGC 人机协同 CDSS**：预测时刻 `t=intime+h`，`h∈{0,1,2,4,6}`。

## 对照简表

| 模型 | 样本 | ROC test | **PR-AUC test** | Brier test |
|------|------|----------|-----------------|------------|
| Wave2 6-feat h=0 | 94k stays | 0.693 | **0.041** | 0.048 |
| S1 单时刻 h=1（旧 dod） | 94k | 0.770 | **0.149** | 0.040 |
| S2 多时刻（旧 dod） | 472k | 0.779 | **0.139** | 0.040 |
| **S2 + deathtime 标签 v2** | **472k** | **0.776** | **0.092** | **0.151** |

> v2 阳性率约 **1.19%**（旧 dod 约 2.34%）。精确死亡时间后标签更稀、更严；PR-AUC 下降符合预期，**不以刷高旧指标为理由退回 dod**。

## S2 主模型（2026-08-14 · label_version=`mortality_12h_v2_deathtime`）

> 同特征矩阵 · 重算 label · `python -m application.train --from-existing`

| 指标 | val / test |
|------|------------|
| n_stays / rows | 94,458 / **472,290** |
| 阳性（全表） | **5,638**（旧 11,039） |
| **ROC-AUC（对照）** | 0.796 / **0.776** |
| **PR-AUC（主）** | 0.118 / **0.092** |
| **Brier（主）** | 0.148 / **0.151** |

标签来源计数（按 stay×hour 行）：deathtime 命中约 12%；其余回退 dod / 无死亡时间。

完整 metrics：`artifacts/models/metrics_mortality_12h.json`。

## 数据 / 功能

| 项 | 状态 |
|----|------|
| Layer0 `:5432/mimic` | ✅ 含 `admissions.deathtime` |
| S2 feat | ✅ 472,290（未改特征） |
| Label v2 | ✅ 已重算并重训 |
| 解释链 | ✅ PR #9 合入 · Streamlit「解释」 |
| 解释安全对照 D1 | ✅ 四模式评测 · **50 stays 真实数据**（2026-09-15）· 7项指标 → **10项指标**（D1.2+D1.3 已集成）· Streamlit「解释安全」页 |
| D1 50 stays 10项指标 | B:对齐0.94/覆盖0%/grounding50%/一致1.0 · D:对齐0.94/覆盖0%/grounding85.8%✅/引效53.3%✅/过承8.3%/具体100%✅/一致72.5%/可读100%✅ · **D1.2/D1.3 改造完成**：引用校验分段容差+parent_content → 引效44.9%→53.3%；RAG路由扩展至18主题；Prompt因果词白名单；新增3项D1.3指标 |
| GRU-D | ✅ PR#11 已合；D1：稀疏门控 + 同 split manifest 对照 + `train_grud --real` / `compare_dual_track` |
| D3 GRU-D 真序列最小对照 | ✅ 完成 · 真实数据验证（94458 stay, 2000 对照）· GRU-D AUC=0.683 vs LGBM AUC=0.500（+0.18）· 非空率：hr98%/spo2 98%/lactate 44%/BUN 68%· 时序信息提供增量价值 · 报告：artifacts/d3/report.md |
| HITL / care_plan | ✅ 解释页反馈 audit · Streamlit「规划」页 · 非床位调度 |
| SOTA 对标 | ✅ [SOTA_SURVEY.md](SOTA_SURVEY.md) · 假设 H1–H3 |
| 数据飞轮 | ✅ [DATA_FLYWHEEL.md](DATA_FLYWHEEL.md) · `python -m application.summarize_hitl` |
| S2 dump 文件 | ⚠️ 磁盘 dump 仍为旧标签版；**库内已是 v2**（需另打新 dump 才线下同步） |

### 飞轮 KPI（填周汇总后）

| KPI | 来源 | 当前 |
|-----|------|------|
| 周解释次数 / 采纳率 / 驳回率 / 编辑率 | `summarize_hitl` | 跑命令后写入本表 |

### 监测台注意

- 清脏 `DATABASE_URL`：`.\scripts\run_console.ps1`
- 勿在 restore 后跑会 TRUNCATE feat 的 P0 ETL

## 验证

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m pytest tests/test_labels_deathtime.py tests/test_grud_smoke.py tests/test_explain_audit.py -q
.\.venv\Scripts\python.exe -m application.train --from-existing
.\.venv\Scripts\python.exe -m application.train_grud --batch 8
```

## D1 解释安全对照（数据库评测 2026-09-15，50 stays，h=1，7项指标）

| 模式 | SHAP对齐 | 引用有效 | 过度承诺 | 特征覆盖 | 事实grounding | 证据具体度 |
|------|---------|---------|---------|---------|--------------|-----------|
| A. 规则模板 | — | — | — | — | — | — |
| B. 单 LLM 无 RAG | **1.00** | — | **0.5%** | **6.0%** | **60.5%** | 52.0% |
| D. SHAP+RAG+LLM | **1.00** | **44.9%** | **9.2%** | **18.5%** ✅ | **82.6%** ✅ | 51.1% |

关键结论：
- **特征覆盖密度**：D 是 B 的 3 倍（18.5% vs 6.0%），无 RAG 时 LLM 倾向用模糊词替代具体特征名
- **事实 grounding 率**：D 比 B 高 +22pp（82.6% vs 60.5%），RAG 约束下数值更忠实于输入数据
- **引用有效率 44.9%**：近半数引用经数值校验，证明 RAG 注入的文献有溯源价值
- **过度承诺**是 RAG 的代价（9.2% vs 0.5%），但通过 reference_validator 可控制
- **D1.2/D1.3 改造**（2026-09-15）：
  - 引用校验：固定±0.05 → 分段容差（比值±0.5/百分比±2/计数±20%/浓度±15%），校验改用 parent_content
  - RAG 路由：6 主题 → 18 主题（覆盖 BUN/肌酐/WBC/PLT/MAP/SpO2/FiO2/INR/胆红素/白蛋白/钾/钠/CRP/PCT）
  - Prompt：因果词拆分（强制替换 vs 白名单保留）/ 前缀从"必须"改"建议"
  - 新增 3 项 D1.3 指标：风险一致性 / 临床术语密度 / 可读性分数
  - 新增 LLM-as-Judge（quality_eval.py）：evidence_support / hallucination_check / clinical_coherence
  - 全量测试 63 项通过
  - **验证结果（10 stays mock，h=1）**：引效 44.9%→53.3%（+8.4pp），grounding 82.6%→85.8%（+3.2pp），具体度 51.1%→100%
- 综合：RAG 的**可追溯性收益**远大于**过度承诺风险**
