# 项目状态

> 更新：2026-10-09 · **独立 CDSS** · DX-1/2/3 + **H4/H5** · 主指标 PR-AUC / Brier / DCA/NB  
> **总纲领**：[MASTER_NARRATIVE.md](MASTER_NARRATIVE.md) · 假设细节：[ACADEMIC_THESIS.md](ACADEMIC_THESIS.md) · 老师口径：[TEACHER_INNOVATION.md](TEACHER_INNOVATION.md)  
> **叙事**：机制创新（效用阈值 + 稀疏融合）+ 人机协同；**不**耦合 scheduling

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
| D3 GRU-D 真序列最小对照 | ✅ 协议已落地（主表 **PR-AUC / Brier**）· labs-only 真跑（limit=400）：GRU PR-AUC **0.046** vs LGBM **0.053**（差 -0.007，容差内通过）· 报告 `artifacts/d3/report.md` · vitals 非空率 0（dump 口径） |
| TFT-lite | ✅ 同 (x,m,δ) 注意力消融 · `python scripts/d3_tft_lite_smoke.py` · **非**默认模型 |
| D3 真库试跑 | ✅ `--labs-only` 通路；缺 chartevents 时不宣称「完整监护时序增益」 |
| 化验时序深化 | ✅ 10 项化验 · 阳/阴混合抽样 + **分层 split** · LGBM / GRU-D / TFT-lite · **因果窗** `charttime < intime+h`（h 与 label 同行）· 旧数字（LGBM 0.471 / GRU 0.502 / TFT 0.451）为 lookback 越界泄漏版，**勿外推** · 合入后需重跑 `d3_lab_triple_compare` 再填主表 · **非**自然患病率 · 默认床旁仍 LGBM |
| HITL / care_plan | ✅ 解释页反馈 audit · Streamlit「规划」页 · 非床位调度 |
| SOTA 对标 | ✅ [SOTA_SURVEY.md](SOTA_SURVEY.md) · 假设 **H1–H5** |
| **H4 效用校准 UCEW** | ✅ 旧 S2：F1 thr=0.537→**NB thr=0.384**（窗[0.05,0.40]、报警≤20%）· test NB **-0.053→-0.035**（Δ**+0.018**）· 报警 1.7%→2.2% · PR-AUC 不变 **0.139** · `artifacts/h4/utility_compare.md` |
| **H5 稀疏门控融合** | ✅ 旧 S2 h=6：轨迹=多时刻 LGBM 代理（无 Layer0）· Brier **0.043→0.040** · 稀疏半 PR-AUC **0.075→0.086** · 整体 PR-AUC 0.164→0.153（容差）· `artifacts/h5/sparsity_fusion.md` |
| 数据飞轮 | ✅ [DATA_FLYWHEEL.md](DATA_FLYWHEEL.md) · `python -m application.summarize_hitl` |
| S2 dump 文件 | ✅ 本机已 restore **旧 S2**（20260802）→ Docker `:5433` · feat/label **472,290** · 五时刻齐全 · **非 v2**（阳性≈2.34%）· 权威 v2 见 [DUMP_READY.md](DUMP_READY.md) |
| 本机整合（2026-10-09） | ✅ Docker · pytest H4/H5 单测绿 · train PR-AUC **0.139** / Brier **0.040** · DCA 已生成 · D3 `--mock` 通 · 真序列/因果窗三方 ⚠ 缺 Layer0 `mimic` |

### 飞轮 KPI（演示种子 · 2026-10-09）

> 来源：`python -m application.summarize_hitl --days 7`  
> 审计文件 gitignore：`artifacts/hitl/explain_audit.jsonl`（**演示种子，非床旁真实临床周报**）

| KPI | 来源 | 当前 |
|-----|------|------|
| 周解释次数 | `n` | **12** |
| 采纳率 | `accept_rate` | **0.5833**（7/12） |
| 驳回率 | `reject_rate` | **0.2500**（3/12） |
| 编辑率 | `edit_rate` | **0.1667**（2/12） |

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
