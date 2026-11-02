# icu-decision-agent · 技术总览（Technical Overview）

> 版本：v2.0（2026-11-02）｜代码基线：`main@3fdc288`
> 本文档定位：**意图与责任分离**的完整技术说明——「为什么这样做」（意图）与「谁来做、在哪做」（责任）分开写，供介绍/展示/答辩/交接四方使用。
> 索引方式：① `docs/README.md` 分类索引；② 本文档 §9 目录树；③ 代码内 docstring。

---

## 0. 文档导航

| 你想做什么 | 读什么 |
|---|---|
| 一句话讲清项目 | 本文 §1、§2 |
| 讲创新点、答"差在哪" | 本文 §3、§4 |
| 讲架构与分层 | 本文 §1.3–1.5 |
| 讲为什么选这些技术 | 本文 §5 |
| 讲实验与数据 | 本文 §6、`docs/STATUS.md`、`docs/LARE.md` |
| 答辩被追问 | 本文 §4（调研结果）与 §10（边界） |

---

## 1. 架构

### 1.1 意图（为什么这么分）

总目标：**可决策、可解释、可人审**的早期预警 CDSS。

| 意图 | 落点 |
|---|---|
| 可决策：工作点由预算与代价决定 | H4 效用阈值 + DCA；LARE 固定 K/α |
| 可解释：每个高风险有结构化因素 | SHAP→结构化→RAG→LLM→引用校验 |
| 可人审：accept/reject/edit 留痕 | HITL JSONL + summarize_hitl |
| 评估先固定标签语义 | `lare.py`（两套标签）；P1 标签审计 |
| 没有数据不宣称可行 | `lar_policy.py`（Hoeffding 上界开关） |

### 1.2 责任（谁做、在哪做）

| 层 | 目录 | 责任 | 关键文件 |
|---|---|---|---|
| L1 | `infra/` | 配置/引擎 | `config.py`、`db.py` |
| L2 | `data_access/` | MIMIC 只读 | `mimic_repo.py` |
| L3 | `domain/` | 特征/标签/模型/解释/规划 | `features/build.py`、`labels/mortality_12h.py`、`models/{lgbm,utility_calibrate,dca,selective_review,pac_safety_net,lare,lar_policy}.py`、`explain/*` |
| L4 | `application/` | 用例编排 | `predict_patient.py`、`train.py`、`care_plan.py`、`summarize_hitl.py` |
| L5 | `presentation/` | UI/MCP | `streamlit_app.py`、`ui/*`、`mcp_tools.py` |

### 1.3 数据流

```text
MIMIC stay → 特征(19, 因果窗 charttime< intime+h)
  → 标签(12h 死亡, deathtime 优先, 回退 dod)
  → LightGBM p(s,h)（64 树/depth4/lr0.1/scale_pos_weight, seed42）
  → H4 效用阈值 / DCA → 工作点
  → SHAP→RAG→LLM 解释 → 引用校验 → HITL
  → LARE：两标签 × 四策略 × 固定 K/α 评估
  → LAR-Policy：min-max 鲁棒双阈值（可选 Hoeffding 上界）
```

### 1.4 关键参数

- 预测时刻 `h∈{0,1,2,4,6}`；标签窗 `[t, t+12h]`；划分 stay 级 0.7/0.1/0.2，seed=42。
- 主指标 PR-AUC/Brier（稀有事件，ROC 仅对照）。
- H4：窗 `[0.05,0.40]`、`max_alert_rate=0.20`、cost_ratio=4.0。
- LARE：K=0.2、α=0.05、δ=0.1、网格 50×50。

---

## 2. 主要创新点（一览）

| ID | 创新点 | 状态 | 证据 | 可主张性 |
|---|---|---|---|---|
| D-M1 | LARE（标签时间歧义下的复核评估） | 已实现+真实数据 | 合成 yield 差 +0.0839、Jaccard 0.2446；MIMIC 首跑 Jaccard 0.2429 | 强（主） |
| D-M2 | LAR-Policy（min-max 鲁棒双阈值） | 已实现 | robust_cost 0.15995、max_regret 0.0134；MIMIC robust_cost 0.0708 | 中（第二点） |
| D-M3 | Hoeffding 有限样本上界 | 已实现 | n=4000/α=5% 不可行；与 n≈47k 同源 | 中（边界） |
| D-B | P1 标签审计（dod 是 deathtime 粗化超集） | 已验证 | 5,401/48.9%；阈值 0.145→0.165/0.250；ΔPR-AUC −0.0174 | 强（证据底座） |
| — | SBR/PAC/H4/H5/H6/证据门控 | 工程/负结果 | H5 反降、H6 退化、CARE 同值 | 不主张 |

---

## 3. 创新点详细解释

### D-M1 LARE

- **问题**：复核/弃权评估默认标签正确；但 `dod` 是 `deathtime` 的粗化超集（dod-only 5,401=48.9%），会改变阳性集合、阈值与复核对象。
- **做法**：`build_coarse_label` 构造超集标签；`fixed/double/sbr/deferral` 四策略；`evaluate_policy` 输出 yield/missed/release_risk/budget/Jaccard/Spearman。
- **与先行工作差异**：conformal deferral 假设标签固定；我们问“时间粗化会不会让复核策略看起来更好、复核了不同的人”。
- **证据**：合成 yield dod 0.1158 vs death 0.0320（+0.0839）；标签专属阈值 Jaccard=0.2446；MIMIC 首跑 Jaccard=0.2429、Δτ_low=−0.1061。

### D-M2 LAR-Policy

- **问题**：按单一标签选的策略换一套标签就不稳。
- **做法**：`fit_lar_policy` 在 `tau_low<tau_high` 网格上 `min max(cost_dod, cost_death)`，约束 `(review+alert)/n ≤ K` 且两套标签 `release_risk ≤ α`；同分取放行最多。
- **与先行工作差异**：创新在“对标签时间歧义鲁棒”的目标，不是新 SBR/deferral 算法。
- **证据**：合成 robust_cost=0.15995（=cost_dod）、max_regret=0.0134；MIMIC robust_cost=0.0708、max_regret=0.0039。

### D-M3 Hoeffding 上界

- `fit_lar_policy(bound="hoeffding", delta=0.1)`：`risk + sqrt(log(m/δ)/(2n)) ≤ α` 才可行。
- 证据：n=4000/α=5% 不可行（radius≈0.0343）；与 α=1% 需 n≈47k 同源——**没有足够数据不宣称可行**。

### D-B P1 标签审计

- dod-only 5,401（48.9%）；基准率 1.19%→2.34%；max-F1 阈值 0.145→0.165、约束 0.250；v2 ΔPR-AUC=−0.0174（CI 全负）、dod +0.0010（CI 跨 0）；α=1% 预算合规 1.86–3.23% → 0.90–1.02%。

---

## 4. 调研结果（撞车尽调摘要）

| 先行工作 | 做了什么 | 我们的差异 | 等级 |
|---|---|---|---|
| conformal + cost-aware deferral（[Sci Rep 2026](https://www.nature.com/articles/s41598-026-40637-w)） | 校准+conformal+成本感知单阈值 | 假设标签固定；我们做标签歧义评估 | 轻/中 |
| ICU 报警 LTT 抑制（[预印本](https://browse-export.arxiv.org/pdf/2609.12365)） | LTT 上界抑制有害报警 | 同上：标签语义不在评估变量 | 轻 |
| EGCL（[NeurIPS 2026](https://neurips.cc/virtual/2026/loc/sydney/172036)） | evidence gate + 四值 release policy | 他们做放行状态机；我们做标签歧义评估 | 重（不主张证据门控） |
| OpenMed guarded provenance（[文档](https://openmed.life/docs/clinical/guarded-output-provenance/)） | 溯源+人审状态机 | 同上 | 重 |
| QCC（[MICCAI 2026](https://papers.miccai.org/miccai-2026/0843-Paper4233.html)） | 证据优先、补证据→重判 | 我们不做新交互 | 重 |
| 适当依赖综述/框架（[2604.23896](https://arxiv.org/abs/2604.23896)、[2606.06081](https://arxiv.org/abs/2606.06081)） | 指标系统化 | 我们只借用指标 | 重（不主张新指标） |

结论：**“证据门控/人补证据/适当依赖”被占；“标签时间歧义 × 复核评估”未被覆盖。**

---

## 5. 选型原因

| 选择 | 原因 |
|---|---|
| LightGBM 骨干 | 表格数据、CPU 可训、SHAP 原生；v2 下 PR-AUC 0.092/Brier 0.151 |
| PR-AUC/Brier 主指标 | 12h 死亡阳性 1.19%，ROC 会虚高 |
| deathtime 优先标签 | dod 是粗化超集，会改变结论（P1 审计） |
| H9 SBR + PAC 工程化 | 作为复核基线与边界（α=1% 需 n≈47k），不主张算法创新 |
| numpy 纯函数（lare/lar_policy） | 无 DB/网络，可单测、口径与 pac_safety_net 一致 |
| RAG+引用校验解释链 | 引用有效 44.9%、grounding 82.6%；LLM 外网仅演示，MIMIC 数据不出本机 |
| MIMIC runner 带 F1 门 | `LARE_MIMIC_ALLOW=1` 才连库；退出码 0/2/3 |

---

## 6. 实现路径

1. **数据**：`icu_decision` 库 `paper_feat.sample_matrix`（472,290 行 ×5 时刻）+ `paper_label.mortality_12h(_dod)`。
2. **新模块落点**（本轮 PR #23/#24/#25）：
   - `domain/models/{lare,lar_policy}.py`（LAR-Policy 含 Hoeffding 上界）
   - `scripts/{lare_demo,lare_mimic_run,import_mimic_31}.py`
   - `tests/test_{lare,lar_policy,lare_runner}.py`
   - `docs/LARE.md`
3. **测试矩阵**：no-DB 回归 **47 passed**；CI 见 `.github/workflows`（仓库暂无 pytest CI，测试本地跑）。
4. **MIMIC 实验**（F1 已确认）：`LARE_MIMIC_ALLOW=1 python scripts/lare_mimic_run.py --seeds 5 --hours 0,1,2,4,6 --dsn postgresql://icu_dev:icu_dev@127.0.0.1:5432/icu_decision`。

---

## 7. 专有名词详解

| 术语 | 解释 |
|---|---|
| LARE | Label-Ambiguity Review Evaluation：在 dod/deathtime 两套标签下评估复核策略 |
| LAR-Policy | 对两套标签做 min-max 的鲁棒双阈值策略 |
| dod / deathtime | 死亡日期口径 / 精确死亡时间口径；dod 是 deathtime 粗化超集 |
| review yield | 复核集合中真阳性比例 P(y=1|review) |
| missed imminent | 放行区漏掉的“即将死亡”质量 E[1{y_death=1}·1{release}] |
| reviewed-set Jaccard | 两套标签各自选出的复核集合重合度 |
| max_regret | 相对标签专属最优成本的最大后悔 |
| Hoeffding bound | 有限样本上界（可选开关，防“没有数据也宣称可行”） |
| SBR / PAC | 选择性复核（预算约束 DP）/ 概率近似正确安全网；本项目作工程基线 |
| HITL | Human-in-the-loop：采纳/驳回/编辑留痕 |
| F1 | MIMIC DUA/CITI 合规门（2026-11-02 用户已确认，真实数据实验已开跑） |

---

## 8. 关键 API 与主要函数

| 函数 | 签名 | 说明 |
|---|---|---|
| `predict_patient` | `(stay_id, hour_index=None) -> dict` | risk_score/recommend/top_factors |
| `build_coarse_label` | `(y_death, p, *, add_rate=0.5, rng=None)` | 构造 dod 超集标签 |
| `evaluate_policy` | `(p, y_dod, y_death, policy_masks, *, K=None, alpha=None) -> dict` | LARE 指标 |
| `fit_lar_policy` | `(p, y_dod, y_death, *, K=0.2, alpha=0.01, n_grid=50, cost_fn=4.0, bound="none", delta=0.1) -> LARPolicy` | 鲁棒双阈值 |
| `label_specific_policies` / `max_regret` | 见 `lar_policy.py` | 标签专属最优/后悔 |
| `select_double_threshold` | `(p, y, alpha, delta, n_low, n_high, min_release_rate) -> DoubleThresholdResult` | PAC 双阈值（复用） |
| `scripts/lare_mimic_run.py` | `(--dsn --seeds --hours --model --out --smoke)` | MIMIC 网格运行器 |
| `scripts/import_mimic_31.py` | `(--src --dsn --dry-run)` | MIMIC-IV 3.1 CSV 导入 |
| MCP `predict_risk` | `(stay_id, hour_index=None) -> JSON` | 对外接口 |

---

## 9. 文档位置与索引方式

```text
docs/
├── README.md              ← 分类索引入口（总览/创新/架构/实验/数据/移交）
├── TECHNICAL_OVERVIEW.md  ← 本文档（技术总览）
├── MASTER_NARRATIVE.md    ← 总目标叙事（SSOT）
├── LARE.md                ← LARE/LAR-Policy 详解
├── PAC_SAFETY_NET.md / H9_SBR.md  ← 工程基线
├── ARCHITECTURE_MATH.md   ← 新旧架构与公式
├── LABEL_AUDIT.md         ← P1 标签审计
├── STATUS.md / ROADMAP.md ← 进度与计划
├── archive/               ← 旧版本（保存不删）
```

**索引方式**：① `docs/README.md` 按用途分六类链接；② 本文 §0 导航表；③ 代码 docstring；④ 旧版文档在 `docs/archive/`。

---

## 10. 合规与移交

- **F1 状态**：DUA/CITI 已于 2026-11-02 经用户确认；真实 MIMIC 实验已启动（`icu_decision` 库 `paper_feat/paper_label` 各 472,290 行）。
- HITL 12 条只作演示数据，不作人因证据。
- 数据不入 Git；LLM 解释链当前为外网配置，MIMIC 数据不出本机（合规红线）。
- 分工：A=ETL/dump；B=特征/标签/模型；C=L4+UI+HITL。
