# icu-decision-agent 全流程计划行动清单（2026.11 – 2027.04）

> **配套规范**：[OPTIMIZATION_SPEC.md](OPTIMIZATION_SPEC.md) · **唯一指标台账**：[METRICS_LEDGER.md](METRICS_LEDGER.md) · **阶段门禁**：[STAGE.md](STAGE.md)  
> **团队边界**：3 人（A=数据/ETL、B=模型/算法、C=应用/UI/集成）、周期 5 个月、剩余经费 206 元、严格遵守 [001-layer-boundaries.md](adr/001-layer-boundaries.md)（不耦合床位调度）。

---

## 阶段 0：仓库治理、文档清淤与代码合仓（第 1 周 · 2026.11 下旬）

- [x] **0.1a 同步远端 `origin/main` 并验证 11 月模块**
  - 已将本地 `main` 快进至 `177f2cc`（标签 `v2026.11-midterm-sync`），拉齐 `domain/labels/time_coarsening.py`、`domain/models/lare.py`、`lar_policy.py`、`policy_set.py`、`selective_review.py`、`pac_safety_net.py`、`scripts/lare_mimic_run.py` 等 10 个 11 月 PR；核心无 DB 单元测试 `90 passed`。
- [ ] **0.1b 合入仍滞留仓外的 2 组 11 月最新脚本（待队友提交 PR）**（负责人：A / B）
  - 合入 11 月 18 日 PPT **8 模型全量体征对照（B0–B7，补齐 3,542 万条体征，AP=0.323/0.334）** 的取数、特征、训练与画图脚本。
  - 合入中期报告附录 B 的 `scripts/lare_controlled_validation.py`。
- [x] **0.2 清理与归档淘汰文档**
  - 已通过 `git mv` 将 25 份过时文档（10 月 9 日旧 H4/H5/H6 CUGEW 文档及 7–9 月进度/计划文档）无损归档至 `docs/archive/`（归档区现共 30 份，保留 `preflight.yaml` 与代码注释引用的必需文档）。
  - 已落地新 SSOT 文档：`docs/OPTIMIZATION_SPEC.md`、`docs/ACTION_CHECKLIST.md`、`docs/METRICS_LEDGER.md`，并更新 `AGENTS.md`、`README.md`、`docs/README.md`、`docs/STAGE.md`、`docs/MASTER_NARRATIVE.md`。
- [x] **0.3 统一旧报告与 PPT 口径冲突**
  - 已在 `docs/METRICS_LEDGER.md` 登记 V0/V1/V2a/V2b 四套历史口径边界，正式作废旧 S2/H4/H5/H6 数字，预留 V3 封版指标槽位。

> **Gate 0 剩余阻塞项**：合入仓外 B0–B7 与 `lare_controlled_validation.py` 脚本；组会拍板多时点网格是否升级为等距逐小时（建议 `1..12h` 或 `1..24h`）。

---

## 阶段 1：数据底座修复与封版（第 2–3 周 · 2026.12 上中旬）

- [ ] **1.1 `labevents.storetime` 专项审计与因果过滤改造**（负责人：A）
  - 统计全部关键化验的 `storetime IS NULL` 比例与 `storetime - charttime` 延迟分位数（重点核查动脉血气/乳酸 `50813`）。
  - 主口径强制 `l.storetime IS NOT NULL AND l.storetime <= t`；同步记录 `COALESCE(l.storetime, l.charttime) <= t` 敏感性差异。
- [ ] **1.2 修复 4 处已定位的源码数据坑**（负责人：A）
  - 修复 `domain/features/build.py`（L194–200）预入科化验锁死不更新 Bug，改为按 `<= t` 最新可用观测取数（含变化量/极值/缺测标志 $m$/间隔 $\Delta t$）。
  - 增加**在院存活资格过滤**：每个预测时刻 $t_{s,h}$ 强制满足 `t < outtime` 且 `(deathtime IS NULL OR t < deathtime)`。
  - 修复 `data_access/mimic_repo.py`（L370）`fetch_gcs_total` 将全缺测变成 `0` 的缺陷（保留 `NULL/NaN`）。
  - 永久禁用出院结算编码 `diagnoses_icd`（`mimic_repo.py` L547），并在 `tests/test_features_leak.py` 新增断言。
- [ ] **1.3 统一患者级切分（`subject_id`）与精确结局标签**（负责人：A + B）
  - 在特征表/样本矩阵及 `migrations/001_layer1_schemas.sql` 显式保留 `subject_id`、`stay_id`、`hour_index`、`intime`、`outtime`、`deathtime`。
  - 改造 `domain/models/split.py` 支持按 `subject_id` 分层 60/20/20（seed=42）切分 `train / dev / test`，导出唯一 `split_manifest_subject_v3.json`。
  - 主结局固定为仅使用 `admissions.deathtime` 的未来 12h 院内死亡（不回退 `dod`），保留原始 `deathtime` 供受控粗化使用。
- [ ] **1.4 编写红线测试并封版**（负责人：A）
  - 新增 `tests/test_causal_data_v3.py`（断言零 `subject_id` 跨折、在院存活、化验 `storetime <= t`、无泄漏列）。
  - 导出封版快照与 `data_manifest_v3.json`（含 SHA256），复现新版 B5/B7 并填入 `docs/METRICS_LEDGER.md`。

---

## 阶段 2：算法构建与模型修改（第 4–6 周 · 2026.12 下旬 – 2027.01 上旬）

- [ ] **2.1 预测模型修改与自定义目标构建（`domain/models/`，负责人：B）**
  - **M0 强基线**：移除 `scale_pos_weight`，统一训练与推理缺失处理为 `np.nan`（彻底消除 `lgbm.py` L262 推理填 `0` Bug），实现 `domain/models/calibration.py`（Platt / Isotonic + Brier / ECE / 可靠性曲线）。
  - **M1a / M1b 事件加权**：实现训练集事件权重归一 $w_{s,h} = \bar{m}/m_s$ 及提前量偏好归一 $u_{s,h}=1+\kappa(\tau_{s,h}/\Delta)$，验证 $\sum_{y=1} w_{s,h}=N_+$。
  - **M2a 自定义 `fobj`**：新建 `domain/models/event_objective.py`，向量化实现事件级 Smooth-Max (LogSumExp) + BCE 的一阶梯度 $g_{s,h}$ 与严格正定凸 MM 海森 $h_{s,h}$（单轮 $O(N)$，下界保护 `>= 1e-6`），配套事件级早停 `feval`。
  - **数学单测**：新建 `tests/test_event_objective.py` 通过有限差分梯度校验（误差 $<10^{-6}$）、$\alpha=0$/$m_s=1$ 精确退化检验及极端 logit 数值正定性检验。
- [ ] **2.2 状态依赖动态复核队列引擎（`domain/decision/`，负责人：B + C）**
  - 新建 `domain/decision/review_queue.py`：实现因果前向状态转移 $\mathbf{z}_{s,h}=(I_{s,h},\tau^{\mathrm{last}}_{s,h},p^{\mathrm{last}}_{s,h})$ 与策略族 `D0 / D1 / D2 / D2-Dual`（防 `h=0→1` 信息到达假跃升），提供开发集预算标定器（对齐 $B \in \{20, 50, 100\}$ / 千 stay-hour）。
  - 新建 `domain/decision/event_metrics.py`：输出独立事件首次覆盖率、漏报事件数、提前时长（中位数/IQR）、Event-NNR、重复报警率及复核/升级分离占用；编写 `tests/test_review_queue.py`。
- [ ] **2.3 扩展现有 LARE 与 $\varepsilon$-近优模块至多自由度策略空间（负责人：A + B）**
  - 复用并扩展 `domain/labels/time_coarsening.py`：补充重叠/保守判定包装及同翻转率时间无关随机噪声生成器（`seed=0..9`）。
  - 复用并扩展 `domain/models/policy_set.py` 与 `domain/models/lare.py`：支持在 D2 $(\tau, T_{\mathrm{sil}}, \delta)$ / D2-Dual $(\tau_{\mathrm{low}}, \tau_{\mathrm{high}}, T_{\mathrm{sil}}, \delta)$ 策略空间枚举，计算住院级去重 Jaccard、共识/争议集、$D_{\mathrm{within}}(\varepsilon)$ 与 $D_{\mathrm{cross}}(\varepsilon, g)$（支持 $\varepsilon \in \{0,1,2,5\}$ 例与开发集 1-SE 带宽）。

---

## 阶段 3：模型训练与全量对照实验（第 7–10 周 · 2027.01 中旬 – 2027.02 下旬）

- [ ] **3.1 目标函数与校准消融（M0 / M1a / M1b / M2a）**（负责人：B）
  - 统一患者切分与 5 种子预算；在开发集搜索 M1b 的 $\kappa \in \{0.5, 1.0, 2.0\}$ 与 M2a 的 $\alpha \in \{0.2, 0.5, 0.8\} \times \tau \in \{0.5, 1.0, 2.0\}$。
  - 完成无校准 / Platt / Isotonic 对比，锁定校准概率输出；核对 B0–B7 强对照，仅在有 GPU 余力时补跑 GRU-D/TFT-lite 附录。
- [ ] **3.2 模型 × 动态决策队列全交叉实验**（负责人：B + C）
  - 跑通 `{M0, M1a, M1b, M2a} × {D0, D1, D2, D2-Dual} × B ∈ {20, 50, 100}` 全矩阵，绘制连续“实际报警负荷 — 独立事件首次覆盖率”曲线与 $T_{\mathrm{sil}} \times \delta$ 敏感性热图。
  - **止损门控**：若 M2a 相对 M0 未达患者级统计显著，立即如实定性为严谨消融负结果，主贡献聚焦 D1/D2 动态队列与 LARE/$\varepsilon$-近优稳健性。
- [ ] **3.3 多粒度同源粗化（LARE）与 $\varepsilon$-近优归因实验**（负责人：A + B）
  - 在最优模型 + D2 / D2-Dual 策略空间跑完 `exact / 1h / 6h / 12h / 24h` + 同率随机对照 + 真实 `dod` 补充对照 + 多组代价比（`4:1:0.2` ~ `32:1:0.2`），输出实验 A/B 完整归因表与 LAR-Policy 鲁棒对照。

---

## 阶段 4：严格结果验证与统计审计（第 11–13 周 · 2027.03 上中旬）

- [ ] **4.1 1,000 次患者级整组 Bootstrap 与配对检验**（负责人：B）
  - 编写 `scripts/audit_bootstrap_v3.py`，按 `subject_id` 整组重抽样 1,000 次计算所有主指标 95% CI 与配对差值显著性（严禁按行重抽）。
- [ ] **4.2 50 例人工时间戳溯源与零泄漏终审**（负责人：A）
  - 抽查 50 例高危及“D2 新增覆盖”病例核对原始 MIMIC 时间戳；交叉复核开发集参数选择未接触测试集。
- [ ] **4.3 亚组稳健性、典型案例与一键出图脚本**（负责人：B + C）
  - 按科室、年龄、高/低缺测亚组、早/后期时段分层验证；整理 10 例成功与 10 例失败/争议复核案例；编写 `scripts/render_paper_artifacts.py` 自动从 `artifacts/v3/` 生成全部论文图表并回填 `docs/METRICS_LEDGER.md`。

---

## 阶段 5：Streamlit 原型升级、软著材料与结题投稿（第 14–20 周 · 2027.03 下旬 – 2027.04）

- [ ] **5.1 推理底座迁移与 ADR-001 队列服务接入**（负责人：C + B）
  - 将 `application/predict_patient.py` 与 `domain/models/lgbm.py` 迁移至 V3 封版模型并彻底移除缺失填 `0`；新增 `application/review_queue_service.py`。
- [ ] **5.2 接通“带时间与来源的事实纠正 + VoI 演示”原型**（负责人：C）
  - 扩展 `domain/explain/hitl_audit.py` 与 `presentation/ui/explain.py`：支持字段级纠错（校验 `timestamp <= t` 防未来回填）→ 联动重算派生特征 → 展示纠正前后校准风险、队列优先级与 SHAP 变化 → 保存双快照 JSONL。
  - 新增动态复核队列 & LARE 共识/争议集可视化演示；严控剩余 206 元经费，不再对 LLM 解释链做大规模重跑。
- [ ] **5.3 软著申报、结题报告与会议投稿**（负责人：全员）
  - 整理软著说明书、源程序代码文档与截图并提交申请。
  - 定稿结题报告、答辩 PPT 与投稿论文；**逐条人工核验所有参考文献链接（含 2026 年 arXiv 预印本）**；封存 `README_REPRO.md` 与结题标签 `v2027.04-final-release`。

---

## 附录：六大里程碑硬性门控（Gate）

| 里程碑 | 预计时间 | 硬性通过条件 |
| :--- | :--- | :--- |
| **Gate 0 合仓清淤** | 2026.11 底 | 主干同步至 `177f2cc`（✅）；旧文档归档（✅）；台账建立（✅）；待补齐 B0–B7 与 `lare_controlled_validation.py` 合仓 |
| **Gate 1 底座封版** | 2026.12 中 | `storetime` + 在院存活 + `subject_id` 红线测试全绿；封版复现 B5/B7 并冻结 SHA256 |
| **Gate 2 算法单测** | 2027.01 上 | M2a 梯度有限差分误差 $<10^{-6}$；动态队列与多自由度 $\varepsilon$-近优单测全绿 |
| **Gate 3 主实验收敛** | 2027.02 底 | M0–M2a × D0–D2 × 多预算 + 多粒度 LARE/$\varepsilon$-近优全量产物落盘 |
| **Gate 4 统计终审** | 2027.03 中 | 1,000 次患者级 Bootstrap 完成；50 例人工溯源零泄漏；论文图表一键生成 |
| **Gate 5 结题交付** | 2027.04 底 | 新版 Streamlit 原型流畅演示；软著提交；结题报告与论文定稿 |
