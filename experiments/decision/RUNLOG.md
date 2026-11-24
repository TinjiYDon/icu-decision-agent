# RUNLOG — 决策侧对照实验（decision-exp / task-2）

记录顺序：做了什么 → 命令 → 耗时 → 关键数字 → 踩过的坑。所有命令在仓库根目录 `C:\Users\lenovo\Desktop\decision shcedule` 下执行。

## 0. 环境、口径、写入边界

| 项 | 值 |
|---|---|
| 建模解释器 | `icu-decision-agent\.venv\Scripts\python.exe`（lightgbm 4.7.0 / scikit-learn 1.9 / scipy / polars 2.0 / pyarrow / pandas） |
| 绘图解释器 | `C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe`（matplotlib 3.11.1，venv 里没有 matplotlib） |
| 数据 | MIMIC-IV 3.1，`experiments/data/*.parquet` 只读，由 data-etl 按 `PROTOCOL.md §2` 产出 |
| 写入边界 | 仅 `experiments/decision/`（`cache/`=中间产物，`results/`=交付物，`figs/`=图） |
| 随机性 | 划分 seed=42（**subject 级**分层 70/10/20，同一 subject 的 5 个时点在同一折）；LightGBM seed 42（B7 用 42–46）；bootstrap seed=42，1000 次 |
| 主指标 | AUPRC（患病率 1.22%，故为主）；同时报 AUROC / Brier / LogLoss / 校准斜率+截距 / ECE(15) / 每千床报警 20-50-100 的灵敏度·PPV / DCA 净收益 / 4:1 漏检代价 |

**纪律**：阈值、等渗校准器、代价阈值全部只在 val 上拟合；test 每个模型只评一次；没有在 test 上挑模型/特征/阈值/亚组；新旧方法共用同一 `splits.json`、同一 `y`。唯一的"非严格因果"特征是合并症 `hx_*`（来自出院 ICD，已单列进消融并在文档中标明）。

## 1. 指标库与自检

`src/eval_lib.py`：`ece_score` / `calibration_slope_intercept` / `budget_metrics` / `dca_curve` / `cost_at_threshold` / `best_cost_threshold_on_val` / 患者级加权 bootstrap（`BootBundle`）/ DeLong。
- bootstrap 用"患者多重性 + 预排序"实现，O(n)/次；自检与 sklearn 逐位一致（含大量并列概率的用例，误差 0.00e+00）。
- DeLong 早期实现用 `2*(1-norm.cdf(|z|))`，在 |z|>8 时下溢为 0；改为 `2*norm.sf(|z|)` 并额外存 `log10_p`。`src/refresh_delong.py` 用它重算了 metrics 里的 8 个比较块。

## 2. 旧方法基线（B4）

```
python experiments\decision\src\build_base.py          # 24 s
```
- `cache/stays.parquet` 94,458×27；`cache/labels.parquet` 472,290 行（`y_death`=`mortality_12h`，`y_dod`=`mortality_12h_dod`）；`cache/feat_old.parquet` 472,290×21（旧仓 `paper_feat.sample_matrix` 的 19 个 `allowed` 特征原样取出）。
- `splits.json`：subject 45,757 / 6,536 / 13,073；stay 66,051 / 9,478 / 18,929；行 236,145 / 47,500 / 94,645。
- **数据事实**：旧矩阵里 `vital_*` 全为 null（旧 ETL 只落了化验与静态项）。因此 B4 的"旧方法水平"是"化验 + 静态 + 少量派生"的真实水平，保持 null 不补，作为差值分母。
- B4（test）：AUPRC 0.1253 / AUROC 0.8291 / Brier 0.01133。

```
python experiments\decision\src\build_comorbid.py      # 8 s
```
- 旧仓 `elix_*` 全零（ETL 没填），改用 `diagnoses_icd` 重建 30 个粗粒度合并症 `hx_*`：chf 0.276 / arrhythmia 0.400 / hypertension 0.642 / diabetes 0.306 / renal 0.417 / liver 0.131 / copd 0.229。

## 3. 体征/化验特征层

```
python experiments\decision\src\build_features.py --hours 0,1,2,4,6     # 12–13 s
```
- 输入 `vitals / labs / inputevents / procedureevents / uroutput`，回看窗 `[-24h, h)`，`charttime < intime + h` 硬过滤（无未来信息）。
- 输出 `cache/feat_new.parquet` **472,290 × 309**；组规模：`v_`=78（生命体征 last/mean/min/max/count/hsince）、`l_`=150（化验同构）、`tr_`=7（6h 斜率，需 ≥3 点）、`sofa_`=7、派生评分 27（qSOFA/NEWS2/SIRS/休克指数/P-F/S-F/尿量率/血管活性药/机械通气）、`hx_`=30。
- 单位与坑：温度 223762 ℃ / 223761 ℉ 分开处理；FiO2 有 0–1 与 0–100 两种量纲（>1 除 100）；血压有创/无创分开再合并，MAP 优先有创；`ventmode/o2dev` 这类分类项 `valuenum` 为空，改为"出现即 1"的存在性行，避免整列被丢弃。
- 数据覆盖（data-etl 全量）：vitals 35,427,459 行 / 94,390 stay（HR 99.9%、RR 99.8%、SpO2 99.8%、GCS 99.3%、FiO2 48.9%）；labs 7,273,135；sofa_hourly 4,479,316。
- 除零产生的 `inf` 统一转 null（模型看到"缺失"而不是 ±inf）。

## 4. 主实验 B0–B7

```
python experiments\decision\src\train_eval.py --suite full --boot 1000 --tag full    # 320 s
python experiments\decision\src\finalize_metrics.py                                  # 2 s
python experiments\decision\src\refresh_delong.py full                               # 2 s
```
| 模型 | 特征 | 模型类 |
|---|---|---|
| B0 | 无 | 训练集患病率常数 |
| B1 | `sofa_hourly.sofa_total` 1 维 | 等渗回归（单调映射为概率），缺失用训练集中位数 |
| B2 | `qsofa, news2` 2 维 | 逻辑回归（标准化 + 零填充） |
| B3 | 全体征矩阵（302 维） | L2 逻辑回归（C=0.1，中位数补缺，标准化，±8 截断） |
| B4 | 旧仓 19 个 `allowed` 特征 | LightGBM |
| B5 | 全体征矩阵（同 B3） | LightGBM（300 树 / lr 0.05 / 31 叶 / min_child 50 / 子采样 0.8 / colsample 0.8 / λ=1，seed 42） |
| B6 | 同 B5 | B5 概率过 val 上拟合的等渗校准器 |
| B7 | 同 B5 | seed 42–46 五个模型的概率平均 |

test（n=94,645，阳性 1,154）：

| 模型 | AUPRC [95%CI] | AUROC [95%CI] | Brier | ECE15 | 斜率 | sens@20/50/100 |
|---|---|---|---|---|---|---|
| B0 | 0.0122 | 0.5000 | 0.01204 | 0.0003 | 0.946 | n/a（常数无排序） |
| B4 | 0.1253 [0.098,0.158] | 0.8291 [0.807,0.849] | 0.01133 | 0.0024 | 0.878 | 0.272/0.392/0.501 |
| B1 | 0.0411 | 0.7427 | 0.01185 | 0.0005 | 0.901 | 0.231@50 |
| B2 | 0.0654 | 0.6993 | 0.01174 | 0.0010 | 0.944 | 0.333@50 |
| B3 | 0.2445 | 0.8958 | 0.01056 | 0.0021 | 0.890 | 0.600@50 |
| B5 | 0.3232 [0.276,0.373] | 0.9346 [0.920,0.947] | 0.00974 | 0.0029 | 0.942 | 0.683@50 |
| B6 | 0.3022 | 0.9341 | 0.00972 | 0.0022 | 0.863 | 0.692@50 |
| B7 | 0.3335 [0.284,0.385] | 0.9356 [0.921,0.948] | 0.00964 | 0.0026 | 0.968 | 0.692@50 |

配对比较（患者级 bootstrap 1000 次 + DeLong）：B5 vs B4 ΔAUPRC **+0.1978 [+0.1554,+0.2405]**，DeLong z=+19.59 p=2.0e-85；B7 vs B4 **+0.2082 [+0.1650,+0.2527]** p=3.3e-87；B6 vs B5 ΔAUPRC −0.0210（AUROC 差 p=0.21，等于没变）；B7 vs B5 +0.0103 [−0.0008,+0.0199] p=0.062（不显著）。

踩过的坑（都已修）：
1. `stays` 静态列与 `feat_new` 自带静态列重名 → join 出 `_right` 列；改为只 join 缺失列，join key 必须保留。
2. `to_numpy()` 返回只读视图，`X[~isfinite]=nan` 报 `assignment destination is read-only` → `np.array(..., copy=True)`。
3. B3 逻辑回归吃到 `inf`（休克指数除零）→ 重建特征层时把 inf 转 null，另在建模入口 `clean()` 再兜一层。
4. B1 等渗回归不吃 NaN（`sofa_total_layer` 有 2,371 行无匹配）→ 训练集中位数补缺，并记录补了多少。
5. DeLong p 下溢为 0 → 用 `norm.sf`。
6. 中间版本 `metrics_vitonly.json`（12:34，labs/sofa/vaso 还没进管线）被误引用 → 改名 `metrics_vitonly_prelabs.json` 留档，最终口径只认 `metrics.json` / `metrics_final.json`，并由 `finalize_metrics.py` 与 `table_main.csv` 逐位对齐。

## 5. LARE 标签歧义稳健性

```
python experiments\decision\src\lare.py preds_full      # 3 s
```
- 标签：`y_death`（死亡时间，1.19%）与 `y_dod`（日期粗化，2.34%，是前者的超集，0 反例）。策略=双阈值 (τ_low, τ_high)：`p<τ_low` 放行、`[τ_low,τ_high)` 复核、`≥τ_high` 报警；预算 = 复核+报警占比；代价 = 4×漏检 + 1×误报 + 0.2×复核（全部按行归一）。
- 网格 903 个可行 (τ_low,τ_high)（41 个分位候选的解析组合）；三个选点：A=按 `y_death` 选、B=按 `y_dod` 选、R=min-max 鲁棒（在 val 上最小化 `max(cost_death, cost_dod)`）。val 选点 → test 交叉评估 → regret。
- test 结果：A(τ=0.0291/0.9743) 复核 5.2%、放出死亡质量 15.9%；B=**R**(0.0126/0.9743) 复核 10.2%、放出死亡质量 9.6%；robust_cost **0.0452**、max_regret **0.0042**、复核名单 Jaccard A vs B = **0.514**。
- 敏感度 K∈{0.05,0.10,0.20,0.35} × α∈{0.001,0.003,0.008,0.05}：α≤0.003 时多数组合**无可行解**（1.19% 患病率 + 当前排序质量决定"放行风险 ≤0.3%"无法与"放行 65% 以上"同时满足）；α=0.008 起稳定，K≥0.20 时约束变松、解不再变化。
- 与旧仓对照（Jaccard 0.243 / robust_cost 0.071 / max_regret 0.004）：**未逐位复现**——旧仓用旧排序器与不同网格，这里只保证机制与量级方向一致（Jaccard 明显 <1、max_regret 很小、robust 解向内收）。文档中如实标注"未复现"。

## 6. 消融 / 漂移 / 亚组

```
python experiments\decision\src\extras.py       # ~8 min（17 次 LightGBM 拟合）
```
产物：`results/extras.json`、`tables/table_ablation.csv`、`table_drift.csv`、`table_era_test.csv`、`table_per_hour.csv`、`table_subgroup.csv`、`preds_ablation.parquet`（含各变体 val/test 概率，便于复算）。
- 加性消融（同划分同超参，seed 42）：静态+合并症 → +化验 → +体征 → +派生评分 → +趋势。
- 留一消融：从全组各去掉体征 / 化验 / 趋势 / 派生评分 / GCS / 合并症。
- 类别权重：`scale_pos_weight=8` 一版，对比默认。
- 漂移：MIMIC 日期已去标识化（`intime_year` 落在 2110–2214，不是真实年份），因此按**行量 70% 分位**切"早期/晚期"，只用时间序关系（脚本里显式打印切分年份）；另附"不重训、按 test 内年份分段评 B4/B5/B7"一张表。
- 亚组：年龄 <65/≥65、性别、种族（White/Black/Other）、病区（MICU/SICU/CCU/其他）、入科类型（急诊/急症/择期），亚组内 n<200 或阳性<10 直接跳过（不挑亚组）。

## 7. 图表

```
C:\...\Python312\python.exe experiments\decision\src\figures.py     # 40 s
```
输出 `experiments/decision/figs/`（300 dpi、CJK、`experiments/common/style.py` 统一配色），并镜像到 `experiments/figures/decision/`：`fig_pr_curve.png`、`fig_roc_curve.png`、`fig_calibration.png`、`fig_alarm_budget.png`、`fig_dca.png`、`fig_feature_importance.png`、`fig_lare_regret.png`、`fig_gain_waterfall.png`。
图里的数字全部从 `metrics.json` / `lare.json` / `extras.json` 现算，不写死；标题只描述画的是什么，不做结论式措辞。

## 8. 复现与验证

```
python experiments\decision\src\build_base.py
python experiments\decision\src\build_comorbid.py
python experiments\decision\src\build_features.py --hours 0,1,2,4,6
python experiments\decision\src\train_eval.py --suite full --boot 1000 --tag full
python experiments\decision\src\refresh_delong.py full
python experiments\decision\src\finalize_metrics.py
python experiments\decision\src\lare.py preds_full
python experiments\decision\src\extras.py
python experiments\decision\src\verify.py          # 逐项复算，退出码 0 = 全过
C:\...\Python312\python.exe experiments\decision\src\figures.py
```
`verify.py` 只读交付物（`preds_test.parquet` + `metrics.json` + `lare.json` + `extras.json`），不重训模型：逐项复算 AUROC/AUPRC/Brier/ECE/校准斜率/预算灵敏度/代价/CI 点估计/DeLong/LARE 阈值下的 yield·cost·Jaccard/消融与分小时的 AUPRC，容差 1e-9（CI 用同 seed 重抽，2e-4 内）。

## 9. 没做 / 边界

- `--suite old` 只跑 B0/B4（旧仓无体征层时的早期对照），最终数字一律以 `--suite full` 为准。
- 微生物、转入转出链（`micro`/`transfers`）没进特征：与 12h 死亡关系弱且易泄露，时间不够，标"未跑"。
- 血管活性药/机械通气仅来自 `procedureevents`（40,812 行），可能低估真实使用率；`inputevents` 的升压药速率已进 `vaso_*`。
- 代价参数（漏:误=4:1、复核 0.2）是假设，换参数会让最优阈值平移；DCA 曲线给出了 1%–20% 全区间的净收益，便于按各中心偏好重选阈值。
- 所有阈值/校准器都在 val 上定，test 只评一次；没有做跨中心的 external validation。
