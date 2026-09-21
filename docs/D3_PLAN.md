# D3 GRU-D 真序列最小对照 · 工程方案

## 问题定义

| 维度 | 当前状态 | D3 目标 |
|------|---------|---------|
| LGBM | 472K 行聚合特征（6h 均值/最大值） | 已有，不改动 |
| GRU-D | 已训，但只输出整体 ROC-AUC | **刻画 6h 序列非空率，在可训子集上对照 LGBM** |
| 对照方式 | 只看 metrics JSON 数值 | **在同一 test 子集上跑两个模型，用配对检验判断差异是否显著** |
| 负结果 | 无明确处理 | **"不显著劣于 LGBM" 即验收通过** |

---

## 架构：高内聚低耦合设计

```
domain/models/temporal/          # 已有，不改
├── grud_model.py               # 已有：GRU-D 模型
├── grud.py                     # 已有：numpy smoke
├── attribution.py              # 已有：gradient×Input
├── train_grud.py               # 已有：完整训练管线（保留不动）
│
application/                     # 已有，不改
├── train_grud.py               # CLI 入口（保留不动）
└── compare_dual_track.py       # 已有：JSON 对比（保留不动）
│
scripts/                         # D3 新增
└── d3_grud_minimal_compare.py  # CLI：单次运行完成全部 D3 验收

tests/                           # D3 新增
└── test_d3_grud_compare.py     # 单元测试：覆盖率、显著性检验、边界
```

### 新增模块：`scripts/d3_grud_minimal_compare.py`

**职责**：唯一入口，按顺序调用三个内部函数，零外部依赖。

```python
def run(lookback_hours: int = 6, min_cells: int = 3) -> D3Report:
    """端到端 D3 验收：刻画非空率 → 在可训子集上对照 LGBM → 输出报告"""
    
    # 1. 序列非空率刻画
    profile = profile_nonnull_rates(lookback_hours)
    
    # 2. 可训子集上 GRU-D 训练 + 预测
    grud_result = train_and_predict_grud(profile.keepable_stay_ids, lookback_hours)
    
    # 3. 同子集同 test split 上 LGBM 预测（复用已有预测接口）
    lgbm_result = predict_lgbm_on_subset(profile.keepable_stay_ids)
    
    # 4. 配对比较
    comparison = paired_compare(grud_result, lgbm_result)
    
    return D3Report(profile, grud_result, lgbm_result, comparison)
```

---

## 四个核心类（单一职责）

### Class 1: `SequenceProfile` — 不可变数据容器
```python
@dataclass
class SequenceProfile:
    """6h 序列质量画像（不可变，只读）。"""
    n_total_stays: int           # 全库 stay 数
    n_keepable: int              # 通过 sparse gate 的 stay 数
    nonnull_rate_by_feature: dict[str, float]   # 每个特征的平均非空率
    nonnull_rate_by_stay: list[float]           # 每个 stay 的非空率（用于分布图）
    keepable_stay_ids: list[int]  # 可训子集的 stay_id 列表
    drop_reasons: dict[str, int]  # 被 drop 的原因计数
```

### Class 2: `GRUDResult` — 单模型结果容器
```python
@dataclass
class GRUDResult:
    stay_ids: list[int]
    y_true: np.ndarray       # 二进制标签
    y_prob: np.ndarray       # 预测概率
    split: dict[str, list[int]]  # {"train": [...], "val": [...], "test": [...]}
    test_stay_ids: list[int]
    test_y_true: np.ndarray
    test_y_prob: np.ndarray
    roc_auc: float
    pr_auc: float
    n_test: int
    n_train: int
```

### Class 3: `PairedComparison` — 统计比较结果
```python
@dataclass
class PairedComparison:
    """在相同 test 子集上的配对比较。"""
    grud_roc_auc: float
    lgbm_roc_auc: float
    auc_diff: float            # grud - lgbm
    p_value_debacka: float | None  # DeLong 检验 p 值（是否有显著差异）
    is_not_worse: bool        # D3 主验收条件：GRU-D 不显著劣于 LGBM
    conclusion: str           # 一句话结论
    notes: list[str]          # 附加说明
```

### Class 4: `D3Report` — 最终输出
```python
@dataclass
class D3Report:
    profile: SequenceProfile
    grud: GRUDResult
    lgbm: GRUDResult          # 结构同 GRUDResult，方便配对比较
    comparison: PairedComparison
    artifact_path: Path       # 保存路径
```

---

## 关键设计决策

### 1. 为什么不对比整体 AUC，只对比特定 split？
- 全局 AUC 受样本量和分布影响大，无法反映"时序信息实际贡献"
- 在 GRU-D 能训的子集上跑 LGBM，保证两者在完全相同的输入分布上比较
- 如果 LGBM 在这个子集上 AUC 和全局差不多，说明表格特征没有损失时序信息；
  如果 LGBM 显著下降，说明时序确实有价值

### 2. 稀疏门控阈值
- 沿用现有 `passes_sparse_gate()` 参数：`min_observed_cells=3`, `min_obs_ratio=0.05`
- 额外报告：如果调高阈值到 5 或 8，keepable 比例如何变化（敏感性分析）

### 3. 显著性检验
- DeLong 检验（`sklearn.metrics.roc_auc_score` 的配对版本）：判断 AUC 差异是否显著
- 同时报告 McNemar 检验（在相同阈值下的分类结果是否不同）
- p < 0.05 且有方向 → 结论"GRU-D 显著优于/劣于 LGBM"
- p ≥ 0.05 → 结论"无显著差异，GRU-D 不提供增量价值"

### 4. 与现有代码的关系
- **不改** `domain/models/temporal/train_grud.py`（已有完整训练）
- **不改** `domain/features/sequence_build.py`（已有 mask/delta 逻辑）
- **不改** `domain/features/sequence_etl.py`（已有 sparse gate）
- 新代码只调用现有模块，不重复实现

---

## 验收标准

| 场景 | 结论 | D3 是否通过 |
|------|------|------------|
| GRU-D AUC > LGBM AUC，且 p < 0.05 | 时序信息确实有价值 | ✅ |
| GRU-D AUC ≈ LGBM AUC，p ≥ 0.05 | 时序信息无增量（负结果） | ✅ |
| GRU-D AUC < LGBM AUC，p < 0.05 | 时序信息有害（需排查） | ⚠️ 需报告 |
| keepable 比例 < 10% | 数据不足，无法得出结论 | ⚠️ 需报告 |

---

## 输出物

1. `artifacts/d3/nonnull_profile.json` — 非空率画像
2. `artifacts/d3/comparison.json` — 配对比较结果
3. `artifacts/d3/report.md` — Markdown 报告（含图表）
4. Streamlit "调参" 页新增"时序对照"tab（可选，后续）

---

## 文件清单

| 新增文件 | 行数估计 | 说明 |
|---------|---------|------|
| `scripts/d3_grud_minimal_compare.py` | ~250 行 | CLI 入口 + 4 个数据类 + 主要逻辑 |
| `tests/test_d3_grud_compare.py` | ~150 行 | 5 个测试用例 |
| `docs/D3_PLAN.md` | 本文件 | 方案文档 |
| `artifacts/d3/` | — | 输出目录（gitignore） |
