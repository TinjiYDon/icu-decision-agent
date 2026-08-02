# mortality_12h 基线实验

> 更新：2026-07-31 · 成员 B 算法复核 · full dump 监督训练

## 数据与复现

| 项 | 值 |
|---|---|
| dump | `icu_decision_P0-full_mimic_94458stays_20260726.dump` |
| dump SHA-256 | `E2A2C6B2367B34D739A52E2A082CAF0252025508AC7BED98F3561D25E0A096CA` |
| 样本 / 阳性 | 94,458 / 2,099（2.222%） |
| 特征 | `anchor_age`、`gender_m`、`careunit_*`（共 6 个） |
| 划分 | stay_id 严格分层 · 0.7/0.1/0.2 · seed=42 |
| 模型 | LightGBM · 64 trees · max_depth=4 · learning_rate=0.1 |
| 类别权重 | 训练集 negative/positive |

```powershell
$env:PYTHONPATH = (Get-Location)
.\.venv\Scripts\python.exe -m application.train --from-existing
```

`--from-existing` 只读取 full dump 已恢复的 `feat.sample_matrix` 和
`label.mortality_12h`，不会从 Layer0 重建或覆盖数据。

## 严格分层结果

| split | n | positive | positive rate |
|---|---:|---:|---:|
| train | 66,121 | 1,469 | 2.2217% |
| val | 9,446 | 210 | 2.2232% |
| test | 18,891 | 420 | 2.2233% |

## 判别与校准

| 指标 | val | test |
|---|---:|---:|
| ROC-AUC | 0.6729 | 0.6935 |
| PR-AUC | 0.0393 | 0.0413 |
| Brier | 0.0483 | 0.0477 |
| Log loss | 0.2180 | 0.2157 |

旧记录 `auc_val=0.711 / auc_test=0.682` 来自非严格分层实现。严格分层后两组
阳性率一致，指标变化说明旧结果包含明显的划分波动，后续以本页结果为基线。

## 阈值

默认阈值 0.5 在 val/test 均预测不到阳性（recall=0），不可使用。

仅用 validation 最大化 F1 后得到阈值 `0.3458`，固定后应用于 test：

| 指标 | val | test |
|---|---:|---:|
| Precision | 5.46% | 5.34% |
| Recall | 19.52% | 18.33% |
| Specificity | 92.31% | 92.60% |
| F1 | 0.0853 | 0.0827 |
| TP / FP / FN / TN | 41 / 710 / 169 / 8,526 | 77 / 1,366 / 343 / 17,105 |

测试集没有参与阈值或超参数选择。

## 结论

- 当前 6 个入科静态特征只能形成弱基线，不能作为临床部署模型。
- PR-AUC 仅略高于 2.22% 的阳性基准，召回率不足。
- LightGBM 输出未做概率校准，现有建议档位阈值不能视为临床阈值。
- 下一步应先完成标签时间精度修正，再加入严格按预测时间截断的生命体征和化验特征。

完整机器可读结果：`artifacts/models/metrics_mortality_12h.json`。
