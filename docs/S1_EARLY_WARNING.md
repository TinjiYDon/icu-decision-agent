# S1 早期预警（取代「整支合 liujiawei」）

> 从 `liujiawei` **移植**特征 SQL/构建逻辑到 main；契约改为 `hour_index=1`。

## 契约

| 项 | 值 |
|----|-----|
| 预测时刻 | `t = intime + 1h` |
| 特征窗 | `charttime ≤ t`（预 ICU lab 优先） |
| 标签 | 死亡 ∈ `[t, t+12h]` |
| 训练列 | Model-A 精简集（见 `features.yaml` allowed） |
| 治疗暴露 | `vasopressor_1h` 允许，STATUS 注明 |

## S2 升级

同一 `_window_hours()` / `prediction_offset_hours`；对 h 网格循环写多行 `hour_index` 即可。

## 基线对照

Wave2：6-feat · hour_index=0 · ROC≈0.67/0.69 · PR-AUC≈0.04（见历史 STATUS）。
