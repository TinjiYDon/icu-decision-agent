# S1 / S2 早期预警契约

## S1（单时刻）

- `t = intime + 1h`，`hour_index=1`
- 已并入 S2 网格中的 h=1 切片

## S2（多时刻流式 · 当前主叙事）

| 项 | 值 |
|----|-----|
| 网格 | `prediction_hours: [0,1,2,4,6]` |
| 特征 | `charttime < intime+h` |
| 标签 | 死亡 ∈ `[intime+h, intime+h+12h]` |
| 划分 | **stay 级**（同一 stay 所有 h 同 fold） |
| 行数 | ≈ 94,458 × 5 |

裸分支 `liujiawei` 不合；特征 SQL 已移植并参数化 `window_hours`。
