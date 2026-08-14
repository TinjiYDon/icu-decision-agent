# Progress · icu-decision-agent

> 更新：2026-08-14（数据侧继续）

## 里程碑

| 里程碑 | 状态 |
|--------|------|
| PR #9 解释链 | ✅ 合入 |
| deathtime 标签路径 | ✅ 代码 + **已在 Layer0 重算并重训** |
| S2 指标（v2） | ✅ test PR-AUC≈0.092 · Brier≈0.151 · 阳性 5638/472290 |
| GRU-D 真序列训练 | 🟡 smoke 有；chart/labs 6h 窗口稀疏，完整 ETL 未完成 |
| 新 dump 导出 | ⬜ 库内 v2 ≠ 磁盘旧 dump |

## 下一刀

1. `pg_dump` 导出带 v2 label 的新 Layer1 dump 并更新 `DUMP_READY.md`  
2. 优化 lab/chart 序列抽取（索引/批量）后再训 PyTorch GRU-D  
3. 校准/工作点针对更低阳性率重定
