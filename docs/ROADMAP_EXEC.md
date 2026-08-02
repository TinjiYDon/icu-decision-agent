# 执行路线图 ROADMAP_EXEC

> 更新：2026-08-02 · **S1 已训出多指标** · 可演进 S2

## 人读摘要

| Wave | 含义 | 状态 |
|------|------|------|
| 1–2 | 6-feat 弱基线对照 | ✅ |
| **S1** | `t=intime+1h` + 扩展特征 | ✅ PR-AUC≈0.15 · ROC≈0.77 |
| **S2** | 多时刻流式样本 | 未开（offset 契约已定） |

## Agent 上下文

```text
契约：hour_index=1 · prediction_offset_hours=1
训练：Layer0 @ :5432 · python -m application.train
主指标：PR-AUC / Brier / 工作点；ROC 对照
禁止：单报 ROC；merge 裸 liujiawei
```

## 清单

- [x] S1 特征/标签重建 + STATUS
- [x] 对照 Wave2 写入 STATUS
- [x] S1 [PR #7](https://github.com/TinjiYDon/icu-decision-agent/pull/7) merge
- [ ] S2 h 网格（后续）
