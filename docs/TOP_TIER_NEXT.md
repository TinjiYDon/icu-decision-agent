# 本仓顶尖视角下一步（decision）

> 2026-09-13 · **独立项目** · AIGC 人机协同 CDSS  
> 不与 scheduling 硬耦合；调度侧文档见对方仓库。

## 总原则

1. **决策价值 > 排行榜 AUC**：主报 PR-AUC、Brier、校准、决策曲线/净受益；ROC 仅对照。
2. **时间因果硬约束**：特征 `charttime < t`；标签窗口 `[t, t+H]`。
3. **人机闭环可审计**：分数 → 档位 → 建议 → 采纳/驳回 → care_plan。
4. **本仓独立交付**：不调用床位优化 API。

## vNext 对齐 ROADMAP Wave D0–D3

| 优先级 | 项 | 建议落地 |
|--------|----|----------|
| P0 | label v2 dump 文档 | `DUMP_READY.md` 标明磁盘 dump 版本 |
| P0 | GRU-D（PR#11） | 合入或拆合；同 split 对照 |
| P1 | 真序列 ETL | Layer0 chart/lab → sequence |
| P1 | HITL | Streamlit 采纳/驳回 + audit log |
| P1 | care_plan | 仓内结构化处置建议（非床位） |
| P2 | MCP `explain_risk` | 仅本仓 |

## 非目标

- 接 scheduling `optimize_beds`
- 用 GRU-D 替换 TreeSHAP→RAG→LLM 主链
- 宣称临床生产上线
