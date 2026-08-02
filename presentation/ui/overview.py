"""项目总览 — 答辩开场叙事。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from application.acceptance import load_metrics_artifact
from presentation.ui.charts import fig_hour_pr_auc
from presentation.ui.theme import disclaimer


def render_overview() -> None:
    st.title("ICU 早期恶化预警 · 项目总览")
    st.caption("实时决策支持演示 · S2 多时刻流式样本 · 主报 PR-AUC / Brier / 工作点")

    st.markdown(
        """
### 要解决什么问题
在 ICU **入科后若干小时**（`t = intime + h`），仅用当时已可得的生命体征与化验，
预测未来 **12 小时内**死亡风险，并给出可解释建议档位——服务床旁「要不要加强监护」的决策，
而不是出院后回顾 AUC。

### S2 契约（演示口径）
| 项 | 定义 |
|----|------|
| 预测网格 | `h ∈ {0,1,2,4,6}` |
| 特征时间 | `charttime < intime+h`（无泄漏） |
| 标签窗口 | `[intime+h, intime+h+12h]` |
| 划分 | stay 级分层 · seed=42 |
| 主指标 | **PR-AUC、Brier、工作点 P/R**；ROC 仅对照 |
"""
    )

    metrics = load_metrics_artifact()
    if not metrics:
        st.warning("未找到 metrics 产物。请先 `python -m application.train --from-existing`。")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("样本行数", f"{metrics.get('total_n', 0):,}")
        c2.metric("PR-AUC（测试）", f"{float(metrics.get('pr_auc_test') or 0):.3f}")
        c3.metric("Brier（测试）", f"{float(metrics.get('brier_test') or 0):.3f}")
        c4.metric("工作点阈值", f"{float(metrics.get('operating_threshold') or 0):.3f}")
        c5.metric("阳性率", f"{float(metrics.get('pos_rate') or 0):.2%}")

        by_h = metrics.get("metrics_by_hour_test") or {}
        if by_h:
            st.plotly_chart(fig_hour_pr_auc(by_h), use_container_width=True)
            rows = []
            for h, mm in sorted(by_h.items(), key=lambda x: int(x[0])):
                rows.append(
                    {
                        "h": int(h),
                        "PR-AUC": mm.get("pr_auc"),
                        "Brier": mm.get("brier"),
                        "ROC（对照）": mm.get("roc_auc"),
                        "精确率": mm.get("precision"),
                        "召回率": mm.get("recall"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown(
        """
### 演示路径
1. **监测**：选住院 / 高低风险样例 → 风险徽章 · 多时刻曲线 · SHAP  
2. **验收**：dump 行数门禁 · 校准 · 决策净受益曲线  
3. **调参**：建议阈值与 `--from-existing` 重训（可选）

详见 [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md)。
"""
    )
    disclaimer()
