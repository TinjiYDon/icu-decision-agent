"""方法 — 预警项目正式方法说明。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from application.acceptance import load_metrics_artifact
from presentation.ui.theme import disclaimer


def render_methods() -> None:
    st.title("方法")
    st.caption("问题定义 · 数据 · 流程 · 模型与损失 · 评价指标")

    st.header("1. 问题定义")
    st.markdown(
        "在 ICU 入科后的指定时刻，利用当时已可获得的临床特征，"
        "估计患者在未来 12 小时内死亡的概率，并为床旁决策提供可解释的风险分层建议。"
    )
    st.latex(r"t = \mathrm{intime} + h,\quad h \in \{0,1,2,4,6\}")
    st.latex(r"Y = \mathbf{1}\{\text{death in }[t,\ t+12\mathrm{h}]\}")
    st.latex(r"p = \Pr(Y=1\mid \mathbf{x}_t)")

    st.header("2. 数据")
    metrics = load_metrics_artifact() or {}
    total_n = int(metrics.get("total_n") or 472290)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "数据集": "全量 S2 样本",
                    "存储位置": "feat.sample_matrix / label.mortality_12h",
                    "规模": f"{total_n:,} 行 · 94,458 stays",
                    "含义": "每个 (住院, 预测时刻) 一条样本",
                },
                {
                    "数据集": "训练集",
                    "存储位置": "按 stay_id 划分（内存）",
                    "规模": "约 70% stays",
                    "含义": "模型拟合",
                },
                {
                    "数据集": "验证集",
                    "存储位置": "按 stay_id 划分（内存）",
                    "规模": "约 10% stays",
                    "含义": "早停与工作点阈值选择",
                },
                {
                    "数据集": "测试集",
                    "存储位置": "metrics_mortality_12h.json",
                    "规模": "约 20% stays",
                    "含义": "最终报告 PR-AUC / Brier / 工作点",
                },
                {
                    "数据集": "源数据 Layer0",
                    "存储位置": "MIMIC-IV（PostgreSQL）",
                    "规模": "全库只读",
                    "含义": "原始住院、化验与监护记录",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    if metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("样本量", f"{total_n:,}")
        c2.metric("测试集 PR-AUC", f"{float(metrics.get('pr_auc_test') or 0):.3f}")
        c3.metric("测试集 Brier", f"{float(metrics.get('brier_test') or 0):.3f}")

    st.header("3. 数据处理流程")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR;
          node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#0f766e"];
          A [label="MIMIC-IV"];
          B [label="特征构建\\n(无时间泄漏)"];
          C [label="12h 死亡标签"];
          D [label="样本矩阵"];
          E [label="Stay 级划分\\n70/10/20"];
          F [label="LightGBM"];
          G [label="评价与解释"];
          A -> B -> D; B -> C -> D; D -> E -> F -> G;
        }
        """
    )
    st.markdown(
        "- **特征时间**：仅使用 `charttime < intime + h` 的信息。  \n"
        "- **标签窗口**：`[intime + h, intime + h + 12h]` 内是否死亡。  \n"
        "- **划分原则**：同一住院的全部时刻属于同一折，避免泄漏。"
    )

    st.header("4. 模型与损失函数")
    st.markdown(
        "采用 **LightGBM** 梯度提升树做二分类概率估计。"
        "默认超参：`n_estimators=64`，`max_depth=4`，`learning_rate=0.1`；"
        "并以 `scale_pos_weight = N_{neg}/N_{pos}` 处理类别不平衡。"
    )
    st.markdown("**二元交叉熵（logistic loss）**")
    st.latex(r"p=\sigma(z)=\frac{1}{1+e^{-z}}")
    st.latex(r"\mathcal{L}(y,p)=-\big[y\log p+(1-y)\log(1-p)\big]")
    st.latex(r"\mathcal{L}_{\mathrm{batch}}=\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(y_i,p_i)")
    st.markdown(
        "**设计要点**：输出为可校准概率，便于映射观察 / 复查 / 加强监护等临床档位；"
        "在阳性率约 2% 的设定下，主评价指标采用 **PR-AUC** 与 **Brier score**，"
        "ROC-AUC 仅作对照。工作点阈值在验证集按 F1 选定后固定到测试集。"
    )

    st.header("5. 可解释性")
    st.markdown(
        "使用 TreeSHAP 给出单次预测的特征贡献。"
        "监测台展示风险概率、建议档位、多时刻轨迹与主要贡献因素。"
    )
    disclaimer()
