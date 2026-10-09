"""给老师看的创新口径：总目标 → 底座 vs 切口。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from presentation.ui.theme import disclaimer

ROOT = Path(__file__).resolve().parents[2]
D3_REPORT = ROOT / "artifacts" / "d3" / "report.md"
TFT_JSON = ROOT / "artifacts" / "d3" / "tft_lite.json"
LAB_TRIPLE = ROOT / "artifacts" / "d3" / "lab_triple_compare.json"
H4_JSON = ROOT / "artifacts" / "h4" / "utility_compare.json"
H5_JSON = ROOT / "artifacts" / "h5" / "sparsity_fusion.json"
MASTER = ROOT / "docs" / "MASTER_NARRATIVE.md"


def render_innovation() -> None:
    st.title("总目标与创新切口")
    st.markdown(
        "**总目标**：在时间因果与 stay 公平协议下，做出"
        "**可决策、可校准效用、可融合、可解释、可人审**的早期预警 CDSS——"
        "创新在机制，不在换一个时髦网络。"
    )
    st.caption(
        "默认推理仍是 LightGBM；答辩请按 H1–H5 讲。"
        "叙事 docs/MASTER_NARRATIVE.md · 公式 docs/ARCHITECTURE_MATH.md。"
    )

    st.header("闭环串线（总内容）")
    st.code(
        "数据因果 → 骨干打分 → H4效用阈值 → H5稀疏融合 → H3解释人审 → PR-AUC/Brier/NB验收",
        language="text",
    )
    st.latex(
        r"p_{\mathrm{fuse}}=(1-c^{\gamma})\,p^{\mathrm{cur}}+c^{\gamma}\,p^{\mathrm{traj}}"
    )
    st.latex(
        r"\tau_{\mathrm{NB}}=\arg\max_{\tau\in[\tau_{\min},\tau_{\max}]\,:\,\mathrm{Alert}\le\alpha}"
        r"\,\mathrm{NB}(\tau;\lambda)"
    )

    st.header("底座 vs 切口")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "层": "底座",
                    "是什么": "MIMIC + 无泄漏特征 + LightGBM",
                    "对老师怎么说": "可复现骨干，不是卖点",
                },
                {
                    "层": "H1 决策评价",
                    "是什么": "DCA / 净受益（标签诚实另表）",
                    "对老师怎么说": "主报 PR-AUC/Brier/NB，不刷 ROC",
                },
                {
                    "层": "H4 效用阈值",
                    "是什么": "val 上约束 max-NB（非 max-F1）",
                    "对老师怎么说": "训练—阈值环对准临床代价",
                },
                {
                    "层": "H2 / H5 融合",
                    "是什么": "公平双轨 + 稀疏门控多时刻",
                    "对老师怎么说": "缺测时信谁，可证伪；无 Layer0 时轨迹=多时刻代理",
                },
                {
                    "层": "H3 解释可信",
                    "是什么": "SHAP→RAG→LLM + HITL",
                    "对老师怎么说": "可追溯、可驳回",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.header("H4 效用校准（旧 S2）")
    if H4_JSON.is_file():
        h4 = json.loads(H4_JSON.read_text(encoding="utf-8"))
        f1 = h4["f1_policy"]["test"]
        ut = h4["utility_policy"]["test"]
        c1, c2, c3 = st.columns(3)
        c1.metric("F1 策略 NB", f"{f1['net_benefit']:.4f}")
        c2.metric("效用策略 NB", f"{ut['net_benefit']:.4f}")
        c3.metric("ΔNB", f"{h4['test_delta']['net_benefit_utility_minus_f1']:+.4f}")
        st.caption(
            f"thr {h4['f1_policy']['threshold']:.3f} → {h4['utility_policy']['threshold']:.3f} · "
            f"报警率 {f1['alert_rate']:.1%} → {ut['alert_rate']:.1%} · PR-AUC 不变"
        )
    else:
        st.info("运行 `python scripts/h4_utility_compare.py --cost-ratio 4`")

    st.header("H5 稀疏门控融合（旧 S2 · h=6）")
    if H5_JSON.is_file():
        h5 = json.loads(H5_JSON.read_text(encoding="utf-8"))
        rows = {r["policy"]: r for r in h5.get("rows", [])}
        cur, fuse = rows.get("current_hour", {}), rows.get("sparsity_gated", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("当前 Brier", f"{cur.get('brier', float('nan')):.4f}")
        c2.metric("门控 Brier", f"{fuse.get('brier', float('nan')):.4f}")
        sparse = (h5.get("subsets") or {}).get("sparse_half") or {}
        if sparse:
            c3.metric(
                "稀疏半 PR-AUC",
                f"{sparse.get('fused_pr_auc', float('nan')):.3f}",
                delta=f"vs {sparse.get('current_pr_auc', float('nan')):.3f}",
            )
        st.caption(h5.get("note", ""))
    else:
        st.info("运行 `python scripts/h5_sparsity_fusion.py --hour-index 6`")

    st.header("主模型对照数字（v2 表 · 勿与旧 S2 混）")
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC（对照）", "0.776")
    c2.metric("PR-AUC（主）", "0.092")
    c3.metric("Brier（主）", "0.151")
    st.caption("v2 阳性更稀；机制实验主表见旧 S2 的 H4/H5。")

    st.header("DX-3 / 时序报告")
    if D3_REPORT.is_file():
        st.markdown(D3_REPORT.read_text(encoding="utf-8")[:8000])
    else:
        st.info("`artifacts/d3/report.md` 未生成时可 `--mock` 冒烟。")
    if TFT_JSON.is_file():
        st.subheader("TFT-lite 烟测")
        st.json(json.loads(TFT_JSON.read_text(encoding="utf-8")))
    if LAB_TRIPLE.is_file():
        st.subheader("化验三方")
        st.json(json.loads(LAB_TRIPLE.read_text(encoding="utf-8")))

    if MASTER.is_file():
        with st.expander("打开总纲领 MASTER_NARRATIVE.md"):
            st.markdown(MASTER.read_text(encoding="utf-8")[:12000])
    disclaimer()
