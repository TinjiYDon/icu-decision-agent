"""给老师看的创新口径：底座 vs 切口。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from presentation.ui.theme import disclaimer

ROOT = Path(__file__).resolve().parents[2]
D3_REPORT = ROOT / "artifacts" / "d3" / "report.md"
TFT_JSON = ROOT / "artifacts" / "d3" / "tft_lite.json"
LAB_TRIPLE = ROOT / "artifacts" / "d3" / "lab_triple_compare.json"


def render_innovation() -> None:
    st.title("创新切口（不是换一个时髦网络）")
    st.caption("默认推理仍是 LightGBM；答辩请讲 H1–H3，不要把树模型品牌当贡献。")

    st.header("底座 vs 切口")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "层": "底座",
                    "是什么": "MIMIC + 无泄漏特征 + LightGBM",
                    "对老师怎么说": "可复现的表格预警骨干，不是论文卖点",
                },
                {
                    "层": "H1 标签与决策价值",
                    "是什么": "deathtime 标签 v2 + DCA",
                    "对老师怎么说": "更严的死亡时间；主报 PR-AUC/Brier，不刷 ROC",
                },
                {
                    "层": "H2 公平双轨",
                    "是什么": "GRU-D vs 表格，同 h / split / stay",
                    "对老师怎么说": "时序有没有增量，看 PR-AUC/Brier 差，不另训另测",
                },
                {
                    "层": "TFT-lite 消融",
                    "是什么": "同输入张量上的时间注意力编码器",
                    "对老师怎么说": "不是 Lim 2021 全文 TFT，也不是默认模型",
                },
                {
                    "层": "H3 解释可信",
                    "是什么": "SHAP→RAG→LLM + 引用校验 + HITL",
                    "对老师怎么说": "深度在可追溯与人可改，不在换大模型",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.header("主模型已发布数字（test）")
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC（对照）", "0.776")
    c2.metric("PR-AUC（主）", "0.092")
    c3.metric("Brier（主）", "0.151")
    st.caption("阳性更稀（约 1.19%）后 PR-AUC 下降是更严，不是退步借口去改回旧 dod。")

    st.header("DX-3 报告")
    if D3_REPORT.is_file():
        st.markdown(D3_REPORT.read_text(encoding="utf-8")[:8000])
    else:
        st.info(
            "`artifacts/d3/report.md` 未入库。本机生成："
            "`python scripts/d3_grud_minimal_compare.py --lookback 6 --real`"
            "（或 `--mock` 冒烟）。STATUS 里若只写了 ROC，答辩仍以报告主表 PR-AUC/Brier 为准。"
        )
    st.header("TFT-lite 烟测")
    if TFT_JSON.is_file():
        import json

        st.json(json.loads(TFT_JSON.read_text(encoding="utf-8")))
    else:
        st.caption("运行 `python scripts/d3_tft_lite_smoke.py` 生成 artifacts/d3/tft_lite.json")
    st.header("化验时序三方对照（深化）")
    if LAB_TRIPLE.is_file():
        import json

        st.json(json.loads(LAB_TRIPLE.read_text(encoding="utf-8")))
        st.caption(
            "LGBM vs GRU-D vs TFT-lite · labs-only · 主看 PR-AUC；"
            "阳/阴混合抽样便于对照，不是自然患病率。默认床旁仍是 LightGBM。"
        )
    else:
        st.caption(
            "运行 `python scripts/d3_lab_triple_compare.py --limit 400` "
            "生成 artifacts/d3/lab_triple_compare.json"
        )
    disclaimer()
