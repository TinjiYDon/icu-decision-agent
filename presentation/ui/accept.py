"""Acceptance / metrics secondary page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from application.acceptance import load_metrics_artifact, layer1_counts
from presentation.ui.charts import fig_calibration
from presentation.ui.theme import disclaimer

STATUS = Path(__file__).resolve().parents[2] / "docs" / "STATUS.md"


def render_accept() -> None:
    st.title("Acceptance gates")
    st.caption("Layer1 dump counts + model artifact metrics")

    try:
        counts = layer1_counts()
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("feat_rows", f"{counts['feat_rows']:,}")
        g2.metric("label_rows", f"{counts['label_rows']:,}")
        g3.metric("stays", f"{counts['stay_count']:,}")
        g4.metric("gate", counts["status"])
        st.json({"by_hour": counts["by_hour"], "expected_hours": counts["expected_hours"]})
        if counts["gate_ok"]:
            st.success("S2 row-count gate passed (~472k × 5 hours)")
        else:
            st.error("Gate failed — restore icu_decision_S2-full_*20260802.dump")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Layer1 query failed: {exc}")

    metrics = load_metrics_artifact()
    if not metrics:
        st.warning("Missing artifacts/models/metrics_mortality_12h.json")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PR-AUC test", f"{metrics.get('pr_auc_test', 0):.3f}")
        m2.metric("Brier test", f"{metrics.get('brier_test', 0):.3f}")
        m3.metric("ROC test (ref)", f"{metrics.get('auc_test', 0):.3f}")
        m4.metric("Operating thr", f"{metrics.get('operating_threshold', 0):.3f}")
        by_h = metrics.get("metrics_by_hour_test") or {}
        if by_h:
            rows = []
            for h, mm in sorted(by_h.items(), key=lambda x: int(x[0])):
                rows.append(
                    {
                        "h": int(h),
                        "roc_auc": mm.get("roc_auc"),
                        "pr_auc": mm.get("pr_auc"),
                        "brier": mm.get("brier"),
                        "precision": mm.get("precision"),
                        "recall": mm.get("recall"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        op = (metrics.get("metrics_at_val_threshold") or {}).get("test") or {}
        if op.get("calibration"):
            cal = op["calibration"]
            pred = list(cal.get("mean_predicted") or [])
            actual = list(cal.get("fraction_positive") or cal.get("mean_actual") or [])
            n = min(len(pred), len(actual))
            if n > 0:
                st.plotly_chart(
                    fig_calibration(pred[:n], actual[:n]),
                    use_container_width=True,
                )
        if op.get("confusion_matrix"):
            st.subheader("Confusion @ operating point")
            st.json(op["confusion_matrix"])

    with st.expander("STATUS.md"):
        if STATUS.exists():
            st.markdown(STATUS.read_text(encoding="utf-8"))
        else:
            st.info("No STATUS.md")
    disclaimer()
