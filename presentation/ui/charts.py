"""Plotly charts for decision console."""

from __future__ import annotations

from typing import Any, Sequence

import plotly.graph_objects as go


TEAL = "#0f766e"
SLATE = "#334155"
AMBER = "#b45309"


def fig_risk_trajectory(points: Sequence[dict[str, Any]]) -> go.Figure:
    xs = [p["hour_index"] for p in points]
    ys = [float(p["risk_score"]) for p in points]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name="risk",
            line=dict(color=TEAL, width=3),
            marker=dict(size=10, color=TEAL),
            hovertemplate="h=%{x}<br>risk=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="多时刻风险曲线（S2 网格）",
        xaxis_title="预测时刻 hour_index（入科后小时）",
        yaxis_title="12 小时死亡风险",
        yaxis=dict(range=[0, max(0.25, max(ys) * 1.25 if ys else 0.25)]),
        margin=dict(l=40, r=20, t=48, b=40),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.9)",
        font=dict(color=SLATE),
    )
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig


def fig_shap_bars(top_factors: Sequence[dict[str, Any]]) -> go.Figure:
    names = [str(r.get("feature", "?")) for r in reversed(list(top_factors))]
    shap = [float(r.get("shap", 0)) for r in reversed(list(top_factors))]
    colors = [AMBER if v >= 0 else TEAL for v in shap]
    fig = go.Figure(
        go.Bar(
            x=shap,
            y=names,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>SHAP=%{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="主要影响因素（SHAP）",
        xaxis_title="SHAP 贡献值",
        margin=dict(l=40, r=20, t=48, b=40),
        height=max(280, 48 * len(names) + 80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.9)",
        font=dict(color=SLATE),
    )
    fig.update_xaxes(gridcolor="#e2e8f0", zeroline=True, zerolinecolor="#94a3b8")
    return fig


def fig_calibration(mean_predicted: Sequence[float], fraction_positive: Sequence[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="perfect",
            line=dict(color="#94a3b8", dash="dash", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(mean_predicted),
            y=list(fraction_positive),
            mode="lines+markers",
            name="model",
            line=dict(color=TEAL, width=2),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        title="校准曲线（测试集 · 工作点）",
        xaxis_title="平均预测概率",
        yaxis_title="实际阳性比例",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=320,
        margin=dict(l=40, r=20, t=48, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.9)",
        font=dict(color=SLATE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def fig_hour_pr_auc(by_hour: dict[str, Any]) -> go.Figure:
    hours = sorted(by_hour.keys(), key=lambda x: int(x))
    pr = [float((by_hour[h] or {}).get("pr_auc") or 0) for h in hours]
    fig = go.Figure(
        go.Bar(
            x=[f"h={h}" for h in hours],
            y=pr,
            marker_color=TEAL,
            hovertemplate="%{x}<br>PR-AUC=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="测试集 · 按预测时刻 PR-AUC",
        yaxis_title="PR-AUC",
        height=300,
        margin=dict(l=40, r=20, t=48, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.9)",
        font=dict(color=SLATE),
    )
    return fig


def fig_net_benefit(curve: dict[str, Any]) -> go.Figure:
    """DCA curve with point estimates only (backward compat)."""
    return fig_net_benefit_with_ci(curve, show_ci=False)


def fig_net_benefit_with_ci(curve: dict[str, Any], *, show_ci: bool = True) -> go.Figure:
    """DCA curve with optional bootstrap confidence interval band."""
    thr = curve.get("thresholds") or []
    nb_model = curve.get("net_benefit_model") or []
    nb_all = curve.get("net_benefit_treat_all") or []
    ci_lo = curve.get("ci_lower")
    ci_hi = curve.get("ci_upper")

    fig = go.Figure()
    # CI band
    if show_ci and ci_lo is not None and ci_hi is not None and len(thr) == len(ci_lo) == len(ci_hi):
        # Upper trace (forward)
        fig.add_trace(go.Scatter(
            x=thr + list(reversed(thr)),
            y=ci_hi + list(reversed(ci_lo)),
            fill="toself", fillcolor="rgba(15,118,110,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False, name="95% CI",
            hoverinfo="skip",
        ))
    # Model NB
    fig.add_trace(go.Scatter(
        x=thr, y=nb_model,
        mode="lines+markers", name="模型",
        line=dict(color=TEAL, width=3),
        marker=dict(size=8),
    ))
    # Treat-all
    fig.add_trace(go.Scatter(
        x=thr, y=nb_all,
        mode="lines", name="全部干预",
        line=dict(color=AMBER, dash="dash"),
    ))
    # Treat-none
    fig.add_trace(go.Scatter(
        x=thr, y=[0.0] * len(thr),
        mode="lines", name="全不干预",
        line=dict(color="#94a3b8", dash="dot"),
    ))
    # Mark working point if present
    wp_thr = curve.get("working_threshold")
    if wp_thr is not None and thr and nb_model:
        idx = int(round((wp_thr - thr[0]) / (thr[-1] - thr[0]) * (len(thr) - 1)))
        idx = max(0, min(idx, len(thr) - 1))
        fig.add_trace(go.Scatter(
            x=[thr[idx]], y=[nb_model[idx]],
            mode="markers", name="工作点",
            marker=dict(size=14, color="red", symbol="diamond"),
        ))
    fig.update_layout(
        title="决策曲线（净受益 Net Benefit）" + (" · 含 Bootstrap 95% CI" if show_ci else ""),
        xaxis_title="阈值概率 (p_t)",
        yaxis_title="净受益",
        height=400 if show_ci else 360,
        margin=dict(l=40, r=20, t=48, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.9)",
        font=dict(color=SLATE),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def fig_dual_encoded_heatmap(
    values: list[list[float]],
    attributions: list[list[float]],
    feature_names: list[str],
    time_labels: list[str],
) -> go.Figure:
    """Dual-encoded heatmap: time×feature grid.

    X-axis: time steps (30-min intervals)
    Y-axis: physiological features
    Color: Per-feature normalized signed attribution — red=increases risk, blue=decreases risk
    Cell text: actual feature value at that timestep (or "—" if missing)
    """
    import numpy as np

    T = len(time_labels)
    F = len(feature_names)
    attr_arr = np.array(attributions) if (attributions is not None and len(attributions) > 0) else np.zeros((T, F))
    val_arr = np.array(values) if (values is not None and len(values) > 0) else np.zeros((T, F))

    # Drop padded (negative-time) columns — they carry no observations by
    # construction (pad_truncate zero-fills the head), so hiding them makes
    # the chart denser without losing any information.
    keep = [i for i, lbl in enumerate(time_labels) if not str(lbl).startswith("pad")]
    if keep and len(keep) < T:
        time_labels = [time_labels[i] for i in keep]
        val_arr = val_arr[keep, :]
        attr_arr = attr_arr[keep, :]
        T = len(keep)

    # Amplify tiny gradients caused by RNN vanishing-gradient over long sequences.
    # This is a visualization-only scaling; it does NOT change the underlying model.
    ATTR_AMPLIFY = 100000.0
    attr_arr = attr_arr * ATTR_AMPLIFY

    # Normalize each feature independently to [-1, 1] so colors are vivid
    normalized = np.zeros_like(attr_arr, dtype=float)
    for f_idx in range(F):
        col = attr_arr[:, f_idx]
        col_abs_max = np.max(np.abs(col))
        if col_abs_max > 1e-12:
            normalized[:, f_idx] = col / col_abs_max

    # Build text grid: show actual value or "—" for missing
    text_vals = []
    for t in range(T):
        row = []
        for f in range(F):
            v = val_arr[t, f]
            # Handle both numeric and string values
            if isinstance(v, str):
                row.append(v)
            elif np.isnan(v) or abs(float(v)) < 1e-6:
                row.append("—")
            else:
                row.append(f"{float(v):.1f}")
        text_vals.append(row)

    # Plotly Heatmap expects z shaped (len(y), len(x)) = (F features, T timesteps).
    # Our matrices are (T, F), so transpose both z and text.
    z_display = normalized.T.tolist()
    text_display = [list(col) for col in zip(*text_vals)]  # transpose (T,F) -> (F,T)

    fig = go.Figure(
        go.Heatmap(
            z=z_display,
            x=time_labels,
            y=feature_names,
            colorscale=[
                [0.0,  "#1565c0"],   # deep blue — max protective
                [0.25, "#64b5f6"],   # light blue
                [0.5,  "#ffffff"],   # white — neutral
                [0.75, "#ff8a65"],   # light red
                [1.0,  "#c62828"],   # deep red — max risk
            ],
            zmin=-1.0,
            zmax=1.0,
            text=text_display,
            texttemplate="%{text}",
            textfont=dict(size=10, color="#1e293b"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "时间: %{x}<br>"
                "归因值: %{z:.3f}<br>"
                "实际值: %{text}<extra></extra>"
            ),
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="归因强度<br>(红↑风险 蓝↓保护)",
                    side="right",
                    font=dict(size=11),
                ),
                tickfont=dict(size=9),
                len=0.55,
                thickness=14,
                tickvals=[-1.0, 0.0, 1.0],
                ticktext=["保护", "中性", "风险"],
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="双编码时序归因热力图（Xian et al., IJMI 2026）<br>"
                  "<sub>颜色=逐特征归一化归因 | 数值=生理变量实际值 | 时间轴=入科后30分钟间隔</sub>",
            font=dict(size=14),
        ),
        xaxis_title="时间（入科后，30min/步）",
        yaxis_title="生理特征",
        xaxis=dict(tickangle=0, tickfont=dict(size=8), side="top"),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        margin=dict(l=90, r=110, t=90, b=40),
        height=max(400, F * 40 + 120),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.95)",
        font=dict(color=SLATE),
        annotations=[
            go.layout.Annotation(
                x=1.02, y=1.06, xref="paper", yref="paper",
                text="▶ 红色 = 推高风险", showarrow=False,
                font=dict(size=11, color="#c62828"),
            ),
            go.layout.Annotation(
                x=1.02, y=0.90, xref="paper", yref="paper",
                text="▶ 蓝色 = 保护因素", showarrow=False,
                font=dict(size=11, color="#1565c0"),
            ),
        ],
    )
    return fig
