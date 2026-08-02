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
        title="Multi-hour risk curve (S2 grid)",
        xaxis_title="hour_index (hours after ICU intime)",
        yaxis_title="12h mortality risk",
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
        title="Top SHAP contributions",
        xaxis_title="SHAP value",
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
        title="Calibration (test @ operating threshold)",
        xaxis_title="Mean predicted",
        yaxis_title="Fraction positive",
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
