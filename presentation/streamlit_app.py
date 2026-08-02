"""ICU decision clinical console — Streamlit + Plotly (CDSS-style)."""

from __future__ import annotations

import streamlit as st

from presentation.ui.accept import render_accept
from presentation.ui.monitor import render_monitor
from presentation.ui.theme import apply_theme
from presentation.ui.tune import render_tune

st.set_page_config(
    page_title="ICU Decision Monitor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

monitor_page = st.Page(render_monitor, title="Monitor", icon=":material/monitor_heart:", default=True)
tune_page = st.Page(render_tune, title="Tune", icon=":material/tune:")
accept_page = st.Page(render_accept, title="Accept", icon=":material/verified:")

nav = st.navigation([monitor_page, tune_page, accept_page])
nav.run()
