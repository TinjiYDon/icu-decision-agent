"""ICU 预警监测台 — Streamlit + Plotly。"""

from __future__ import annotations

import streamlit as st

from presentation.ui.accept import render_accept
from presentation.ui.monitor import render_monitor
from presentation.ui.theme import apply_theme
from presentation.ui.tune import render_tune

st.set_page_config(
    page_title="ICU 早期恶化预警",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

monitor_page = st.Page(render_monitor, title="监测", icon=":material/monitor_heart:", default=True)
tune_page = st.Page(render_tune, title="调参", icon=":material/tune:")
accept_page = st.Page(render_accept, title="验收", icon=":material/verified:")

nav = st.navigation([monitor_page, tune_page, accept_page])
nav.run()
