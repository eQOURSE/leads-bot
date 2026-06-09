"""Phase 10 — Streamlit dashboard entry point.

Modern, non-technical-friendly UI for browsing leads, marking replies, and
triggering pipeline runs. Reads from Google Sheets (the shared source of truth
between the GitHub Actions cron and this dashboard).

Run locally:
    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on sys.path so `from dashboard.lib...` resolves on Streamlit Cloud.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from dashboard.lib import ui  # noqa: E402
from dashboard.lib.auth import check_password  # noqa: E402

st.set_page_config(
    page_title="eQOURSE Lead Generation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject_css()

if not check_password():
    st.stop()

# ---- Authenticated: build navigation ----
pages = [
    st.Page("views/overview.py", title="Overview", icon="📊", default=True),
    st.Page("views/lead_browser.py", title="Lead Browser", icon="🔍"),
    st.Page("views/needs_review.py", title="Needs Review", icon="⚠️"),
    st.Page("views/manual_lookup.py", title="Manual Lookup", icon="🔎"),
    st.Page("views/run_history.py", title="Run History", icon="📜"),
    st.Page("views/run_now.py", title="Run Now", icon="▶️"),
    st.Page("views/settings.py", title="Settings", icon="⚙️"),
]

with st.sidebar:
    ui.sidebar_brand()
    st.divider()

nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.divider()
    last = st.session_state.get("last_refresh", "—")
    st.caption(f"Last refreshed: {last}")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        from datetime import datetime
        st.session_state["last_refresh"] = datetime.now().strftime("%H:%M:%S")
        st.rerun()
    if st.button("🔒 Log out", use_container_width=True):
        st.session_state["authed"] = False
        st.rerun()

nav.run()
