"""Phase 10 — Streamlit dashboard entry point + auth gate + sidebar.

Run locally:
    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable (config, dashboard.lib, etc.)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from dashboard.lib.auth import check_password  # noqa: E402

st.set_page_config(
    page_title="eQOURSE Lead Generation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not check_password():
    st.stop()

# Sidebar
st.sidebar.title("Lead Gen Dashboard")
st.sidebar.divider()
st.sidebar.caption(f"Last refresh: {st.session_state.get('last_refresh', '—')}")
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    from datetime import datetime
    st.session_state["last_refresh"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.rerun()

# Landing content (Streamlit auto-lists pages/ in the sidebar nav).
st.title("🎯 eQOURSE Lead Generation")
st.markdown(
    """
Welcome to the lead generation dashboard. Use the sidebar to navigate:

- **Overview** — today's run, funnel, and recent activity
- **Lead Browser** — filterable list of every lead with full detail
- **Needs Review** — leads the validator flagged for human judgment
- **Manual Lookup** — companies whose domain couldn't be resolved
- **Run History** — past runs and API cost tracking
- **Run Now** — trigger the pipeline manually via GitHub Actions
- **Settings** — view ICP configs and app configuration
"""
)

st.info("Select a page from the sidebar to get started.")
