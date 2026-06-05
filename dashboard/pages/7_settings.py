"""Settings — view ICP configs and app configuration."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dashboard.lib.bootstrap import bootstrap; bootstrap()  # noqa: E702

import streamlit as st

from dashboard.lib.data_access import load_icp_configs

st.title("⚙️ Settings")

# --- ICP configurations ---
st.subheader("ICP Configurations")
try:
    icps = load_icp_configs()
    segment = st.selectbox("View ICP for", list(icps.keys()))
    st.json(icps[segment])
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load ICP configs: {exc}")

st.caption(
    "Edit ICPs by modifying config/icp_configs.json in the repo and pushing. "
    "Changes apply on the next pipeline run."
)

st.divider()

# --- App configuration ---
st.subheader("App Configuration")
try:
    from config.settings import get_settings
    s = get_settings()
    st.code(
        f"""SQLite path:           {s.SQLITE_PATH}
Sheet ID:              {s.SHEET_ID}
Telegram empty digest: {s.TELEGRAM_SEND_EMPTY_DIGEST}
Daily lead target:     {s.DAILY_LEAD_TARGET_PER_SEGMENT}
Gemini auth mode:      {s.gemini_auth_mode}"""
    )
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load settings: {exc}")

st.divider()

# --- API budgets ---
st.subheader("API Budgets")
try:
    import pandas as pd
    from dashboard.lib.data_access import get_api_usage_df

    usage = get_api_usage_df(days=30)
    if usage.empty:
        st.caption("No API usage recorded in the last 30 days.")
    else:
        st.dataframe(usage, use_container_width=True, hide_index=True)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load API usage: {exc}")
