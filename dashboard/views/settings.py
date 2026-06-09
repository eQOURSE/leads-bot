"""Settings — view ICP configs and app configuration."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from dashboard.lib import ui
from dashboard.lib.data_access import get_api_usage_df, load_icp_configs, sheet_id

ui.inject_css()
ui.hero("Settings", "Review your ICP configurations and dashboard connection status.")

# ---- ICP configs ----
st.subheader("ICP configurations")
try:
    icps = load_icp_configs()
    seg = st.selectbox("View ICP for", list(icps.keys()))
    with st.expander("View full ICP JSON", expanded=False):
        st.json(icps[seg])

    icp = icps[seg]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Segment:** {icp.get('segment_name', seg)}")
        st.markdown(f"**Value prop:** {icp.get('value_prop_one_liner', '—')}")
    with c2:
        thr = icp.get("scoring_thresholds", {})
        st.markdown(f"**Auto-drop below:** {thr.get('auto_drop_below', '—')}")
        st.markdown(f"**Tier 1 above:** {thr.get('tier_1_above', '—')}")
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load ICP configs: {exc}")

st.caption("Edit ICPs by changing config/icp_configs.json in the repo and pushing. Applies on the next run.")

st.divider()

# ---- Connection status ----
st.subheader("Connection status")
sid = sheet_id()
c1, c2 = st.columns(2)
with c1:
    if sid:
        ui.metric_card("Google Sheet", "Connected", f"ID: …{str(sid)[-6:]}")
    else:
        ui.metric_card("Google Sheet", "Not set", "Add SHEET_ID to secrets")
with c2:
    try:
        from dashboard.lib.data_access import get_runs_df
        n = len(get_runs_df(limit=1000))
        ui.metric_card("Runs visible", n, "from Run History tab")
    except Exception:
        ui.metric_card("Runs visible", "—", "")

st.divider()

# ---- API usage ----
st.subheader("API usage (last 30 days)")
usage = get_api_usage_df(days=30)
if usage.empty:
    st.caption("No API usage recorded yet.")
else:
    st.dataframe(usage, use_container_width=True, hide_index=True)
