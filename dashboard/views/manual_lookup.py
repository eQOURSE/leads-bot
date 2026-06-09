"""Manual Lookup — companies whose domain couldn't be resolved."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from dashboard.lib import ui
from dashboard.lib.data_access import get_needs_manual_lookup_df

ui.inject_css()
ui.hero("Manual Lookup", "Companies whose website couldn't be auto-resolved. Worth a quick manual search.")

df = get_needs_manual_lookup_df()
if df.empty:
    ui.empty_state("🔎", "Nothing to look up", "No unresolved companies right now.")
    st.stop()

st.caption(f"{len(df)} compan{'ies' if len(df) != 1 else 'y'} to research")
st.dataframe(df, use_container_width=True, hide_index=True)

csv = df.to_csv(index=False)
st.download_button("📥 Download as CSV", csv, "manual_lookup.csv", "text/csv")
