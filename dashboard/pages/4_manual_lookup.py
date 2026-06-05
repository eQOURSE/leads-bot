"""Manual Lookup — companies whose domain couldn't be resolved."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dashboard.lib.bootstrap import bootstrap; bootstrap()  # noqa: E702

import streamlit as st

from dashboard.lib.data_access import get_needs_manual_lookup_df

st.title("🔎 Manual Lookup")
st.caption(
    "Companies whose domain couldn't be resolved (news-source URLs, .unknown). "
    "Worth manual research."
)

df = get_needs_manual_lookup_df()
if df.empty:
    st.info("Nothing to lookup right now (or Google Sheets is unavailable).")
    st.stop()

st.dataframe(df, use_container_width=True, hide_index=True)

csv = df.to_csv(index=False)
st.download_button("📥 Download as CSV", csv, "manual_lookup.csv", "text/csv")
