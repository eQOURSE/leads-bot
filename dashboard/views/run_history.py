"""Run History — past runs and API cost tracking."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from dashboard.lib import charts, ui
from dashboard.lib.data_access import get_runs_df

ui.inject_css()
ui.hero("Run History", "Every pipeline run, with funnel trends and API cost over time.")

df = get_runs_df(limit=1000)
if df.empty:
    ui.empty_state("📜", "No runs recorded yet", "Runs will appear here after the first pipeline execution.")
    st.stop()

days = st.slider("Show runs from the last N days", 1, 90, 30)

df["_started"] = pd.to_datetime(df["started_at"], errors="coerce")
cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
window = df[df["_started"] >= cutoff]

if window.empty:
    st.info("No runs in the selected window. Showing all runs below.")
    window = df

c1, c2 = st.columns(2)
with c1:
    st.subheader("Runs per day")
    st.line_chart(charts.runs_per_day(window), color=ui.PRIMARY, height=240)
with c2:
    st.subheader("Qualified per day")
    st.line_chart(charts.qualified_per_day(window), color=ui.ACCENT, height=240)

st.subheader("All runs")
display_cols = [
    c for c in [
        "started_at", "segment", "status", "candidates_found",
        "qualified_count", "ready_to_send", "api_credits_used",
    ] if c in window.columns
]
st.dataframe(
    window[display_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "started_at": "Date",
        "segment": "Segment",
        "status": "Status",
        "candidates_found": "Hunted",
        "qualified_count": "Qualified",
        "ready_to_send": "Ready",
        "api_credits_used": "API Credits",
    },
)
