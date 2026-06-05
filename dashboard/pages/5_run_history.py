"""Run History — past runs and API cost tracking."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dashboard.lib.bootstrap import bootstrap; bootstrap()  # noqa: E702

import pandas as pd
import streamlit as st

from dashboard.lib import charts
from dashboard.lib.data_access import get_runs_df

st.title("📜 Run History")

days = st.sidebar.slider("Show runs from last N days", 1, 90, 30)

df = get_runs_df(limit=1000)
if df.empty:
    st.info("No runs recorded yet.")
    st.stop()

df["_started"] = pd.to_datetime(df["started_at"], errors="coerce")
cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
df = df[df["_started"] >= cutoff]

st.subheader(f"Last {days} days")

if df.empty:
    st.caption("No runs in the selected window.")
else:
    col1, col2 = st.columns(2)
    col1.markdown("**Runs per day**")
    col1.line_chart(charts.runs_per_day(df))
    col2.markdown("**Qualified leads per day**")
    col2.line_chart(charts.qualified_per_day(df))

st.subheader("All runs")
display_cols = [
    c for c in [
        "started_at", "completed_at", "segment", "status", "candidates_found",
        "qualified_count", "api_credits_used",
    ] if c in df.columns
]
st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
