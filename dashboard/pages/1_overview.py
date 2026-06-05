"""Overview page — today's run, funnel, recent activity, cost."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dashboard.lib.bootstrap import bootstrap; bootstrap()  # noqa: E702

import pandas as pd
import streamlit as st

from dashboard.lib import charts
from dashboard.lib.data_access import (
    build_funnel_data,
    count_leads_today,
    get_api_usage_df,
    get_runs_df,
)

st.title("📊 Overview")

runs = get_runs_df(limit=1)
if runs.empty:
    st.info("No pipeline runs yet. Trigger one from the **Run Now** page.")
    st.stop()

last_run = runs.iloc[0]

# --- Top metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Last run", str(last_run.get("started_at", "—"))[:16])
col2.metric("Status", str(last_run.get("status", "—")))
col3.metric("Qualified", int(last_run["qualified_count"]) if pd.notna(last_run.get("qualified_count")) else 0)
col4.metric("Leads today", count_leads_today())

# --- Funnel ---
st.subheader("Latest Funnel")
funnel = build_funnel_data()
st.caption(charts.funnel_text(funnel))
st.bar_chart(charts.funnel_dataframe(funnel))

# --- Recent runs ---
st.subheader("Recent Runs")
runs_10 = get_runs_df(limit=10)
cols = [c for c in ["started_at", "segment", "status", "candidates_found", "qualified_count"] if c in runs_10.columns]
st.dataframe(runs_10[cols], use_container_width=True, hide_index=True)

# --- API cost last 7 days ---
st.subheader("API Cost — Last 7 Days")
cost_df = get_api_usage_df(days=7)
if cost_df.empty:
    st.caption("No API usage recorded in the last 7 days.")
else:
    st.bar_chart(cost_df.set_index("source")["credits"])
