"""Overview — today's run, funnel, recent activity, cost."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from dashboard.lib import charts, ui
from dashboard.lib.data_access import (
    build_funnel_data,
    count_leads_today,
    get_api_usage_df,
    get_leads_df,
    get_runs_df,
)

ui.inject_css()
ui.hero("Overview", "A snapshot of your most recent lead generation run.")

runs = get_runs_df(limit=50)

if runs.empty:
    ui.empty_state(
        "🚀",
        "No pipeline runs yet",
        "Head to the <b>Run Now</b> page to trigger your first run, or wait for "
        "the daily 6:30 AM run to populate this dashboard.",
    )
    st.stop()

leads = get_leads_df()
ready_total = 0
if not leads.empty:
    ready_total = int(leads["status"].isin(["ready_to_send", "approved", "sent"]).sum())

latest = runs.iloc[0]

# ---- Metric cards ----
c1, c2, c3, c4 = st.columns(4)
with c1:
    ui.metric_card("Last run", str(latest.get("started_at", "—"))[:16] or "—")
with c2:
    total_qualified = int(pd.to_numeric(runs.drop_duplicates("segment")["qualified_count"], errors="coerce").fillna(0).sum())
    ui.metric_card("Qualified (latest)", total_qualified, "across all segments")
with c3:
    ui.metric_card("Ready to send", ready_total, "leads awaiting outreach")
with c4:
    ui.metric_card("Added today", count_leads_today(), "new leads")

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ---- Funnel + segment breakdown ----
left, right = st.columns([1.3, 1])

with left:
    st.subheader("Latest funnel")
    funnel = build_funnel_data()
    st.caption(charts.funnel_text(funnel))
    st.bar_chart(charts.funnel_dataframe(funnel), color=ui.PRIMARY, height=260)

with right:
    st.subheader("By segment")
    latest_per_seg = runs.drop_duplicates("segment")
    for _, r in latest_per_seg.iterrows():
        seg = str(r.get("segment", "—")).replace("_", " ").title()
        c = int(pd.to_numeric(r.get("candidates_found"), errors="coerce") or 0)
        q = int(pd.to_numeric(r.get("qualified_count"), errors="coerce") or 0)
        rd = int(pd.to_numeric(r.get("ready_to_send"), errors="coerce") or 0)
        st.markdown(
            f"""<div class="lead-card">
            <div class="company">{seg}</div>
            <div class="person">{c} hunted &nbsp;→&nbsp; {q} qualified &nbsp;→&nbsp; {rd} ready</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ---- Recent runs ----
st.subheader("Recent runs")
cols = [c for c in ["started_at", "segment", "status", "candidates_found", "qualified_count", "ready_to_send"] if c in runs.columns]
st.dataframe(
    runs[cols].head(10),
    use_container_width=True,
    hide_index=True,
    column_config={
        "started_at": "Date",
        "segment": "Segment",
        "status": "Status",
        "candidates_found": "Hunted",
        "qualified_count": "Qualified",
        "ready_to_send": "Ready",
    },
)

# ---- API cost ----
st.subheader("API usage (last 7 days)")
cost_df = get_api_usage_df(days=7)
if cost_df.empty:
    st.caption("No API usage recorded yet.")
else:
    st.bar_chart(cost_df.set_index("source")["credits"], color=ui.ACCENT, height=240)
