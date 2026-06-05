"""Lead Browser — filterable list of every lead with full detail + actions."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dashboard.lib.bootstrap import bootstrap; bootstrap()  # noqa: E702

import pandas as pd
import streamlit as st

from dashboard.lib.data_access import (
    get_leads_df,
    mark_lead_replied,
    update_lead_status,
)

st.title("🔍 Lead Browser")

# --- Filters ---
segment = st.sidebar.selectbox(
    "Segment", ["All", "tutrain", "eqourse_content", "eqourse_ai_data"]
)
status = st.sidebar.selectbox(
    "Status",
    ["All", "ready_to_send", "needs_review", "approved", "sent", "replied", "rejected"],
)
min_score = st.sidebar.slider("Min qualifier score", 0, 100, 0)
search = st.sidebar.text_input("Search (company, name, email)")

df = get_leads_df(
    segment=None if segment == "All" else segment,
    status=None if status == "All" else status,
)

if not df.empty and "qualifier_score" in df.columns:
    df = df[pd.to_numeric(df["qualifier_score"], errors="coerce").fillna(0) >= min_score]

if search and not df.empty:
    def _col(name):
        return df[name].astype(str) if name in df.columns else pd.Series([""] * len(df), index=df.index)
    mask = (
        _col("company_name").str.contains(search, case=False, na=False)
        | _col("decision_maker_name").str.contains(search, case=False, na=False)
        | _col("email").str.contains(search, case=False, na=False)
    )
    df = df[mask]

st.caption(f"Showing {len(df)} leads")

if df.empty:
    st.info("No leads match the current filters.")
    st.stop()


def _g(row, key, default="—"):
    val = row.get(key, default)
    if pd.isna(val) or val == "" or val is None:
        return default
    return val


for _, row in df.iterrows():
    lead_id = str(row.get("id", ""))
    title_line = (
        f"**{_g(row, 'company_name')}** — {_g(row, 'decision_maker_name')} "
        f"({_g(row, 'title')}) — Score: {_g(row, 'qualifier_score')}"
    )
    with st.expander(title_line):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Email:** `{_g(row, 'email')}`")
            conf = row.get("email_confidence")
            st.markdown(f"**Confidence:** {float(conf):.2f}" if pd.notna(conf) else "**Confidence:** —")
            st.markdown(f"**LinkedIn:** {_g(row, 'linkedin_url')}")
            st.markdown(f"**Funding:** {_g(row, 'funding_amount')} on {_g(row, 'funding_date')}")
            st.markdown(f"**Hook:** _{_g(row, 'personalization_hook', 'none')}_")
            st.divider()
            tab1, tab2 = st.tabs(["📧 Email", "💼 LinkedIn DM"])
            with tab1:
                st.markdown(f"**Subject A:** {_g(row, 'email_subject_a')}")
                st.markdown(f"**Subject B:** {_g(row, 'email_subject_b')}")
                st.text_area(
                    "Body", str(_g(row, "email_body", "")),
                    height=200, key=f"body_{lead_id}", disabled=True,
                )
            with tab2:
                st.text_area(
                    "LinkedIn DM", str(_g(row, "linkedin_dm", "")),
                    height=100, key=f"dm_{lead_id}", disabled=True,
                )
        with col2:
            rl = row.get("reply_likelihood")
            st.metric("Reply Likelihood", f"{int(rl)}/10" if pd.notna(rl) else "—/10")
            st.markdown(f"**Status:** `{_g(row, 'status')}`")
            reasons = _g(row, "validation_reasons", "")
            if reasons and reasons != "—" and reasons != "[]":
                st.warning(f"Flags: {reasons}")

            cur_status = str(row.get("status", ""))
            reply_received = bool(row.get("reply_received", 0))

            if cur_status == "needs_review":
                if st.button("✅ Approve", key=f"approve_{lead_id}"):
                    update_lead_status(lead_id, "approved")
                    st.rerun()
                if st.button("❌ Reject", key=f"reject_{lead_id}"):
                    update_lead_status(lead_id, "rejected")
                    st.rerun()

            if cur_status in ["ready_to_send", "approved", "sent"] and not reply_received:
                reply_text = st.text_input("Reply note (optional)", key=f"reply_text_{lead_id}")
                if st.button("✉️ Mark Replied", key=f"replied_{lead_id}"):
                    mark_lead_replied(lead_id, reply_text)
                    st.rerun()

# --- Export ---
st.divider()
csv = df.to_csv(index=False)
st.download_button("📥 Download as CSV", csv, "leads.csv", "text/csv")
