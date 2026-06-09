"""Lead Browser — filterable list of every lead with full detail + actions."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from dashboard.lib import ui
from dashboard.lib.data_access import (
    get_leads_df,
    mark_lead_replied,
    update_lead_status,
)

ui.inject_css()
ui.hero("Lead Browser", "Browse, review, and act on every lead the system found.")

# ---- Filters ----
fc1, fc2, fc3 = st.columns([1, 1, 2])
with fc1:
    segment = st.selectbox("Segment", ["All", "tutrain", "eqourse_content", "eqourse_ai_data"])
with fc2:
    status = st.selectbox(
        "Status",
        ["All", "ready_to_send", "needs_review", "approved", "sent", "replied", "rejected"],
    )
with fc3:
    search = st.text_input("Search", placeholder="Company, person, or email…")

df = get_leads_df(
    segment=None if segment == "All" else segment,
    status=None if status == "All" else status,
)

if search and not df.empty:
    def _col(name):
        return df[name].astype(str) if name in df.columns else pd.Series([""] * len(df), index=df.index)
    mask = (
        _col("company_name").str.contains(search, case=False, na=False)
        | _col("decision_maker_name").str.contains(search, case=False, na=False)
        | _col("email").str.contains(search, case=False, na=False)
    )
    df = df[mask]

if df.empty:
    ui.empty_state("🔍", "No leads match your filters", "Try widening the segment or status filter.")
    st.stop()

st.caption(f"Showing {len(df)} lead{'s' if len(df) != 1 else ''}")


def _g(row, key, default="—"):
    val = row.get(key, default)
    if pd.isna(val) or val == "" or val is None:
        return default
    return val


for _, row in df.iterrows():
    lead_id = str(row.get("id", ""))
    company = _g(row, "company_name")
    person = _g(row, "decision_maker_name")
    title = _g(row, "title")
    score = _g(row, "qualifier_score")

    header = f"{company}  —  {person} ({title})"
    with st.expander(header):
        # Top row: pills
        st.markdown(
            ui.tier_pill(_g(row, "tier", "")) + " &nbsp; " + ui.status_pill(_g(row, "status", "")),
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Email:** `{_g(row, 'email')}`")
            conf = row.get("email_confidence")
            st.markdown(f"**Email confidence:** {float(conf):.0%}" if pd.notna(conf) else "**Email confidence:** —")
            st.markdown(f"**LinkedIn:** {_g(row, 'linkedin_url')}")
            st.markdown(f"**Funding:** {_g(row, 'funding_amount')} · {_g(row, 'funding_stage')} · {_g(row, 'funding_date')}")
            hook = _g(row, "personalization_hook", "none")
            st.markdown(f"**Why-now hook:** _{hook}_")

            st.divider()
            t1, t2 = st.tabs(["📧 Email", "💼 LinkedIn DM"])
            with t1:
                st.markdown(f"**Subject A:** {_g(row, 'email_subject_a')}")
                st.markdown(f"**Subject B:** {_g(row, 'email_subject_b')}")
                st.text_area("Body (copy & send from your inbox)", str(_g(row, "email_body", "")),
                             height=200, key=f"body_{lead_id}")
            with t2:
                st.text_area("LinkedIn DM", str(_g(row, "linkedin_dm", "")),
                             height=120, key=f"dm_{lead_id}")

        with col2:
            rl = row.get("reply_likelihood")
            ui.metric_card("Reply likelihood", f"{int(rl)}/10" if pd.notna(rl) else "—")
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            reasons = _g(row, "validation_reasons", "")
            if reasons and reasons not in ("—", "[]"):
                st.warning(f"Flags: {reasons}")

            cur_status = str(row.get("status", ""))
            reply_received = bool(row.get("reply_received", False))

            if cur_status == "needs_review":
                if st.button("✅ Approve", key=f"approve_{lead_id}", use_container_width=True, type="primary"):
                    update_lead_status(lead_id, "approved")
                    st.rerun()
                if st.button("❌ Reject", key=f"reject_{lead_id}", use_container_width=True):
                    update_lead_status(lead_id, "rejected")
                    st.rerun()

            if cur_status in ["ready_to_send", "approved", "sent"] and not reply_received:
                note = st.text_input("Reply note (optional)", key=f"note_{lead_id}")
                if st.button("✉️ Mark replied", key=f"replied_{lead_id}", use_container_width=True, type="primary"):
                    mark_lead_replied(lead_id, note)
                    st.rerun()

st.divider()
csv = df.to_csv(index=False)
st.download_button("📥 Download as CSV", csv, "leads.csv", "text/csv")
