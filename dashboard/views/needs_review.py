"""Needs Review — leads flagged by the validator for human judgment."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from dashboard.lib import ui
from dashboard.lib.data_access import get_leads_df, update_lead_status

ui.inject_css()
ui.hero("Needs Review", "Leads the validator flagged for a quick human check before sending.")

df = get_leads_df(status="needs_review")
if df.empty:
    ui.empty_state("✅", "All clear", "No leads need review right now.")
    st.stop()

st.caption(f"{len(df)} lead{'s' if len(df) != 1 else ''} awaiting your decision")


def _g(row, key, default="—"):
    val = row.get(key, default)
    if pd.isna(val) or val == "" or val is None:
        return default
    return val


for _, row in df.iterrows():
    lead_id = str(row.get("id", ""))
    with st.container(border=True):
        top1, top2 = st.columns([3, 1])
        with top1:
            st.markdown(
                f"<div class='company' style='font-size:1.1rem;font-weight:700;'>{_g(row, 'company_name')}</div>"
                f"<div class='person'>{_g(row, 'decision_maker_name')} · {_g(row, 'title')}</div>",
                unsafe_allow_html=True,
            )
        with top2:
            rl = row.get("reply_likelihood")
            st.markdown(
                f"<div style='text-align:right'>{ui.tier_pill(_g(row, 'tier', ''))}</div>",
                unsafe_allow_html=True,
            )

        reasons = _g(row, "validation_reasons", "")
        if reasons and reasons not in ("—", "[]"):
            st.warning(f"**Why flagged:** {reasons}")

        st.markdown(f"**Subject A:** {_g(row, 'email_subject_a')}")
        st.text_area("Email body", str(_g(row, "email_body", "")), height=130, key=f"nr_body_{lead_id}")

        b1, b2, _ = st.columns([1, 1, 3])
        if b1.button("✅ Approve", key=f"nr_app_{lead_id}", type="primary", use_container_width=True):
            update_lead_status(lead_id, "approved")
            st.rerun()
        if b2.button("❌ Reject", key=f"nr_rej_{lead_id}", use_container_width=True):
            update_lead_status(lead_id, "rejected")
            st.rerun()
