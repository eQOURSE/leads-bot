"""Needs Review — leads flagged by the validator for human judgment."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dashboard.lib.bootstrap import bootstrap; bootstrap()  # noqa: E702

import pandas as pd
import streamlit as st

from dashboard.lib.data_access import get_leads_df, update_lead_status

st.title("⚠️ Needs Review")
st.caption("Leads flagged by the validator for human judgment before sending.")

df = get_leads_df(status="needs_review")
if df.empty:
    st.success("All clear — no leads need review.")
    st.stop()


def _g(row, key, default="—"):
    val = row.get(key, default)
    if pd.isna(val) or val == "" or val is None:
        return default
    return val


for _, row in df.iterrows():
    lead_id = str(row.get("id", ""))
    with st.container(border=True):
        st.subheader(f"{_g(row, 'company_name')} — {_g(row, 'decision_maker_name')}")

        reasons = _g(row, "validation_reasons", "")
        if reasons and reasons not in ("—", "[]"):
            st.error(f"**Flagged reasons:** {reasons}")

        col1, col2, col3 = st.columns([3, 3, 1])
        col1.markdown(f"**Subject A:** {_g(row, 'email_subject_a')}")
        col1.markdown(f"**Subject B:** {_g(row, 'email_subject_b')}")
        col2.text_area(
            "Body", str(_g(row, "email_body", "")),
            height=150, key=f"nr_body_{lead_id}", disabled=True,
        )

        rl = row.get("reply_likelihood")
        col3.metric("Reply?", f"{int(rl)}/10" if pd.notna(rl) else "—/10")
        if col3.button("✅ Approve", key=f"nr_app_{lead_id}", type="primary"):
            update_lead_status(lead_id, "approved")
            st.rerun()
        if col3.button("❌ Reject", key=f"nr_rej_{lead_id}"):
            update_lead_status(lead_id, "rejected")
            st.rerun()
