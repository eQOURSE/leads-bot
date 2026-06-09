"""Run Now — trigger the pipeline manually via GitHub Actions workflow_dispatch."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from dashboard.lib import ui
from dashboard.lib.github_trigger import get_workflow_status, trigger_workflow

ui.inject_css()
ui.hero("Run Now", "Trigger the lead generation pipeline on demand. It runs in the background on GitHub Actions.")

col1, col2 = st.columns([2, 1])

with col1:
    with st.container(border=True):
        segment_choice = st.radio(
            "What to run",
            ["All segments", "tutrain only", "eqourse_content only", "eqourse_ai_data only"],
        )
        target = st.slider("Target leads per segment", 5, 50, 30)

        if st.button("🚀 Trigger pipeline", type="primary", use_container_width=True):
            seg_value = "all" if segment_choice == "All segments" else segment_choice.split(" only")[0]
            params = {"segment": seg_value, "target": str(target)}
            try:
                with st.spinner("Dispatching to GitHub Actions…"):
                    run_url = trigger_workflow("daily-pipeline.yml", inputs=params)
                st.success("Pipeline triggered successfully!")
                st.markdown(f"[▶️ Watch the run on GitHub Actions]({run_url})")
                st.session_state["last_triggered_run_url"] = run_url
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to trigger: {exc}")
                st.caption(
                    "Check that GITHUB_TOKEN (with Actions write) and GITHUB_REPO "
                    "are set in the dashboard secrets."
                )

with col2:
    with st.container(border=True):
        st.markdown("**Last triggered run**")
        url = st.session_state.get("last_triggered_run_url")
        if url:
            st.markdown(f"[Open run]({url})")
            if st.button("🔄 Check status", use_container_width=True):
                status = get_workflow_status(url)
                st.json(status)
        else:
            st.caption("No run triggered yet this session.")
