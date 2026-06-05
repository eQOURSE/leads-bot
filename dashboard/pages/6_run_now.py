"""Run Now — trigger the pipeline manually via GitHub Actions workflow_dispatch."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dashboard.lib.bootstrap import bootstrap; bootstrap()  # noqa: E702

import streamlit as st

from dashboard.lib.github_trigger import get_workflow_status, trigger_workflow

st.title("▶️ Run Now")
st.caption("Trigger the pipeline manually. Runs via GitHub Actions in the background.")

col1, col2 = st.columns([2, 1])

with col1:
    segment_choice = st.radio(
        "What to run",
        ["All segments", "tutrain only", "eqourse_content only", "eqourse_ai_data only"],
    )
    target = st.slider("Target leads per segment", 5, 50, 30)

    if st.button("🚀 Trigger Pipeline", type="primary"):
        if segment_choice == "All segments":
            seg_value = "all"
        else:
            seg_value = segment_choice.split(" only")[0]
        params = {"segment": seg_value, "target": str(target)}
        try:
            run_url = trigger_workflow("daily-pipeline.yml", inputs=params)
            st.success(f"Pipeline triggered! Watch progress: [GitHub Actions run]({run_url})")
            st.session_state["last_triggered_run_url"] = run_url
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to trigger: {exc}")

with col2:
    st.markdown("**Last triggered:**")
    st.code(st.session_state.get("last_triggered_run_url", "—"))

    if st.button("🔄 Check status"):
        run_url = st.session_state.get("last_triggered_run_url")
        if run_url:
            status = get_workflow_status(run_url)
            st.write(status)
        else:
            st.info("No run triggered yet this session.")
