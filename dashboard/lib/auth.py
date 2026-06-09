"""Phase 10 — Single-password auth gate for the dashboard."""

from __future__ import annotations

import streamlit as st


def check_password() -> bool:
    """Single-password gate. Returns True if authenticated, False otherwise.

    The password is read from ``st.secrets["DASHBOARD_PASSWORD"]``. On a wrong
    password the form is re-shown with an error.
    """
    if st.session_state.get("authed"):
        return True

    expected = _expected_password()
    if not expected:
        st.error(
            "DASHBOARD_PASSWORD is not configured. Add it to "
            ".streamlit/secrets.toml (local) or the Streamlit Cloud secrets UI."
        )
        return False

    with st.form("login"):
        pwd = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")
        if submitted:
            if pwd == expected:
                st.session_state.authed = True
                st.rerun()
            else:
                st.error("Wrong password")
    return False


def _expected_password() -> str:
    """Read the configured password from st.secrets, tolerant of absence."""
    try:
        return st.secrets.get("DASHBOARD_PASSWORD", "")  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001 - secrets file may be missing in local dev
        return ""
