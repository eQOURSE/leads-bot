"""Phase 10 — Single-password auth gate with a polished login screen."""

from __future__ import annotations

import streamlit as st


def check_password() -> bool:
    """Single-password gate. Returns True if authenticated, False otherwise.

    Password is read from ``st.secrets["DASHBOARD_PASSWORD"]``.
    """
    if st.session_state.get("authed"):
        return True

    expected = _expected_password()

    # Centered login card
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;font-size:3rem;'>🎯</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='text-align:center;font-size:1.4rem;font-weight:800;"
            "color:#1A1A2E;'>eQOURSE Lead Generation</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='text-align:center;color:#6B7280;margin-bottom:18px;'>"
            "Sign in to view your leads</div>",
            unsafe_allow_html=True,
        )

        if not expected:
            st.error(
                "DASHBOARD_PASSWORD is not configured. Add it in the Streamlit "
                "Cloud secrets (or .streamlit/secrets.toml locally)."
            )
            return False

        with st.form("login", clear_on_submit=False):
            pwd = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")
            if submitted:
                if pwd == expected:
                    st.session_state.authed = True
                    st.rerun()
                else:
                    st.error("Incorrect password. Please try again.")
    return False


def _expected_password() -> str:
    try:
        return st.secrets.get("DASHBOARD_PASSWORD", "")  # type: ignore[no-any-return]
    except Exception:
        return ""
