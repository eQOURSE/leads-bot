"""Phase 10 — Shared UI components and styling for a modern, friendly dashboard.

Centralises the custom CSS and reusable render helpers (hero headers, metric
cards, lead cards, empty states) so every page has a consistent, polished look
that works for non-technical users.
"""

from __future__ import annotations

import streamlit as st

# Brand palette
PRIMARY = "#6C5CE7"
PRIMARY_DARK = "#5848c2"
ACCENT = "#00B894"
WARN = "#FDCB6E"
DANGER = "#E17055"
INK = "#1A1A2E"
MUTED = "#6B7280"
SURFACE = "#F4F3FB"

_GLOBAL_CSS = f"""
<style>
/* ---- Layout polish ---- */
.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}}

/* Hide Streamlit chrome we don't need */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #ffffff 0%, {SURFACE} 100%);
    border-right: 1px solid #ECEAF6;
}}
section[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.95rem;
}}

/* ---- Hero header ---- */
.hero {{
    background: linear-gradient(120deg, {PRIMARY} 0%, #8E7CF0 100%);
    border-radius: 18px;
    padding: 26px 30px;
    color: #fff;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(108,92,231,0.25);
}}
.hero h1 {{
    color: #fff;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0 0 4px 0;
    padding: 0;
}}
.hero p {{
    color: rgba(255,255,255,0.9);
    margin: 0;
    font-size: 0.98rem;
}}

/* ---- Metric cards ---- */
.metric-card {{
    background: #fff;
    border: 1px solid #ECEAF6;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(26,26,46,0.04);
    height: 100%;
}}
.metric-card .label {{
    color: {MUTED};
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}}
.metric-card .value {{
    color: {INK};
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
}}
.metric-card .sub {{
    color: {MUTED};
    font-size: 0.82rem;
    margin-top: 4px;
}}

/* ---- Pills / badges ---- */
.pill {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}}
.pill-green  {{ background: #E3F9F1; color: #00875A; }}
.pill-yellow {{ background: #FFF6E0; color: #B7791F; }}
.pill-red    {{ background: #FCEBE6; color: #C0392B; }}
.pill-purple {{ background: #EEEBFB; color: {PRIMARY_DARK}; }}
.pill-grey   {{ background: #EEF0F4; color: {MUTED}; }}

/* ---- Lead card ---- */
.lead-card {{
    background: #fff;
    border: 1px solid #ECEAF6;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(26,26,46,0.03);
}}
.lead-card .company {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {INK};
}}
.lead-card .person {{
    color: {MUTED};
    font-size: 0.9rem;
}}

/* ---- Empty state ---- */
.empty-state {{
    text-align: center;
    padding: 48px 20px;
    color: {MUTED};
}}
.empty-state .emoji {{ font-size: 3rem; }}
.empty-state .title {{ font-size: 1.15rem; font-weight: 700; color: {INK}; margin-top: 8px; }}
.empty-state .desc  {{ font-size: 0.92rem; margin-top: 4px; }}

/* ---- Buttons ---- */
.stButton > button {{
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid #E2DEF6;
}}
.stButton > button[kind="primary"] {{
    background: {PRIMARY};
    border: none;
}}
.stButton > button[kind="primary"]:hover {{
    background: {PRIMARY_DARK};
}}

/* ---- Login card ---- */
.login-wrap {{ max-width: 420px; margin: 6vh auto 0 auto; }}
</style>
"""


def inject_css() -> None:
    """Inject the global stylesheet. Call once per page (cheap, idempotent-ish)."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str = "") -> None:
    """Render the gradient hero header."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="hero"><h1>{title}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value, sub: str = "") -> None:
    """Render a single metric card (use inside a st.columns context)."""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def status_pill(status: str) -> str:
    """Return an HTML pill for a lead status."""
    mapping = {
        "ready_to_send": ("pill-green", "Ready to send"),
        "approved": ("pill-green", "Approved"),
        "sent": ("pill-purple", "Sent"),
        "replied": ("pill-purple", "Replied"),
        "needs_review": ("pill-yellow", "Needs review"),
        "rejected": ("pill-red", "Rejected"),
    }
    cls, label = mapping.get(str(status), ("pill-grey", str(status) or "—"))
    return f'<span class="pill {cls}">{label}</span>'


def tier_pill(tier: str) -> str:
    cls = "pill-purple" if str(tier) == "tier_1" else "pill-grey"
    label = "Tier 1" if str(tier) == "tier_1" else ("Tier 2" if str(tier) == "tier_2" else str(tier))
    return f'<span class="pill {cls}">{label}</span>'


def empty_state(emoji: str, title: str, desc: str = "") -> None:
    """Render a friendly empty-state block."""
    desc_html = f'<div class="desc">{desc}</div>' if desc else ""
    st.markdown(
        f'<div class="empty-state"><div class="emoji">{emoji}</div>'
        f'<div class="title">{title}</div>{desc_html}</div>',
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    """Render the brand block at the top of the sidebar."""
    st.markdown(
        f"""
        <div style="padding: 6px 4px 14px 4px;">
            <div style="font-size:1.25rem;font-weight:800;color:{PRIMARY_DARK};">
                🎯 eQOURSE
            </div>
            <div style="font-size:0.78rem;color:{MUTED};letter-spacing:0.04em;">
                LEAD GENERATION
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
