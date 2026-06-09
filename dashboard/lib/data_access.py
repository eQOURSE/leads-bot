"""Phase 10 — Data access layer for the Streamlit dashboard.

ARCHITECTURE NOTE
-----------------
The dashboard and the pipeline cron run on DIFFERENT machines (Streamlit Cloud
vs GitHub Actions). SQLite is local to each, so it cannot be the shared store.
**Google Sheets is the single source of truth the dashboard reads from** — both
the cron and local runs write leads + run history there.

Reads come from these tabs (created by sinks/google_sheets_sink.py):
  - <segment>_Leads tabs + "Needs Review"  → leads
  - "Run History"                          → runs
  - "Manual Lookup"                        → unresolved companies

All reads are cached (st.cache_data, 5-min TTL) and degrade to an empty
DataFrame with the expected columns if Sheets is unavailable, so no page
ever crashes on missing data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Tab names (must match sinks/google_sheets_sink.py)
_SEGMENT_TABS = {
    "tutrain": "TUTRAIN_Leads",
    "eqourse_content": "eQOURSE_Content_Leads",
    "eqourse_ai_data": "eQOURSE_AI_Data_Leads",
}
_NEEDS_REVIEW_TAB = "Needs Review"
_MANUAL_LOOKUP_TAB = "Manual Lookup"
_RUN_HISTORY_TAB = "Run History"

# Normalized lead columns the pages rely on.
_LEAD_COLUMNS = [
    "id", "created_at", "run_id", "segment", "tier", "company_name", "domain",
    "decision_maker_name", "title", "email", "email_confidence", "email_source",
    "phone", "linkedin_url", "funding_amount", "funding_stage", "funding_date",
    "qualifier_score", "personalization_hook", "email_subject_a",
    "email_subject_b", "email_body", "linkedin_dm", "reply_likelihood",
    "quality_flags", "validation_reasons", "status", "sent", "replied",
    "notes", "reply_received",
]

# Map Sheet lead headers → normalized column names.
_LEAD_HEADER_MAP = {
    "Date Added": "created_at",
    "Run ID": "run_id",
    "Segment": "segment",
    "Tier": "tier",
    "Company": "company_name",
    "Domain": "domain",
    "Decision Maker": "decision_maker_name",
    "Title": "title",
    "Email": "email",
    "Email Confidence": "email_confidence",
    "Email Source": "email_source",
    "Phone": "phone",
    "LinkedIn": "linkedin_url",
    "Funding Amount": "funding_amount",
    "Funding Stage": "funding_stage",
    "Funding Date": "funding_date",
    "Qualifier Score": "qualifier_score",
    "Why-Now Hook": "personalization_hook",
    "Subject A": "email_subject_a",
    "Subject B": "email_subject_b",
    "Email Body": "email_body",
    "LinkedIn DM": "linkedin_dm",
    "Reply Likelihood": "reply_likelihood",
    "Quality Flags": "quality_flags",
    "Validation Reasons": "validation_reasons",
    "Status": "status",
    "Sent?": "sent",
    "Replied?": "replied",
    "Notes": "notes",
}

_RUN_COLUMNS = [
    "started_at", "completed_at", "run_id", "segment", "status",
    "candidates_found", "qualified_count", "dms_found", "emails_found",
    "messages_generated", "ready_to_send", "needs_review", "rejected",
    "api_credits_used", "errors",
]

_RUN_HEADER_MAP = {
    "Date": "started_at",
    "Run ID": "run_id",
    "Segment": "segment",
    "Status": "status",
    "Duration (s)": "duration_seconds",
    "Candidates Hunted": "candidates_found",
    "Qualified": "qualified_count",
    "DMs Found": "dms_found",
    "Emails Found": "emails_found",
    "Messages Generated": "messages_generated",
    "Ready to Send": "ready_to_send",
    "Needs Review": "needs_review",
    "Rejected": "rejected",
    "API Credits Used": "api_credits_used",
    "Errors": "errors",
}


# ---------------------------------------------------------------------------
# Config / secrets resolution (works on Cloud and locally)
# ---------------------------------------------------------------------------

def _settings():
    from config.settings import get_settings
    return get_settings()


def _secret(key: str, default=None):
    """Read a value from st.secrets first (Cloud), else fall back to Settings/env."""
    if st is not None:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    try:
        return getattr(_settings(), key, default)
    except Exception:
        return default


def sheet_id() -> Optional[str]:
    return _secret("SHEET_ID")


# ---------------------------------------------------------------------------
# Cache decorator shim (works with or without a live Streamlit runtime)
# ---------------------------------------------------------------------------

def _cache_data(ttl: int = 300):
    if st is not None and hasattr(st, "cache_data"):
        return st.cache_data(ttl=ttl)

    def _identity(fn):
        return fn

    return _identity


def clear_cache() -> None:
    if st is not None and hasattr(st, "cache_data"):
        try:
            st.cache_data.clear()
        except Exception:
            pass


# Backwards-compatible private alias (older code/tests referenced _clear_cache)
_clear_cache = clear_cache


# ---------------------------------------------------------------------------
# Google Sheets client — supports Cloud (secrets dict) and local (file)
# ---------------------------------------------------------------------------

def _gsheet_client():
    import gspread

    # Cloud: service account provided as a TOML table in st.secrets.
    if st is not None:
        try:
            if "gcp_service_account" in st.secrets:
                sa = dict(st.secrets["gcp_service_account"])
                return gspread.service_account_from_dict(sa)
        except Exception:
            pass

    # Local: service account JSON file.
    creds_path = _secret("GOOGLE_SHEETS_CREDS_PATH", "./secrets/gcp-service-account.json")
    if creds_path and not Path(creds_path).is_absolute():
        creds_path = str((_PROJECT_ROOT / creds_path).resolve())
    return gspread.service_account(filename=creds_path)


def _open_spreadsheet():
    sid = sheet_id()
    if not sid:
        raise RuntimeError("SHEET_ID is not configured (add it to Streamlit secrets).")
    return _gsheet_client().open_by_key(sid)


def _read_tab_records(tab_name: str) -> list[dict]:
    """Return all records from a worksheet, or [] if missing/unavailable."""
    try:
        ss = _open_spreadsheet()
        ws = ss.worksheet(tab_name)
        return ws.get_all_records()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Reads — Leads
# ---------------------------------------------------------------------------

def _normalize_leads(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=_LEAD_COLUMNS)

    df = pd.DataFrame(records)
    df = df.rename(columns=_LEAD_HEADER_MAP)

    # Ensure all expected columns exist.
    for col in _LEAD_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Synthesize a stable id (run_id|email) for action buttons.
    df["id"] = (
        df["run_id"].astype(str).fillna("") + "|" +
        df["email"].astype(str).fillna("")
    )

    # Coerce numerics.
    for num_col in ("email_confidence", "qualifier_score", "reply_likelihood"):
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # reply_received from "Replied?" == Yes
    df["reply_received"] = df["replied"].astype(str).str.lower().isin(["yes", "true", "1"])

    return df[_LEAD_COLUMNS]


@_cache_data(ttl=300)
def get_leads_df(segment: Optional[str] = None, status: Optional[str] = None) -> pd.DataFrame:
    """Read all leads from the segment tabs + Needs Review, with optional filters."""
    records: list[dict] = []

    # Decide which tabs to read.
    if segment and segment in _SEGMENT_TABS:
        tabs = [_SEGMENT_TABS[segment], _NEEDS_REVIEW_TAB]
    else:
        tabs = list(_SEGMENT_TABS.values()) + [_NEEDS_REVIEW_TAB]

    seen_tabs = set()
    for tab in tabs:
        if tab in seen_tabs:
            continue
        seen_tabs.add(tab)
        records.extend(_read_tab_records(tab))

    df = _normalize_leads(records)

    if df.empty:
        return df

    if segment:
        df = df[df["segment"] == segment]
    if status:
        df = df[df["status"] == status]

    # Sort newest first.
    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False, na_position="last")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reads — Runs
# ---------------------------------------------------------------------------

def _normalize_runs(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=_RUN_COLUMNS)

    df = pd.DataFrame(records)
    df = df.rename(columns=_RUN_HEADER_MAP)

    for col in _RUN_COLUMNS:
        if col not in df.columns:
            df[col] = None

    for num_col in (
        "candidates_found", "qualified_count", "dms_found", "emails_found",
        "messages_generated", "ready_to_send", "needs_review", "rejected",
    ):
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0).astype(int)

    return df


@_cache_data(ttl=300)
def get_runs_df(limit: int = 100) -> pd.DataFrame:
    """Read run history from the 'Run History' tab, most recent first."""
    records = _read_tab_records(_RUN_HISTORY_TAB)
    df = _normalize_runs(records)
    if df.empty:
        return df
    if "started_at" in df.columns:
        df = df.sort_values("started_at", ascending=False, na_position="last")
    return df.head(int(limit)).reset_index(drop=True)


@_cache_data(ttl=300)
def get_needs_manual_lookup_df() -> pd.DataFrame:
    """Read the 'Manual Lookup' tab. Empty DF on any failure."""
    records = _read_tab_records(_MANUAL_LOOKUP_TAB)
    return pd.DataFrame(records) if records else pd.DataFrame()


@_cache_data(ttl=300)
def get_api_usage_df(days: int = 7) -> pd.DataFrame:
    """Aggregate API credits per source from Run History 'API Credits Used' JSON."""
    runs = get_runs_df(limit=1000)
    if runs.empty or "api_credits_used" not in runs.columns:
        return pd.DataFrame(columns=["source", "credits"])

    totals: dict[str, int] = {}
    for raw in runs["api_credits_used"].dropna():
        try:
            credits = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except Exception:
            continue
        for k, v in (credits or {}).items():
            try:
                totals[k] = totals.get(k, 0) + int(v)
            except Exception:
                continue

    if not totals:
        return pd.DataFrame(columns=["source", "credits"])

    return (
        pd.DataFrame([{"source": k, "credits": v} for k, v in totals.items()])
        .sort_values("credits", ascending=False)
        .reset_index(drop=True)
    )


def count_leads_today() -> int:
    """Count leads whose created_at falls on today's date."""
    df = get_leads_df()
    if df.empty or "created_at" not in df.columns:
        return 0
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    created = df["created_at"].astype(str)
    return int(created.str.startswith(today).sum())


# ---------------------------------------------------------------------------
# Funnel
# ---------------------------------------------------------------------------

def build_funnel_data(segment: Optional[str] = None) -> dict[str, int]:
    """Build candidates → qualified → ready funnel from the latest run(s)."""
    runs = get_runs_df(limit=50)
    if runs.empty:
        return {"candidates": 0, "qualified": 0, "ready_to_send": 0}

    if segment and segment != "All":
        runs = runs[runs["segment"] == segment]
    if runs.empty:
        return {"candidates": 0, "qualified": 0, "ready_to_send": 0}

    latest = runs.drop_duplicates("segment")
    candidates = int(pd.to_numeric(latest["candidates_found"], errors="coerce").fillna(0).sum())
    qualified = int(pd.to_numeric(latest["qualified_count"], errors="coerce").fillna(0).sum())
    ready = int(pd.to_numeric(latest["ready_to_send"], errors="coerce").fillna(0).sum())

    return {"candidates": candidates, "qualified": qualified, "ready_to_send": ready}


# ---------------------------------------------------------------------------
# Writes — update the Google Sheet directly (find row by run_id + email)
# ---------------------------------------------------------------------------

def _find_and_update(lead_id: str, column_header: str, value: str) -> bool:
    """Find a lead row (by 'Run ID|Email' id) across lead tabs and update one column."""
    try:
        run_id, _, email = lead_id.partition("|")
        ss = _open_spreadsheet()
        for tab in list(_SEGMENT_TABS.values()) + [_NEEDS_REVIEW_TAB]:
            try:
                ws = ss.worksheet(tab)
            except Exception:
                continue
            headers = ws.row_values(1)
            if "Email" not in headers or column_header not in headers:
                continue
            records = ws.get_all_records()
            email_col = headers.index("Email") + 1
            runid_col = headers.index("Run ID") + 1 if "Run ID" in headers else None
            target_col = headers.index(column_header) + 1
            for i, rec in enumerate(records):
                row_idx = i + 2  # +1 header, +1 1-based
                rec_email = str(rec.get("Email", ""))
                rec_runid = str(rec.get("Run ID", ""))
                if rec_email == email and (not run_id or rec_runid == run_id):
                    ws.update_cell(row_idx, target_col, value)
                    return True
        return False
    except Exception:
        return False


def mark_lead_replied(lead_id: str, reply_text: str = "") -> None:
    """Mark a lead as replied in Google Sheets."""
    _find_and_update(lead_id, "Replied?", "Yes")
    if reply_text:
        _find_and_update(lead_id, "Notes", reply_text)
    clear_cache()


def update_lead_status(lead_id: str, new_status: str) -> None:
    """Update a lead's status in Google Sheets (e.g. approve/reject)."""
    _find_and_update(lead_id, "Status", new_status)
    clear_cache()


# ---------------------------------------------------------------------------
# ICP / config helpers for the Settings page
# ---------------------------------------------------------------------------

def load_icp_configs() -> dict:
    path = _PROJECT_ROOT / "config" / "icp_configs.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
