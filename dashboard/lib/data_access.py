"""Phase 10 — Data access layer for the Streamlit dashboard.

SQLite is the primary read source (fast, local). Google Sheets is the sync
layer for human edits (status flips, replies) so they survive a fresh
``init_db``. Writes go to BOTH SQLite and Sheets, then the cache is cleared.

All read functions are wrapped in ``st.cache_data`` with a 5-minute TTL so
rapid page navigation never re-hits SQLite/Sheets.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover - streamlit always present in app context
    st = None  # type: ignore

# Project root so we can resolve the SQLite path / configs regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _settings():
    """Lazily load Settings (avoids import cost at module import time)."""
    from config.settings import get_settings
    return get_settings()


def _sqlite_path() -> str:
    """Resolve the SQLite path to an absolute location under the project root."""
    raw = _settings().SQLITE_PATH
    p = Path(raw)
    if not p.is_absolute():
        p = (_PROJECT_ROOT / raw).resolve()
    return str(p)


def _connect() -> sqlite3.Connection:
    # check_same_thread=False because Streamlit may call from worker threads.
    return sqlite3.connect(_sqlite_path(), check_same_thread=False)


# ---------------------------------------------------------------------------
# Cache decorator shim — works with or without a live Streamlit runtime
# (so the data layer is unit-testable outside the app).
# ---------------------------------------------------------------------------

def _cache_data(ttl: int = 300):
    if st is not None and hasattr(st, "cache_data"):
        return st.cache_data(ttl=ttl)

    def _identity(fn):
        return fn

    return _identity


def _clear_cache() -> None:
    if st is not None and hasattr(st, "cache_data"):
        try:
            st.cache_data.clear()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Reads (SQLite)
# ---------------------------------------------------------------------------

@_cache_data(ttl=300)
def get_leads_df(segment: Optional[str] = None, status: Optional[str] = None) -> pd.DataFrame:
    """Read leads from SQLite with optional segment/status filters."""
    conn = _connect()
    try:
        query = "SELECT * FROM leads WHERE 1=1"
        params: list = []
        if segment:
            query += " AND segment = ?"
            params.append(segment)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


@_cache_data(ttl=300)
def get_runs_df(limit: int = 100) -> pd.DataFrame:
    """Read run history from SQLite, most recent first."""
    conn = _connect()
    try:
        return pd.read_sql_query(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
            conn,
            params=[int(limit)],
        )
    finally:
        conn.close()


@_cache_data(ttl=300)
def get_replies_df(limit: int = 500) -> pd.DataFrame:
    """Read the replies table."""
    conn = _connect()
    try:
        return pd.read_sql_query(
            "SELECT * FROM replies ORDER BY received_at DESC LIMIT ?",
            conn,
            params=[int(limit)],
        )
    finally:
        conn.close()


@_cache_data(ttl=300)
def get_api_usage_df(days: int = 7) -> pd.DataFrame:
    """Aggregate api_usage credits per source over the last N days."""
    conn = _connect()
    try:
        df = pd.read_sql_query("SELECT source, date, credits_used FROM api_usage", conn)
    finally:
        conn.close()
    if df.empty:
        return pd.DataFrame(columns=["source", "credits"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    df = df[df["date"] >= cutoff]
    if df.empty:
        return pd.DataFrame(columns=["source", "credits"])
    agg = (
        df.groupby("source")["credits_used"]
        .sum()
        .reset_index()
        .rename(columns={"credits_used": "credits"})
        .sort_values("credits", ascending=False)
    )
    return agg


def count_leads_today() -> int:
    """Count leads created today (UTC date match on created_at)."""
    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM leads WHERE date(created_at) = date('now')"
        )
        return int(cur.fetchone()[0])
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Reads (Google Sheets) — Manual Lookup tab is not mirrored into SQLite
# ---------------------------------------------------------------------------

def _gsheet_client():
    import gspread
    creds_path = _settings().GOOGLE_SHEETS_CREDS_PATH
    if not Path(creds_path).is_absolute():
        creds_path = str((_PROJECT_ROOT / creds_path).resolve())
    return gspread.service_account(filename=creds_path)


@_cache_data(ttl=300)
def get_needs_manual_lookup_df() -> pd.DataFrame:
    """Read the 'Manual Lookup' tab from Google Sheets. Empty DF on any failure."""
    try:
        client = _gsheet_client()
        sheet = client.open_by_key(_settings().SHEET_ID).worksheet("Manual Lookup")
        rows = sheet.get_all_records()
        return pd.DataFrame(rows)
    except Exception:  # noqa: BLE001 - dashboard must not crash if Sheets is down
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Funnel data for the Overview page
# ---------------------------------------------------------------------------

def build_funnel_data(segment: Optional[str] = None) -> dict[str, int]:
    """Build a candidates → qualified → ready funnel from the latest run(s).

    Reads from the most recent runs table rows. Returns a stage→count dict.
    """
    runs = get_runs_df(limit=10)
    if runs.empty:
        return {"candidates": 0, "qualified": 0, "ready_to_send": 0}

    if segment and segment != "All":
        runs = runs[runs["segment"] == segment]
    if runs.empty:
        return {"candidates": 0, "qualified": 0, "ready_to_send": 0}

    # Use the most recent run per segment (group), summed across segments.
    latest = runs.sort_values("started_at", ascending=False).drop_duplicates("segment")
    candidates = int(latest["candidates_found"].fillna(0).sum())
    qualified = int(latest["qualified_count"].fillna(0).sum())

    leads = get_leads_df(segment=None if not segment or segment == "All" else segment)
    ready = 0
    if not leads.empty and "status" in leads.columns:
        ready = int((leads["status"].isin(["ready_to_send", "approved", "sent"])).sum())

    return {"candidates": candidates, "qualified": qualified, "ready_to_send": ready}


# ---------------------------------------------------------------------------
# Writes — SQLite + Google Sheets, then cache invalidation
# ---------------------------------------------------------------------------

def mark_lead_replied(lead_id: str, reply_text: str = "") -> None:
    """Mark a lead as replied in SQLite + Sheets and log the reply text."""
    import uuid

    conn = _connect()
    try:
        conn.execute(
            "UPDATE leads SET reply_received=1, status='replied' WHERE id=?",
            (lead_id,),
        )
        conn.execute(
            "INSERT INTO replies (id, lead_id, received_at, reply_text) "
            "VALUES (?, ?, datetime('now'), ?)",
            (str(uuid.uuid4()), lead_id, reply_text),
        )
        conn.commit()
    finally:
        conn.close()

    _update_sheet_cell_for_lead(lead_id, column_header="Replied?", value="Yes")
    _clear_cache()


def update_lead_status(lead_id: str, new_status: str) -> None:
    """User override of validator status (approve / reject a needs_review lead)."""
    conn = _connect()
    try:
        conn.execute(
            "UPDATE leads SET status=? WHERE id=?",
            (new_status, lead_id),
        )
        conn.commit()
    finally:
        conn.close()

    _update_sheet_cell_for_lead(lead_id, column_header="Status", value=new_status)
    _clear_cache()


def _update_sheet_cell_for_lead(lead_id: str, column_header: str, value: str) -> bool:
    """Best-effort: find the lead's row in its segment tab via SQLite-stored
    sheets_row_index and update one column. Returns True on success.

    Never raises — Sheets being unavailable must not break a local SQLite edit.
    """
    try:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT segment, sheets_row_index FROM leads WHERE id=?",
                (lead_id,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return False
        segment, sheets_row_index = row
        if not sheets_row_index:
            return False

        tab = _segment_tab_name(segment)
        client = _gsheet_client()
        ss = client.open_by_key(_settings().SHEET_ID)
        ws = ss.worksheet(tab)

        # Find the column index from the header row.
        headers = ws.row_values(1)
        if column_header not in headers:
            return False
        col_idx = headers.index(column_header) + 1  # 1-based
        ws.update_cell(int(sheets_row_index), col_idx, value)
        return True
    except Exception:  # noqa: BLE001
        return False


def _segment_tab_name(segment: str) -> str:
    s = _settings()
    return {
        "tutrain": s.SHEET_TAB_TUTRAIN,
        "eqourse_content": s.SHEET_TAB_CONTENT,
        "eqourse_ai_data": s.SHEET_TAB_AI_DATA,
    }.get(segment, s.SHEET_TAB_AI_DATA)


# ---------------------------------------------------------------------------
# ICP / config helpers for the Settings page
# ---------------------------------------------------------------------------

def load_icp_configs() -> dict:
    path = _PROJECT_ROOT / "config" / "icp_configs.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
