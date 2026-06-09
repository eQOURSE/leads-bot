"""Phase 10 — Dashboard data-layer and trigger tests.

The dashboard reads from Google Sheets (the shared source of truth between the
GitHub Actions cron and Streamlit Cloud). These tests mock the Sheets record
reads and verify normalization, filtering, funnel building, write-back, the
GitHub trigger, and auth.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Sample Google Sheets records (as gspread get_all_records() would return)
# ---------------------------------------------------------------------------

_LEAD_RECORDS = [
    {
        "Date Added": "2026-06-09 10:00", "Run ID": "run-1", "Segment": "tutrain",
        "Tier": "tier_1", "Company": "Acme", "Domain": "acme.com",
        "Decision Maker": "Dana CEO", "Title": "CEO", "Email": "dana@acme.com",
        "Email Confidence": "0.90", "Email Source": "hunter", "Phone": "",
        "LinkedIn": "https://linkedin.com/in/dana", "Funding Amount": "5000000",
        "Funding Stage": "Series A", "Funding Date": "2026-05-01",
        "Qualifier Score": "88", "Why-Now Hook": "Just raised",
        "Subject A": "Quick idea", "Subject B": "Scaling", "Email Body": "Hi Dana...",
        "LinkedIn DM": "Hi Dana", "Reply Likelihood": "8", "Quality Flags": "[]",
        "Validation Reasons": "[]", "Status": "ready_to_send", "Sent?": "No",
        "Replied?": "No", "Notes": "",
    },
]

_NEEDS_REVIEW_RECORDS = [
    {
        "Date Added": "2026-06-09 10:01", "Run ID": "run-1", "Segment": "eqourse_ai_data",
        "Tier": "tier_2", "Company": "DataCo", "Domain": "dataco.com",
        "Decision Maker": "Sam ML", "Title": "Head of ML", "Email": "sam@dataco.com",
        "Email Confidence": "0.50", "Email Source": "pattern", "Phone": "",
        "LinkedIn": "", "Funding Amount": "", "Funding Stage": "", "Funding Date": "",
        "Qualifier Score": "72", "Why-Now Hook": "", "Subject A": "Annotation",
        "Subject B": "", "Email Body": "Hi Sam...", "LinkedIn DM": "",
        "Reply Likelihood": "5", "Quality Flags": '["low"]',
        "Validation Reasons": '["low_reply_likelihood"]', "Status": "needs_review",
        "Sent?": "No", "Replied?": "No", "Notes": "",
    },
]

_RUN_RECORDS = [
    {
        "Date": "2026-06-09 10:00", "Run ID": "run-1", "Segment": "tutrain",
        "Status": "completed", "Duration (s)": "15.0", "Candidates Hunted": "20",
        "Qualified": "2", "DMs Found": "3", "Emails Found": "2",
        "Messages Generated": "2", "Ready to Send": "1", "Needs Review": "0",
        "Rejected": "0", "API Credits Used": '{"gemini": 3}', "Errors": "",
    },
]


def _patch_tab_reads(monkeypatch):
    """Patch _read_tab_records to return our sample records per tab."""
    from dashboard.lib import data_access

    def fake_read(tab_name):
        if tab_name == "Run History":
            return _RUN_RECORDS
        if tab_name == "Needs Review":
            return _NEEDS_REVIEW_RECORDS
        if tab_name in (
            data_access._SEGMENT_TABS["tutrain"],
            data_access._SEGMENT_TABS["eqourse_content"],
            data_access._SEGMENT_TABS["eqourse_ai_data"],
        ):
            # Only tutrain tab has the ready lead in this fixture
            if tab_name == data_access._SEGMENT_TABS["tutrain"]:
                return _LEAD_RECORDS
            return []
        return []

    monkeypatch.setattr(data_access, "_read_tab_records", fake_read)
    # Also stub sheet_id so config resolution never fails
    monkeypatch.setattr(data_access, "sheet_id", lambda: "test-sheet")
    return data_access


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def test_get_leads_df_returns_dataframe(monkeypatch):
    da = _patch_tab_reads(monkeypatch)
    df = da.get_leads_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # 1 from tutrain tab + 1 from needs review
    assert "company_name" in df.columns
    assert set(df["company_name"]) == {"Acme", "DataCo"}


def test_get_leads_df_filters_by_segment(monkeypatch):
    da = _patch_tab_reads(monkeypatch)
    df = da.get_leads_df(segment="tutrain")
    assert len(df) == 1
    assert df.iloc[0]["company_name"] == "Acme"


def test_get_leads_df_filters_by_status(monkeypatch):
    da = _patch_tab_reads(monkeypatch)
    df = da.get_leads_df(status="needs_review")
    assert len(df) == 1
    assert df.iloc[0]["company_name"] == "DataCo"


def test_get_runs_df(monkeypatch):
    da = _patch_tab_reads(monkeypatch)
    runs = da.get_runs_df(limit=10)
    assert len(runs) == 1
    assert runs.iloc[0]["status"] == "completed"
    assert runs.iloc[0]["candidates_found"] == 20


def test_count_leads_today(monkeypatch):
    da = _patch_tab_reads(monkeypatch)
    # The sample created_at uses 2026-06-09; only counts if today matches.
    count = da.count_leads_today()
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    expected = 2 if today == "2026-06-09" else 0
    assert count == expected


def test_build_funnel_data(monkeypatch):
    da = _patch_tab_reads(monkeypatch)
    funnel = da.build_funnel_data()
    assert funnel["candidates"] == 20
    assert funnel["qualified"] == 2
    assert funnel["ready_to_send"] == 1


def test_get_api_usage_df(monkeypatch):
    da = _patch_tab_reads(monkeypatch)
    usage = da.get_api_usage_df(days=30)
    assert not usage.empty
    assert "gemini" in set(usage["source"])
    assert int(usage[usage["source"] == "gemini"]["credits"].iloc[0]) == 3


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def test_mark_lead_replied_calls_sheet_update(monkeypatch):
    da = _patch_tab_reads(monkeypatch)
    calls = []

    def fake_update(lead_id, column_header, value):
        calls.append((lead_id, column_header, value))
        return True

    monkeypatch.setattr(da, "_find_and_update", fake_update)
    monkeypatch.setattr(da, "clear_cache", lambda: None)

    da.mark_lead_replied("run-1|dana@acme.com", "Interested!")

    # Must update Replied? and Notes
    headers = [c[1] for c in calls]
    assert "Replied?" in headers
    assert "Notes" in headers


def test_update_lead_status_calls_sheet_update(monkeypatch):
    da = _patch_tab_reads(monkeypatch)
    calls = []
    monkeypatch.setattr(da, "_find_and_update", lambda lid, col, val: calls.append((lid, col, val)) or True)
    monkeypatch.setattr(da, "clear_cache", lambda: None)

    da.update_lead_status("run-1|sam@dataco.com", "approved")

    assert calls == [("run-1|sam@dataco.com", "Status", "approved")]


def test_update_lead_status_invalidates_cache(monkeypatch):
    da = _patch_tab_reads(monkeypatch)
    cleared = {"v": False}
    monkeypatch.setattr(da, "_find_and_update", lambda *a: True)
    monkeypatch.setattr(da, "clear_cache", lambda: cleared.__setitem__("v", True))

    da.update_lead_status("run-1|x@y.com", "rejected")
    assert cleared["v"] is True


# ---------------------------------------------------------------------------
# GitHub trigger
# ---------------------------------------------------------------------------

def test_github_trigger_calls_workflow_dispatch_with_inputs():
    from dashboard.lib import github_trigger

    mock_workflow = MagicMock()
    mock_workflow.create_dispatch.return_value = True
    mock_run = MagicMock()
    mock_run.html_url = "https://github.com/u/r/actions/runs/123"
    mock_workflow.get_runs.return_value = [mock_run]
    mock_repo = MagicMock()
    mock_repo.get_workflow.return_value = mock_workflow
    mock_gh = MagicMock()
    mock_gh.get_repo.return_value = mock_repo

    secrets = {"GITHUB_TOKEN": "tok", "GITHUB_REPO": "u/r"}
    with patch.object(github_trigger, "_secret", side_effect=lambda k, d=None: secrets.get(k, d)), \
         patch.object(github_trigger, "_github_client", return_value=mock_gh), \
         patch("time.sleep", return_value=None):
        url = github_trigger.trigger_workflow("daily-pipeline.yml", inputs={"segment": "all", "target": 30})

    mock_workflow.create_dispatch.assert_called_once()
    kwargs = mock_workflow.create_dispatch.call_args.kwargs
    assert kwargs["ref"] == "main"
    assert kwargs["inputs"] == {"segment": "all", "target": "30"}
    assert url == "https://github.com/u/r/actions/runs/123"


def test_github_get_workflow_status_parses_run_id():
    from dashboard.lib import github_trigger

    mock_run = MagicMock()
    mock_run.status = "completed"
    mock_run.conclusion = "success"
    mock_run.html_url = "https://github.com/u/r/actions/runs/456"
    mock_run.run_started_at = None
    mock_repo = MagicMock()
    mock_repo.get_workflow_run.return_value = mock_run
    mock_gh = MagicMock()
    mock_gh.get_repo.return_value = mock_repo

    secrets = {"GITHUB_TOKEN": "tok", "GITHUB_REPO": "u/r"}
    with patch.object(github_trigger, "_secret", side_effect=lambda k, d=None: secrets.get(k, d)), \
         patch.object(github_trigger, "_github_client", return_value=mock_gh):
        status = github_trigger.get_workflow_status("https://github.com/u/r/actions/runs/456")

    mock_repo.get_workflow_run.assert_called_once_with(456)
    assert status["status"] == "completed"
    assert status["conclusion"] == "success"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_auth_already_authed_short_circuits():
    from dashboard.lib import auth

    fake_st = MagicMock()
    fake_st.session_state = {"authed": True}
    with patch.object(auth, "st", fake_st):
        assert auth.check_password() is True
    fake_st.form.assert_not_called()


def test_auth_no_password_configured_returns_false():
    from dashboard.lib import auth

    class SessionState(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

    fake_st = MagicMock()
    fake_st.session_state = SessionState()
    fake_st.secrets = {}  # no DASHBOARD_PASSWORD
    fake_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
    # Make columns context-managers
    for c in fake_st.columns.return_value:
        c.__enter__ = MagicMock(return_value=c)
        c.__exit__ = MagicMock(return_value=False)

    with patch.object(auth, "st", fake_st):
        result = auth.check_password()
    assert result is False
