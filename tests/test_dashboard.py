"""Phase 10 — Dashboard data-layer and trigger tests.

Streamlit rendering is not unit-tested (verified manually). These tests focus
on the data access layer, write-back behaviour, the GitHub trigger, and auth.
"""

from __future__ import annotations

import sqlite3
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_db(tmp_path):
    """Create a SQLite DB with sample leads + runs and point data_access at it."""
    from scripts.init_db import init_db, Lead, Run
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from datetime import datetime

    db_path = tmp_path / "leads.db"
    init_db(str(db_path))

    engine = create_engine(f"sqlite:///{db_path}", future=True)
    with Session(engine) as s:
        s.add(Lead(
            id="lead-1", segment="tutrain", company_name="Acme",
            decision_maker_name="Dana CEO", title="CEO", email="dana@acme.com",
            email_confidence=0.9, qualifier_score=88, status="ready_to_send",
            reply_likelihood=8, lead_hash="h1", reply_received=False,
            created_at=datetime.utcnow(), sheets_row_index=2,
        ))
        s.add(Lead(
            id="lead-2", segment="eqourse_ai_data", company_name="DataCo",
            decision_maker_name="Sam ML", title="Head of ML", email="sam@dataco.com",
            email_confidence=0.5, qualifier_score=72, status="needs_review",
            reply_likelihood=5, lead_hash="h2", reply_received=False,
            validation_reasons='["low_reply_likelihood"]',
            created_at=datetime.utcnow(), sheets_row_index=3,
        ))
        s.add(Run(
            id="run-1", segment="tutrain", status="success",
            candidates_found=20, qualified_count=2,
            started_at=datetime.utcnow(), completed_at=datetime.utcnow(),
            api_credits_used='{"gemini": 3}',
        ))
        s.commit()
    engine.dispose()

    # Point data_access at this DB and disable cache for deterministic tests.
    from dashboard.lib import data_access
    monkey_settings = MagicMock()
    monkey_settings.SQLITE_PATH = str(db_path)
    monkey_settings.SHEET_ID = "sheet-x"
    monkey_settings.GOOGLE_SHEETS_CREDS_PATH = "./secrets/gcp-service-account.json"
    monkey_settings.SHEET_TAB_TUTRAIN = "TUTRAIN_Leads"
    monkey_settings.SHEET_TAB_CONTENT = "eQOURSE_Content_Leads"
    monkey_settings.SHEET_TAB_AI_DATA = "eQOURSE_AI_Data_Leads"

    with patch.object(data_access, "_settings", return_value=monkey_settings):
        yield data_access, str(db_path)


# ---------------------------------------------------------------------------
# data_access reads
# ---------------------------------------------------------------------------

def test_data_access_get_leads_df_returns_dataframe(seeded_db):
    data_access, _ = seeded_db
    df = data_access.get_leads_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(["id", "company_name", "status"]).issubset(df.columns)


def test_data_access_filters_by_segment_and_status(seeded_db):
    data_access, _ = seeded_db

    by_segment = data_access.get_leads_df(segment="tutrain")
    assert len(by_segment) == 1
    assert by_segment.iloc[0]["company_name"] == "Acme"

    by_status = data_access.get_leads_df(status="needs_review")
    assert len(by_status) == 1
    assert by_status.iloc[0]["company_name"] == "DataCo"


def test_data_access_get_runs_df(seeded_db):
    data_access, _ = seeded_db
    runs = data_access.get_runs_df(limit=10)
    assert len(runs) == 1
    assert runs.iloc[0]["status"] == "success"


def test_count_leads_today(seeded_db):
    data_access, _ = seeded_db
    assert data_access.count_leads_today() == 2


def test_build_funnel_data(seeded_db):
    data_access, _ = seeded_db
    funnel = data_access.build_funnel_data()
    assert funnel["candidates"] == 20
    assert funnel["qualified"] == 2
    # lead-1 is ready_to_send → ready count >= 1
    assert funnel["ready_to_send"] >= 1


# ---------------------------------------------------------------------------
# data_access writes
# ---------------------------------------------------------------------------

def test_mark_lead_replied_updates_sqlite(seeded_db):
    data_access, db_path = seeded_db
    # Mock the Sheets write so no network call happens.
    with patch.object(data_access, "_update_sheet_cell_for_lead", return_value=True):
        data_access.mark_lead_replied("lead-1", "Thanks, interested!")

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT reply_received, status FROM leads WHERE id='lead-1'"
        ).fetchone()
        reply = conn.execute(
            "SELECT reply_text FROM replies WHERE lead_id='lead-1'"
        ).fetchone()
    finally:
        conn.close()

    assert row[0] == 1            # reply_received
    assert row[1] == "replied"    # status
    assert reply is not None and reply[0] == "Thanks, interested!"


def test_update_lead_status_updates_sqlite(seeded_db):
    data_access, db_path = seeded_db
    with patch.object(data_access, "_update_sheet_cell_for_lead", return_value=True):
        data_access.update_lead_status("lead-2", "approved")

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT status FROM leads WHERE id='lead-2'").fetchone()
    finally:
        conn.close()
    assert row[0] == "approved"


def test_update_lead_status_invalidates_cache(seeded_db):
    data_access, _ = seeded_db
    called = {"cleared": False}

    def fake_clear():
        called["cleared"] = True

    with patch.object(data_access, "_update_sheet_cell_for_lead", return_value=True), \
         patch.object(data_access, "_clear_cache", side_effect=fake_clear):
        data_access.update_lead_status("lead-1", "approved")

    assert called["cleared"] is True


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
        url = github_trigger.trigger_workflow(
            "daily-pipeline.yml", inputs={"segment": "all", "target": 30}
        )

    # create_dispatch must have been called with stringified inputs + ref
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
        status = github_trigger.get_workflow_status(
            "https://github.com/u/r/actions/runs/456"
        )

    mock_repo.get_workflow_run.assert_called_once_with(456)
    assert status["status"] == "completed"
    assert status["conclusion"] == "success"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_auth_wrong_password_returns_false():
    from dashboard.lib import auth

    fake_st = MagicMock()
    fake_st.session_state = {}
    fake_st.secrets = {"DASHBOARD_PASSWORD": "correct"}
    # Simulate the form: text_input returns wrong pw, submit True
    fake_form = MagicMock()
    fake_form.__enter__ = MagicMock(return_value=fake_form)
    fake_form.__exit__ = MagicMock(return_value=False)
    fake_st.form.return_value = fake_form
    fake_st.text_input.return_value = "wrong"
    fake_st.form_submit_button.return_value = True

    with patch.object(auth, "st", fake_st):
        result = auth.check_password()

    assert result is False
    fake_st.error.assert_called()


def test_auth_correct_password_sets_session_state():
    from dashboard.lib import auth

    class SessionState(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__

    fake_st = MagicMock()
    fake_st.session_state = SessionState()
    fake_st.secrets = {"DASHBOARD_PASSWORD": "correct"}
    fake_form = MagicMock()
    fake_form.__enter__ = MagicMock(return_value=fake_form)
    fake_form.__exit__ = MagicMock(return_value=False)
    fake_st.form.return_value = fake_form
    fake_st.text_input.return_value = "correct"
    fake_st.form_submit_button.return_value = True
    fake_st.rerun.side_effect = RuntimeError("rerun called")  # stops flow like real st.rerun

    with patch.object(auth, "st", fake_st):
        try:
            auth.check_password()
        except RuntimeError:
            pass  # st.rerun() raises in our stub to halt execution

    assert fake_st.session_state.get("authed") is True


def test_auth_already_authed_short_circuits():
    from dashboard.lib import auth

    fake_st = MagicMock()
    fake_st.session_state = {"authed": True}
    with patch.object(auth, "st", fake_st):
        assert auth.check_password() is True
    fake_st.form.assert_not_called()
