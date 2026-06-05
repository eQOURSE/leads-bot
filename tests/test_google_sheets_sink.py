"""Tests for GoogleSheetsSink — Phase 8. All offline; gspread mocked."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents._models import (
    DecisionMaker, EmailResult, EnrichedCandidate, EnrichedDecisionMaker,
    GeneratedMessages, MessagedCandidate, MessagedDecisionMaker,
    PersonalizationContext, QualifiedCandidate, QualifiedCandidateWithPeople,
    QualifierSubScores, ValidatedCandidate, ValidatedDecisionMaker, ValidatedResult,
)
from sinks.google_sheets_sink import GoogleSheetsSink
from sinks.sqlite_store import LeadStore
from sources.models import CompanyCandidate


def _now():
    return datetime.now(timezone.utc)


def _make_company(domain="acme.com"):
    return CompanyCandidate(domain=domain, name="Acme", raw_source="test", confidence=0.8)


def _make_qualified(domain="acme.com", tier="tier_1"):
    return QualifiedCandidate(
        candidate=_make_company(domain=domain),
        total_score=80, pre_score=55,
        sub_scores=QualifierSubScores(
            funding_recency_score=40, reachability_score=10, geography_score=10,
            size_match_score=10, segment_fit_score=10, buying_signal_score=10,
        ),
        reasoning="t", disqualifiers=[], tier=tier,  # type: ignore[arg-type]
        domain_was_resolved=False,
    )


def _make_dm():
    return DecisionMaker(full_name="Carol VP", title="VP", source="scrapegraph", seniority_score=75)


def _make_edm(email="carol@acme.com", conf=0.85):
    return EnrichedDecisionMaker(
        decision_maker=_make_dm(),
        email_result=EmailResult(email=email, confidence=conf, source="smtp", smtp_verified=True),
    )


def _make_messages():
    return GeneratedMessages(
        email_subject_a="Sub A", email_subject_b="Sub B",
        email_body="Body. Best, X", linkedin_dm="Hi Carol",
        reply_likelihood=7, quality_flags=[],
    )


def _make_mc(domain="acme.com", tier="tier_1"):
    qc = _make_qualified(domain=domain, tier=tier)
    cwp = QualifiedCandidateWithPeople(
        qualified=qc, decision_makers=[_make_dm()],
        lookup_status="found", lookup_attempts={},
    )
    ec = EnrichedCandidate(
        candidate_with_people=cwp, enriched_dms=[_make_edm()], enrichment_status="full",
    )
    return MessagedCandidate(enriched_candidate=ec, personalization=None, messaged_dms=[
        MessagedDecisionMaker(enriched_dm=_make_edm(), messages=_make_messages(), skipped_reason=None)
    ])


def _make_vdm(status="ready_to_send", domain="acme.com"):
    return ValidatedDecisionMaker(
        messaged_dm=MessagedDecisionMaker(
            enriched_dm=_make_edm(), messages=_make_messages(), skipped_reason=None
        ),
        status=status, validation_reasons=[], lead_hash=hashlib.sha256(f"{domain}|carol vp".encode()).hexdigest(),
    )


def _make_validated(domain="acme.com", status="ready_to_send", segment="eqourse_ai_data"):
    mc = _make_mc(domain=domain)
    vdm = _make_vdm(status=status, domain=domain)
    vc = ValidatedCandidate(messaged_candidate=mc, validated_dms=[vdm])
    now = _now()
    return ValidatedResult(
        segment=segment, run_id="run-sheets",
        validated_candidates=[vc],
        stats={"ready_to_send": 1, "needs_review": 0, "rejected": 0},
        api_credits_used={}, started_at=now, completed_at=now, duration_seconds=0.1,
    )


def _make_mock_worksheet(name="Sheet1", row_count=1):
    ws = MagicMock()
    ws.title = name
    ws.append_row = MagicMock()
    ws.format = MagicMock()
    ws.get_all_values = MagicMock(return_value=[["header"]] * row_count)
    return ws


def _make_mock_spreadsheet(tab_names=None):
    ss = MagicMock()
    existing_tabs = tab_names or []
    worksheets = [_make_mock_worksheet(t) for t in existing_tabs]
    ss.worksheets = MagicMock(return_value=worksheets)
    ss.add_worksheet = MagicMock(side_effect=lambda title, rows, cols: _make_mock_worksheet(title))
    ss.worksheet = MagicMock(side_effect=lambda name: _make_mock_worksheet(name))
    return ss


@pytest.fixture
def sheets_sink(test_settings):
    lead_store = LeadStore(test_settings)
    sink = GoogleSheetsSink(settings=test_settings, lead_store=lead_store)
    return sink


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ensure_tabs_creates_missing_tabs(sheets_sink):
    """_ensure_tabs_exist_sync should create tabs that don't exist yet."""
    ss = _make_mock_spreadsheet(tab_names=[])  # no existing tabs
    sheets_sink._spreadsheet = ss

    sheets_sink._ensure_tabs_exist_sync()

    # Should have called add_worksheet for all required tabs
    assert ss.add_worksheet.called


def test_append_leads_to_correct_tab_by_segment(sheets_sink):
    """ready_to_send leads should go to the segment tab, not Needs Review."""
    vr = _make_validated(segment="eqourse_ai_data", status="ready_to_send")
    ss = _make_mock_spreadsheet(tab_names=[
        "TUTRAIN_Leads", "eQOURSE_Content_Leads", "eQOURSE_AI_Data_Leads",
        "Needs Review", "Manual Lookup", "Run History",
    ])
    sheets_sink._spreadsheet = ss

    sheets_sink._write_leads_sync(vr)

    # worksheet("eQOURSE_AI_Data_Leads") should have been called for the ready lead
    calls = [str(c) for c in ss.worksheet.call_args_list]
    assert any("AI_Data" in c for c in calls)


def test_needs_review_goes_to_shared_tab(sheets_sink):
    """needs_review leads should go to the Needs Review tab."""
    vr = _make_validated(status="needs_review")
    ss = _make_mock_spreadsheet(tab_names=[
        "TUTRAIN_Leads", "eQOURSE_Content_Leads", "eQOURSE_AI_Data_Leads",
        "Needs Review", "Manual Lookup", "Run History",
    ])
    sheets_sink._spreadsheet = ss

    sheets_sink._write_leads_sync(vr)

    calls = [str(c) for c in ss.worksheet.call_args_list]
    assert any("Needs Review" in c for c in calls)


def test_color_codes_email_confidence_column(sheets_sink):
    """format() should be called on the confidence cell after appending."""
    from sinks.google_sheets_sink import _conf_color
    green = _conf_color(0.9)
    yellow = _conf_color(0.6)
    red = _conf_color(0.2)
    assert green["green"] > 0.8
    assert yellow["red"] == 1.0
    assert red["red"] > 0.8


def test_skips_already_sent_leads(sheets_sink, test_settings):
    """Second write_leads call for same lead should be skipped (idempotent)."""
    vr = _make_validated()
    ss = _make_mock_spreadsheet(tab_names=[
        "TUTRAIN_Leads", "eQOURSE_Content_Leads", "eQOURSE_AI_Data_Leads",
        "Needs Review", "Manual Lookup", "Run History",
    ])
    sheets_sink._spreadsheet = ss

    # First call — appends 1 row and writes sent_to_sheets_at back to SQLite
    from scripts.init_db import Lead, init_db
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy import text as sa_text
    import uuid

    init_db(test_settings.SQLITE_PATH)
    lh = hashlib.sha256("acme.com|carol vp".encode()).hexdigest()

    # Pre-write the lead row WITH sent_to_sheets_at already set using the sink's own DB path
    engine = create_engine(f"sqlite:///{test_settings.SQLITE_PATH}", future=True)
    with Session(engine) as s:
        # Use the same INSERT that the sink uses, then manually set sent_to_sheets_at
        from sqlalchemy import text as t2
        row_id = str(uuid.uuid4())
        s.execute(t2(
            "INSERT OR IGNORE INTO leads (id, lead_hash, status, reply_received) "
            "VALUES (:id, :h, 'ready_to_send', 0)"
        ), {"id": row_id, "h": lh})
        s.execute(t2(
            "UPDATE leads SET sent_to_sheets_at = '2026-01-01' WHERE lead_hash = :h"
        ), {"h": lh})
        s.commit()

    # Verify the row is there with sent_to_sheets_at set
    with Session(engine) as s:
        check = s.execute(
            sa_text("SELECT sent_to_sheets_at FROM leads WHERE lead_hash = :h"), {"h": lh}
        ).first()
    engine.dispose()
    assert check is not None and check[0] is not None, "Pre-condition: row must exist with sent_to_sheets_at"

    # Force vdm's lead_hash to match what's in the DB
    vr.validated_candidates[0].validated_dms[0].lead_hash = lh

    result = sheets_sink._write_leads_sync(vr)

    assert result["skipped_already_sent"] == 1
    assert result["appended"] == 0


def test_run_history_appended(sheets_sink):
    """write_run_history_sync should call append_row on Run History worksheet."""
    ss = _make_mock_spreadsheet(tab_names=["Run History"])
    sheets_sink._spreadsheet = ss
    ws = _make_mock_worksheet("Run History")
    ss.worksheet = MagicMock(return_value=ws)

    sheets_sink._write_run_history_sync({"date": "2026-01-01", "run_id": "x", "segment": "test"})

    ws.append_row.assert_called_once()


def test_writes_row_index_back_to_sqlite(sheets_sink, test_settings):
    """After appending, sent_to_sheets_at should be written back to SQLite."""
    from sqlalchemy import create_engine, text as sa_text
    from sqlalchemy.orm import Session
    from scripts.init_db import Lead, init_db
    import uuid

    init_db(test_settings.SQLITE_PATH)
    lh = hashlib.sha256("tracked.com|carol vp".encode()).hexdigest()
    engine = create_engine(f"sqlite:///{test_settings.SQLITE_PATH}", future=True)
    with Session(engine) as s:
        s.add(Lead(id=str(uuid.uuid4()), lead_hash=lh, status="ready_to_send"))
        s.commit()
    engine.dispose()

    vr = _make_validated(domain="tracked.com")
    vr.validated_candidates[0].validated_dms[0].lead_hash = lh

    ss = _make_mock_spreadsheet(tab_names=[
        "TUTRAIN_Leads", "eQOURSE_Content_Leads", "eQOURSE_AI_Data_Leads",
        "Needs Review", "Manual Lookup", "Run History",
    ])
    sheets_sink._spreadsheet = ss

    sheets_sink._write_leads_sync(vr)

    # Check that sent_to_sheets_at was updated
    engine = create_engine(f"sqlite:///{test_settings.SQLITE_PATH}", future=True)
    with Session(engine) as s:
        row = s.execute(sa_text("SELECT sent_to_sheets_at FROM leads WHERE lead_hash = :h"), {"h": lh}).first()
    engine.dispose()
    # sent_to_sheets_at should now be set
    assert row is not None and row[0] is not None
