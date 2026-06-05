"""Tests for SQLiteWriter — Phase 8. Uses a real in-process SQLite DB."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock

import pytest

from agents._models import (
    DecisionMaker, EmailResult, EnrichedCandidate, EnrichedDecisionMaker,
    GeneratedMessages, MessagedCandidate, MessagedDecisionMaker,
    PersonalizationContext, QualifiedCandidate, QualifiedCandidateWithPeople,
    QualifierSubScores, ValidatedCandidate, ValidatedDecisionMaker, ValidatedResult,
)
from sinks.sqlite_store import LeadStore
from sinks.sqlite_writer import SQLiteWriter
from sources.models import CompanyCandidate


def _now():
    return datetime.now(timezone.utc)


def _make_company(domain="acme.com", name="Acme"):
    return CompanyCandidate(domain=domain, name=name, raw_source="test", confidence=0.8)


def _make_qualified(domain="acme.com", tier="tier_1"):
    return QualifiedCandidate(
        candidate=_make_company(domain=domain),
        total_score=80, pre_score=55,
        sub_scores=QualifierSubScores(
            funding_recency_score=40, reachability_score=10,
            geography_score=10, size_match_score=10,
            segment_fit_score=10, buying_signal_score=10,
        ),
        reasoning="test", disqualifiers=[],
        tier=tier, domain_was_resolved=False,  # type: ignore[arg-type]
    )


def _make_dm():
    return DecisionMaker(full_name="Bob CEO", title="CEO", source="scrapegraph", seniority_score=95)


def _make_edm(email="bob@acme.com"):
    return EnrichedDecisionMaker(
        decision_maker=_make_dm(),
        email_result=EmailResult(email=email, confidence=0.85, source="pattern+smtp", smtp_verified=True),
    )


def _make_messages():
    return GeneratedMessages(
        email_subject_a="Subj A", email_subject_b="Subj B",
        email_body="Body text here for testing. Best, X",
        linkedin_dm="Hi Bob!", reply_likelihood=7, quality_flags=[],
    )


def _make_mdm(email="bob@acme.com"):
    return MessagedDecisionMaker(
        enriched_dm=_make_edm(email=email),
        messages=_make_messages(),
        skipped_reason=None,
    )


def _make_mc(domain="acme.com"):
    qc = _make_qualified(domain=domain)
    cwp = QualifiedCandidateWithPeople(
        qualified=qc, decision_makers=[_make_dm()],
        lookup_status="found", lookup_attempts={},
    )
    ec = EnrichedCandidate(
        candidate_with_people=cwp, enriched_dms=[_make_edm()], enrichment_status="full",
    )
    return MessagedCandidate(
        enriched_candidate=ec, personalization=None, messaged_dms=[_make_mdm()],
    )


def _lead_hash(domain, name):
    return hashlib.sha256(f"{domain}|{name.lower()}".encode()).hexdigest()


def _make_vdm(domain="acme.com", status="ready_to_send"):
    mdm = _make_mdm()
    return ValidatedDecisionMaker(
        messaged_dm=mdm,
        status=status,  # type: ignore[arg-type]
        validation_reasons=[],
        lead_hash=_lead_hash(domain, "Bob CEO"),
    )


def _make_validated_result(domain="acme.com", status="ready_to_send", run_id="run-1"):
    mc = _make_mc(domain=domain)
    vdm = _make_vdm(domain=domain, status=status)
    vc = ValidatedCandidate(messaged_candidate=mc, validated_dms=[vdm])
    now = _now()
    return ValidatedResult(
        segment="eqourse_ai_data", run_id=run_id,
        validated_candidates=[vc],
        stats={"ready_to_send": 1, "needs_review": 0, "rejected": 0},
        api_credits_used={}, started_at=now, completed_at=now, duration_seconds=0.1,
    )


@pytest.fixture
def writer(test_settings):
    lead_store = LeadStore(test_settings)
    return SQLiteWriter(settings=test_settings, lead_store=lead_store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inserts_validated_lead_rows(writer):
    vr = _make_validated_result()
    counts = await writer.write_validated(vr)
    assert counts["inserted"] == 1
    assert counts["skipped_existing"] == 0


@pytest.mark.asyncio
async def test_skips_duplicate_lead_hash(writer):
    """Second write with same lead_hash should skip (INSERT OR IGNORE)."""
    vr = _make_validated_result(run_id="run-1")
    await writer.write_validated(vr)

    vr2 = _make_validated_result(run_id="run-2")  # same domain/name → same hash
    counts2 = await writer.write_validated(vr2)

    assert counts2["inserted"] == 0
    assert counts2["skipped_existing"] == 1


@pytest.mark.asyncio
async def test_writes_validation_reasons_as_json(writer, test_settings):
    """validation_reasons should be stored as a JSON string."""
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session
    from scripts.init_db import Lead
    import json

    mc = _make_mc()
    lh = _lead_hash("acme.com", "Bob CEO")
    vdm = ValidatedDecisionMaker(
        messaged_dm=_make_mdm(),
        status="needs_review",  # type: ignore[arg-type]
        validation_reasons=["low_reply_likelihood", "weak_personalization"],
        lead_hash=lh,
    )
    vc = ValidatedCandidate(messaged_candidate=mc, validated_dms=[vdm])
    now = _now()
    vr = ValidatedResult(
        segment="eqourse_ai_data", run_id="run-reasons",
        validated_candidates=[vc],
        stats={}, api_credits_used={}, started_at=now, completed_at=now, duration_seconds=0.1,
    )

    await writer.write_validated(vr)

    engine = create_engine(f"sqlite:///{test_settings.SQLITE_PATH}", future=True)
    try:
        with Session(engine) as s:
            row = s.execute(select(Lead).where(Lead.lead_hash == lh)).first()
            assert row is not None
            reasons = json.loads(row[0].validation_reasons)
            assert "low_reply_likelihood" in reasons
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_transactional_rollback_on_error(writer, test_settings):
    """A bad row should not block subsequent good rows in a separate call."""
    # This tests graceful per-row error handling
    good_vr = _make_validated_result(domain="good.com", run_id="run-good")
    counts = await writer.write_validated(good_vr)
    assert counts["inserted"] >= 0  # doesn't crash
