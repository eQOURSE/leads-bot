"""Tests for TelegramSink — Phase 8. All offline; telegram.Bot mocked."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents._models import (
    DecisionMaker, EmailResult, EnrichedCandidate, EnrichedDecisionMaker,
    GeneratedMessages, MessagedCandidate, MessagedDecisionMaker,
    QualifiedCandidate, QualifiedCandidateWithPeople, QualifierSubScores,
    ValidatedCandidate, ValidatedDecisionMaker, ValidatedResult,
)
from sinks.telegram_sink import TelegramSink, _md2
from sources.models import CompanyCandidate


def _now():
    return datetime.now(timezone.utc)


def _make_company(domain="acme.com", name="Acme"):
    return CompanyCandidate(domain=domain, name=name, raw_source="test", confidence=0.8)


def _make_qualified(tier="tier_1"):
    return QualifiedCandidate(
        candidate=_make_company(),
        total_score=80, pre_score=55,
        sub_scores=QualifierSubScores(
            funding_recency_score=40, reachability_score=10, geography_score=10,
            size_match_score=10, segment_fit_score=10, buying_signal_score=10,
        ),
        reasoning="t", disqualifiers=[], tier=tier,  # type: ignore[arg-type]
        domain_was_resolved=False,
    )


def _make_dm():
    return DecisionMaker(full_name="Dana CEO", title="CEO", source="scrapegraph", seniority_score=95)


def _make_edm(email="dana@acme.com"):
    return EnrichedDecisionMaker(
        decision_maker=_make_dm(),
        email_result=EmailResult(email=email, confidence=0.9, source="smtp", smtp_verified=True),
    )


def _make_messages(likelihood=8):
    return GeneratedMessages(
        email_subject_a="Saw the raise", email_subject_b="Cut admin by half",
        email_body="Body. Best, X", linkedin_dm="Hi Dana",
        reply_likelihood=likelihood, quality_flags=[],
    )


def _make_mdm(likelihood=8):
    return MessagedDecisionMaker(
        enriched_dm=_make_edm(), messages=_make_messages(likelihood), skipped_reason=None,
    )


def _make_vdm(status="ready_to_send", likelihood=8):
    lh = hashlib.sha256("acme.com|dana ceo".encode()).hexdigest()
    return ValidatedDecisionMaker(
        messaged_dm=_make_mdm(likelihood=likelihood),
        status=status,  # type: ignore[arg-type]
        validation_reasons=[], lead_hash=lh,
    )


def _make_validated(stats=None, n_ready=1):
    qc = _make_qualified()
    cwp = QualifiedCandidateWithPeople(
        qualified=qc, decision_makers=[_make_dm()],
        lookup_status="found", lookup_attempts={},
    )
    ec = EnrichedCandidate(
        candidate_with_people=cwp, enriched_dms=[_make_edm()], enrichment_status="full",
    )
    mc = MessagedCandidate(enriched_candidate=ec, personalization=None, messaged_dms=[_make_mdm()])
    vdms = [_make_vdm("ready_to_send") for _ in range(n_ready)]
    vc = ValidatedCandidate(messaged_candidate=mc, validated_dms=vdms)
    now = _now()
    return ValidatedResult(
        segment="eqourse_ai_data", run_id="run-tg",
        validated_candidates=[vc],
        stats=stats or {"ready_to_send": n_ready, "needs_review": 0, "rejected": 0},
        api_credits_used={}, started_at=now, completed_at=now, duration_seconds=0.1,
    )


@pytest.fixture
def tg_settings(test_settings):
    test_settings.TELEGRAM_BOT_TOKEN = "test-bot-token"
    test_settings.TELEGRAM_CHAT_ID = "12345"
    return test_settings


@pytest.fixture
def sink(tg_settings):
    return TelegramSink(settings=tg_settings)


def _make_mock_bot(message_id=42):
    bot = MagicMock()
    msg = MagicMock()
    msg.message_id = message_id
    bot.send_message = AsyncMock(return_value=msg)
    return bot


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_markdown_v2_special_chars_escaped():
    """_md2 should escape MarkdownV2 special characters."""
    result = _md2("Hello! World_test.now")
    # These chars must be escaped in MarkdownV2
    assert "\\!" in result or "!" not in result  # ! must be escaped
    assert "\\_" in result or "_" not in result  # _ must be escaped


def test_digest_includes_top_5_ready_leads(sink):
    """Digest should list up to 5 ready-to-send leads."""
    # Create 7 ready leads; only top 5 should appear
    vr = _make_validated(n_ready=7)
    text, leads_included, _ = sink._build_digest({"eqourse_ai_data": vr}, "")
    assert leads_included <= 5


def test_digest_excludes_needs_review_from_top(sink):
    """needs_review leads should not appear in the top-5 section."""
    # Make one needs_review and one ready
    qc = _make_qualified()
    cwp = QualifiedCandidateWithPeople(
        qualified=qc, decision_makers=[_make_dm()],
        lookup_status="found", lookup_attempts={},
    )
    ec = EnrichedCandidate(
        candidate_with_people=cwp, enriched_dms=[_make_edm()], enrichment_status="full",
    )
    mc = MessagedCandidate(enriched_candidate=ec, personalization=None, messaged_dms=[_make_mdm()])
    vc = ValidatedCandidate(messaged_candidate=mc, validated_dms=[
        _make_vdm("needs_review"),
        _make_vdm("ready_to_send"),
    ])
    now = _now()
    vr = ValidatedResult(
        segment="eqourse_ai_data", run_id="run-tg",
        validated_candidates=[vc],
        stats={"ready_to_send": 1, "needs_review": 1, "rejected": 0},
        api_credits_used={}, started_at=now, completed_at=now, duration_seconds=0.1,
    )
    text, leads_included, _ = sink._build_digest({"eqourse_ai_data": vr}, "")
    assert leads_included == 1  # only 1 ready lead in top


@pytest.mark.asyncio
async def test_sends_to_correct_chat_id(sink):
    """Bot.send_message should be called with the configured chat_id."""
    vr = _make_validated()
    mock_bot = _make_mock_bot()
    with patch.object(sink, "_make_bot", return_value=mock_bot):
        result = await sink.send_run_digest({"eqourse_ai_data": vr})

    call_kwargs = mock_bot.send_message.call_args
    assert call_kwargs.kwargs.get("chat_id") == "12345" or str(call_kwargs).find("12345") >= 0


@pytest.mark.asyncio
async def test_marks_sent_to_telegram_at_after_success(sink, test_settings):
    """After a successful send, included leads should have sent_to_telegram_at set."""
    from sqlalchemy import create_engine, text as sa_text
    from sqlalchemy.orm import Session
    from scripts.init_db import Lead, init_db
    import uuid

    init_db(test_settings.SQLITE_PATH)
    lh = hashlib.sha256("acme.com|dana ceo".encode()).hexdigest()
    engine = create_engine(f"sqlite:///{test_settings.SQLITE_PATH}", future=True)
    with Session(engine) as s:
        s.add(Lead(id=str(uuid.uuid4()), lead_hash=lh, status="ready_to_send"))
        s.commit()
    engine.dispose()

    vr = _make_validated()
    mock_bot = _make_mock_bot()
    with patch.object(sink, "_make_bot", return_value=mock_bot):
        await sink.send_run_digest({"eqourse_ai_data": vr}, "https://sheets.test")

    # Allow async mark to complete
    import asyncio
    await asyncio.sleep(0.1)

    engine = create_engine(f"sqlite:///{test_settings.SQLITE_PATH}", future=True)
    with Session(engine) as s:
        row = s.execute(sa_text("SELECT sent_to_telegram_at FROM leads WHERE lead_hash = :h"), {"h": lh}).first()
    engine.dispose()
    assert row is not None and row[0] is not None


@pytest.mark.asyncio
async def test_error_alert_format(sink):
    """send_error_alert should call bot.send_message without crashing."""
    mock_bot = _make_mock_bot()
    with patch.object(sink, "_make_bot", return_value=mock_bot):
        await sink.send_error_alert("run-err", "Connection timeout on step 3")

    mock_bot.send_message.assert_called_once()
    call_text = mock_bot.send_message.call_args.kwargs.get("text", "")
    assert "run-err" in call_text or "Error" in call_text
