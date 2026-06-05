"""Tests for MessageWriter — Phase 7. All offline; Gemini mocked."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents._models import (
    DecisionMaker,
    EmailResult,
    EnrichedCandidate,
    EnrichedDecisionMaker,
    EnrichedResult,
    GeneratedMessages,
    PersonalizationContext,
    QualifiedCandidate,
    QualifiedCandidateWithPeople,
    QualifierSubScores,
)
from agents.message_writer import MessageWriter, _post_process_validate_standalone
from sources.models import CompanyCandidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc)


def _make_company(domain: str = "acme.com", name: str = "Acme") -> CompanyCandidate:
    return CompanyCandidate(domain=domain, name=name, raw_source="test", confidence=0.8)


def _make_qualified(domain: str = "acme.com", tier: str = "tier_1") -> QualifiedCandidate:
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


def _make_dm(name: str = "Alice Smith", title: str = "CEO") -> DecisionMaker:
    from agents._constants import seniority_score as ss
    return DecisionMaker(
        full_name=name, title=title, source="scrapegraph", seniority_score=ss(title)
    )


def _make_edm(email: Optional[str] = "alice@acme.com", confidence: float = 0.85) -> EnrichedDecisionMaker:
    return EnrichedDecisionMaker(
        decision_maker=_make_dm(),
        email_result=EmailResult(
            email=email, confidence=confidence,
            source="pattern+smtp", smtp_verified=True,
        ),
    )


def _make_cwp(domain: str = "acme.com", tier: str = "tier_1") -> QualifiedCandidateWithPeople:
    return QualifiedCandidateWithPeople(
        qualified=_make_qualified(domain=domain, tier=tier),
        decision_makers=[_make_dm()],
        lookup_status="found", lookup_attempts={},
    )


def _make_ec(domain: str = "acme.com", tier: str = "tier_1", email: Optional[str] = "alice@acme.com") -> EnrichedCandidate:
    return EnrichedCandidate(
        candidate_with_people=_make_cwp(domain=domain, tier=tier),
        enriched_dms=[_make_edm(email=email)],
        enrichment_status="full",
    )


def _make_enriched(ecs: list[EnrichedCandidate], segment: str = "eqourse_ai_data", run_id: str = "run-id") -> EnrichedResult:
    now = _now()
    return EnrichedResult(
        segment=segment, run_id=run_id,
        enriched_candidates=ecs, stats={}, api_credits_used={},
        started_at=now, completed_at=now, duration_seconds=0.1,
    )


def _make_hook(domain: str = "acme.com") -> PersonalizationContext:
    return PersonalizationContext(
        domain=domain,
        company_one_liner="Acme builds AI tutoring tools.",
        recent_milestone="Raised $3M seed Jan 2026",
        pain_hypothesis_specific="Their instructors spend 4h/day on admin.",
        why_now_hook="Saw the Jan 2026 seed — admin automation is timely.",
        personalization_quality="high",
        built_at=_now(),
    )


def _make_good_messages() -> GeneratedMessages:
    body = (
        "Saw the Jan 2026 seed — congratulations on the raise. "
        "Growing fast usually means instructor admin starts to pile up.\n\n"
        "Most edtech teams at your stage spend 30% of their time on course "
        "coordination that could be automated.\n\n"
        "We help companies like Acme cut that in half within 60 days. "
        "Would a 20-minute call this week make sense?\n\n"
        "Best, Alex | eQOURSE x TUTRAIN"
    )
    return GeneratedMessages(
        email_subject_a="The $3M question for Acme",
        email_subject_b="Cut instructor admin by 50% in 60 days",
        email_body=body,
        linkedin_dm="Hi Alice — saw the Jan raise, congrats! Curious if instructor admin is on your radar. Happy to share how we help. Thoughts?",
        reply_likelihood=8,
        quality_flags=[],
    )


@pytest.fixture
def mock_icp():
    icp = MagicMock()
    icp.value_prop_one_liner = "AI course automation"
    icp.what_we_offer = "Automated course admin and scheduling"
    icp.outreach_angle.pain_hypothesis = "Instructors waste time on manual admin"
    icp.outreach_angle.value_framing = "Cut admin by 50% in 60 days"
    icp.outreach_angle.primary_cta = "20-minute call this week?"
    icp.outreach_angle.fallback_cta = "Send a one-pager?"
    icp.segment_name = "eqourse_ai_data"
    return icp


@pytest.fixture
def mock_icp_strategist(mock_icp):
    s = MagicMock()
    s.load_strategy.return_value = mock_icp
    return s


@pytest.fixture
def mock_gemini():
    g = MagicMock()
    g.generate_json = AsyncMock(return_value=_make_good_messages())
    return g


@pytest.fixture
def mock_lead_store():
    s = MagicMock()
    s.update_run = AsyncMock()
    return s


@pytest.fixture
def writer(test_settings, mock_icp_strategist, mock_gemini, mock_lead_store):
    return MessageWriter(
        settings=test_settings,
        icp_strategist=mock_icp_strategist,
        gemini_agent=mock_gemini,
        lead_store=mock_lead_store,
    )


# ---------------------------------------------------------------------------
# Core generation tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_writes_messages_for_dm_with_email(writer, mock_gemini):
    ec = _make_ec(domain="acme.com", email="alice@acme.com")
    p_map = {"acme.com": _make_hook()}
    result = await writer.write_for_enriched(_make_enriched([ec]), p_map)

    assert len(result.messaged_candidates) == 1
    mc = result.messaged_candidates[0]
    assert mc.messaged_dms[0].messages is not None
    assert mc.messaged_dms[0].skipped_reason is None
    assert mock_gemini.generate_json.call_count == 1


@pytest.mark.asyncio
async def test_skips_dm_with_no_email(writer, mock_gemini):
    ec = _make_ec(domain="noemail.com", email=None)
    result = await writer.write_for_enriched(_make_enriched([ec]), {})

    mc = result.messaged_candidates[0]
    assert mc.messaged_dms[0].messages is None
    assert mc.messaged_dms[0].skipped_reason == "no_email"
    mock_gemini.generate_json.assert_not_called()


@pytest.mark.asyncio
async def test_skips_dm_below_confidence_threshold(writer, mock_gemini):
    ec = _make_ec(domain="lowconf.com", email="test@lowconf.com")
    # Set low confidence
    ec.enriched_dms[0].email_result.confidence = 0.1

    result = await writer.write_for_enriched(
        _make_enriched([ec]), {}, min_email_confidence=0.3
    )

    mc = result.messaged_candidates[0]
    assert mc.messaged_dms[0].messages is None
    assert mc.messaged_dms[0].skipped_reason == "low_confidence"


# ---------------------------------------------------------------------------
# Post-processing validation tests (direct unit tests)
# ---------------------------------------------------------------------------

def test_subject_over_50_chars_flagged():
    msgs = _make_good_messages()
    msgs.email_subject_a = "A" * 51
    msgs.quality_flags = []
    result = _post_process_validate_standalone(msgs)
    assert "subject_a_too_long" in result.quality_flags


def test_banned_phrase_in_body_flagged():
    msgs = _make_good_messages()
    msgs.email_body = msgs.email_body + " I hope this email finds you well."
    msgs.quality_flags = []
    result = _post_process_validate_standalone(msgs)
    assert any("banned_phrase" in f for f in result.quality_flags)


def test_multiple_ctas_flagged():
    msgs = _make_good_messages()
    # Add multiple CTA phrases
    msgs.email_body = msgs.email_body + " Happy to jump on a call. Let me know if you are interested in a demo."
    msgs.quality_flags = []
    result = _post_process_validate_standalone(msgs)
    assert "multiple_ctas" in result.quality_flags


def test_word_count_validation_low_and_high():
    msgs = _make_good_messages()

    # Too short
    msgs.email_body = "Short. Best, Alex"
    msgs.quality_flags = []
    result = _post_process_validate_standalone(msgs)
    assert "body_word_count_low" in result.quality_flags

    # Too long
    msgs.email_body = ("word " * 150).strip() + " Best, Alex"
    msgs.quality_flags = []
    result = _post_process_validate_standalone(msgs)
    assert "body_word_count_high" in result.quality_flags


def test_linkedin_dm_over_280_flagged():
    msgs = _make_good_messages()
    msgs.linkedin_dm = "x" * 281
    msgs.quality_flags = []
    result = _post_process_validate_standalone(msgs)
    assert "linkedin_dm_too_long" in result.quality_flags


@pytest.mark.asyncio
async def test_fallback_messages_when_no_personalization(writer, mock_gemini):
    """When no personalization context exists, fallback is used."""
    fallback_msgs = _make_good_messages()
    fallback_msgs.quality_flags = ["fallback_no_personalization"]
    fallback_msgs.reply_likelihood = 4
    mock_gemini.generate_json = AsyncMock(return_value=fallback_msgs)

    ec = _make_ec(domain="nohook.com", email="bob@nohook.com")
    result = await writer.write_for_enriched(_make_enriched([ec]), personalization_map={})

    mc = result.messaged_candidates[0]
    msgs = mc.messaged_dms[0].messages
    assert msgs is not None
    assert "fallback_no_personalization" in msgs.quality_flags
    assert msgs.reply_likelihood <= 5


def test_reply_likelihood_penalty_per_flag():
    msgs = _make_good_messages()
    msgs.reply_likelihood = 8
    msgs.quality_flags = []
    msgs.email_subject_a = "A" * 51      # adds subject_a_too_long
    msgs.email_subject_b = "B" * 51      # adds subject_b_too_long
    result = _post_process_validate_standalone(msgs)
    # Two flags should reduce likelihood by 2
    assert result.reply_likelihood <= 6


@pytest.mark.asyncio
async def test_concurrent_writes_respect_semaphore(writer, mock_gemini):
    """max_concurrent_calls=2 should still complete all DMs."""
    ecs = [_make_ec(domain=f"co{i}.com", email=f"user@co{i}.com") for i in range(5)]
    p_map = {f"co{i}.com": _make_hook(domain=f"co{i}.com") for i in range(5)}
    enriched = _make_enriched(ecs)

    result = await writer.write_for_enriched(enriched, p_map, max_concurrent_calls=2)

    generated = sum(
        1 for mc in result.messaged_candidates
        for mdm in mc.messaged_dms
        if mdm.messages is not None
    )
    assert generated == 5


@pytest.mark.asyncio
async def test_messaged_result_writes_to_run_record(writer, mock_lead_store):
    ec = _make_ec(email="x@acme.com")
    await writer.write_for_enriched(_make_enriched([ec]), {"acme.com": _make_hook()})
    mock_lead_store.update_run.assert_called_once()


def test_post_process_validate_unit():
    """Direct call to validator with deliberately bad output."""
    msgs = GeneratedMessages(
        email_subject_a="S" * 55,          # too long
        email_subject_b="Fine subject",
        email_body="Too short. Best, X",   # under 80 words
        linkedin_dm="OK",
        reply_likelihood=9,
        quality_flags=[],
    )
    result = _post_process_validate_standalone(msgs)
    assert "subject_a_too_long" in result.quality_flags
    assert "body_word_count_low" in result.quality_flags
    assert result.reply_likelihood < 9  # penalised
