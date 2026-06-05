"""Tests for Validator — Phase 8. All offline; dependencies mocked."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents._models import (
    DecisionMaker, EmailResult, EnrichedCandidate, EnrichedDecisionMaker,
    EnrichedResult, GeneratedMessages, MessagedCandidate, MessagedDecisionMaker,
    MessagedResult, PersonalizationContext, QualifiedCandidate,
    QualifiedCandidateWithPeople, QualifierSubScores,
)
from agents.validator import Validator
from sources.models import CompanyCandidate
from sources.smtp_verifier import SMTPResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_dm(title="CEO"):
    from agents._constants import seniority_score as ss
    return DecisionMaker(full_name="Alice Smith", title=title, source="scrapegraph", seniority_score=ss(title))


def _make_edm(email="alice@acme.com", conf=0.85):
    return EnrichedDecisionMaker(
        decision_maker=_make_dm(),
        email_result=EmailResult(email=email, confidence=conf, source="pattern+smtp", smtp_verified=True),
    )


def _make_messages(
    subject_a="The $3M question", subject_b="Cut admin by 50%",
    body=None, linkedin_dm="Hi Alice, congrats on the raise!", reply=8, flags=None
):
    if body is None:
        body = (
            "Saw the January seed — great milestone for the team. "
            "Growing fast usually means admin overhead scales too.\n\n"
            "We help edtech companies cut course coordination time by half. "
            "Worth a 20-minute call this week?\n\n"
            "Best, Alex | eQOURSE x TUTRAIN"
        )
    return GeneratedMessages(
        email_subject_a=subject_a, email_subject_b=subject_b,
        email_body=body, linkedin_dm=linkedin_dm,
        reply_likelihood=reply, quality_flags=flags or [],
    )


def _make_hook(quality="high"):
    return PersonalizationContext(
        domain="acme.com", company_one_liner="Acme builds AI tools.",
        recent_milestone="Raised $3M Jan 2026",
        pain_hypothesis_specific="Admin overhead scales with growth.",
        why_now_hook="Saw the Jan raise.",
        personalization_quality=quality,  # type: ignore[arg-type]
        built_at=_now(),
    )


def _make_mdm(email="alice@acme.com", conf=0.85, messages=None, skipped=None):
    return MessagedDecisionMaker(
        enriched_dm=_make_edm(email=email, conf=conf),
        messages=messages or _make_messages(),
        skipped_reason=skipped,
    )


def _make_ec(domain="acme.com", tier="tier_1"):
    qc = _make_qualified(domain=domain, tier=tier)
    cwp = QualifiedCandidateWithPeople(
        qualified=qc, decision_makers=[_make_dm()],
        lookup_status="found", lookup_attempts={},
    )
    return EnrichedCandidate(
        candidate_with_people=cwp,
        enriched_dms=[_make_edm()],
        enrichment_status="full",
    )


def _make_mc(domain="acme.com", tier="tier_1", mdm=None, hook=None):
    ec = _make_ec(domain=domain, tier=tier)
    return MessagedCandidate(
        enriched_candidate=ec,
        personalization=hook or _make_hook(),
        messaged_dms=[mdm if mdm is not None else _make_mdm()],
    )


def _make_messaged(mcs, segment="eqourse_ai_data", run_id="run-id"):
    now = _now()
    return MessagedResult(
        segment=segment, run_id=run_id,
        messaged_candidates=mcs, stats={}, api_credits_used={},
        started_at=now, completed_at=now, duration_seconds=0.1,
    )


def _lead_hash(domain="acme.com", name="Alice Smith"):
    return hashlib.sha256(f"{domain}|{name.lower()}".encode()).hexdigest()


@pytest.fixture
def mock_gemini():
    from pydantic import BaseModel
    class _Aligned(BaseModel):
        aligned: bool
        reason: str
    g = MagicMock()
    g.generate_json = AsyncMock(return_value=_Aligned(aligned=True, reason="looks good"))
    return g


@pytest.fixture
def mock_smtp():
    s = MagicMock()
    s.verify_email = AsyncMock()
    return s


@pytest.fixture
def mock_lead_store():
    s = MagicMock()
    s.update_run = AsyncMock()
    return s


@pytest.fixture
def validator(test_settings, mock_lead_store, mock_smtp, mock_gemini):
    return Validator(
        settings=test_settings,
        lead_store=mock_lead_store,
        smtp_verifier=mock_smtp,
        gemini_agent=mock_gemini,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ready_to_send_when_all_pass(validator):
    mc = _make_mc()
    result = await validator.validate(_make_messaged([mc]))
    vdm = result.validated_candidates[0].validated_dms[0]
    assert vdm.status == "ready_to_send"
    assert vdm.validation_reasons == []


# ---------------------------------------------------------------------------
# Hard rejections
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_email_passes(validator):
    mc = _make_mc()
    mc.messaged_dms[0].enriched_dm.email_result.email = "alice@acme.com"
    result = await validator.validate(_make_messaged([mc]))
    vdm = result.validated_candidates[0].validated_dms[0]
    assert "invalid_email_syntax" not in vdm.validation_reasons


@pytest.mark.asyncio
async def test_invalid_email_syntax_rejected(validator):
    mdm = _make_mdm(email="not-an-email")
    mc = _make_mc(mdm=mdm)
    result = await validator.validate(_make_messaged([mc]))
    vdm = result.validated_candidates[0].validated_dms[0]
    assert vdm.status == "rejected"
    assert "invalid_email_syntax" in vdm.validation_reasons


@pytest.mark.asyncio
async def test_missing_mx_rejected(validator, test_settings):
    """Cache hit with mx_records_found=False → no_mx_record → rejected."""
    smtp_cache_data = {
        "email": "alice@nomx.com", "exists": False,
        "smtp_response": "no_mx", "mx_records_found": False, "duration_ms": 50,
    }
    from sources._cache import cache_set
    await cache_set("smtp_verifier", "alice@nomx.com", smtp_cache_data, 7, test_settings)

    mdm = _make_mdm(email="alice@nomx.com")
    mc = _make_mc(domain="nomx.com", mdm=mdm)
    result = await validator.validate(_make_messaged([mc]))
    vdm = result.validated_candidates[0].validated_dms[0]
    assert vdm.status == "rejected"
    assert "no_mx_record" in vdm.validation_reasons


@pytest.mark.asyncio
async def test_duplicate_lead_rejected(validator, test_settings):
    """lead_hash already exists in DB with non-rejected status → duplicate."""
    from scripts.init_db import init_db, Lead
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    import uuid

    init_db(test_settings.SQLITE_PATH)
    lh = _lead_hash("acme.com", "Alice Smith")
    engine = create_engine(f"sqlite:///{test_settings.SQLITE_PATH}", future=True)
    with Session(engine) as s:
        s.add(Lead(id=str(uuid.uuid4()), lead_hash=lh, status="ready_to_send"))
        s.commit()
    engine.dispose()

    mc = _make_mc()
    result = await validator.validate(_make_messaged([mc]))
    vdm = result.validated_candidates[0].validated_dms[0]
    assert vdm.status == "rejected"
    assert "duplicate_lead" in vdm.validation_reasons


@pytest.mark.asyncio
async def test_profanity_in_body_rejected(validator):
    msgs = _make_messages(body="You're a shit product. Best, X")
    mdm = _make_mdm(messages=msgs)
    mc = _make_mc(mdm=mdm)
    result = await validator.validate(_make_messaged([mc]))
    vdm = result.validated_candidates[0].validated_dms[0]
    assert vdm.status == "rejected"
    assert "contains_profanity" in vdm.validation_reasons


@pytest.mark.asyncio
async def test_no_messages_rejected(validator):
    mdm = MessagedDecisionMaker(
        enriched_dm=_make_edm(),
        messages=None,
        skipped_reason="no_email",
    )
    mc = MessagedCandidate(
        enriched_candidate=_make_ec(),
        personalization=_make_hook(),
        messaged_dms=[mdm],
    )
    result = await validator.validate(_make_messaged([mc]))
    vdm = result.validated_candidates[0].validated_dms[0]
    assert vdm.status == "rejected"
    assert "no_messages_generated" in vdm.validation_reasons


# ---------------------------------------------------------------------------
# Soft flags → needs_review
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_caps_subject_rejected(validator):
    msgs = _make_messages(subject_a="THIS IS ALL CAPS SUBJECT HERE NOW")
    mdm = _make_mdm(messages=msgs)
    mc = _make_mc(mdm=mdm)
    result = await validator.validate(_make_messaged([mc]))
    vdm = result.validated_candidates[0].validated_dms[0]
    assert vdm.status == "needs_review"
    assert "all_caps_subject" in vdm.validation_reasons


@pytest.mark.asyncio
async def test_url_in_linkedin_dm_flagged(validator):
    msgs = _make_messages(linkedin_dm="Check this out: https://example.com/link")
    mdm = _make_mdm(messages=msgs)
    mc = _make_mc(mdm=mdm)
    result = await validator.validate(_make_messaged([mc]))
    vdm = result.validated_candidates[0].validated_dms[0]
    assert "url_in_linkedin_dm" in vdm.validation_reasons


@pytest.mark.asyncio
async def test_low_reply_likelihood_needs_review(validator, mock_gemini):
    mock_gemini.generate_json = AsyncMock(return_value=None)  # skip alignment
    msgs = _make_messages(reply=3)
    mdm = _make_mdm(messages=msgs)
    mc = _make_mc(mdm=mdm)
    result = await validator.validate(_make_messaged([mc]))
    vdm = result.validated_candidates[0].validated_dms[0]
    assert vdm.status == "needs_review"
    assert "low_reply_likelihood" in vdm.validation_reasons


@pytest.mark.asyncio
async def test_too_many_quality_flags_needs_review(validator, mock_gemini):
    mock_gemini.generate_json = AsyncMock(return_value=None)
    msgs = _make_messages(flags=["flag1", "flag2", "flag3"])
    mdm = _make_mdm(messages=msgs)
    mc = _make_mc(mdm=mdm)
    result = await validator.validate(_make_messaged([mc]), max_quality_flags_for_ready=1)
    vdm = result.validated_candidates[0].validated_dms[0]
    assert "too_many_quality_flags" in vdm.validation_reasons


@pytest.mark.asyncio
async def test_weak_personalization_needs_review(validator, mock_gemini):
    mock_gemini.generate_json = AsyncMock(return_value=None)
    hook = _make_hook(quality="low")
    mc = _make_mc(hook=hook)
    result = await validator.validate(_make_messaged([mc]))
    vdm = result.validated_candidates[0].validated_dms[0]
    assert "weak_personalization" in vdm.validation_reasons


@pytest.mark.asyncio
async def test_subject_body_mismatch_flag(validator, mock_gemini):
    from pydantic import BaseModel
    class _Resp(BaseModel):
        aligned: bool
        reason: str
    mock_gemini.generate_json = AsyncMock(return_value=_Resp(aligned=False, reason="mismatch"))
    mc = _make_mc()
    result = await validator.validate(_make_messaged([mc]))
    vdm = result.validated_candidates[0].validated_dms[0]
    assert "subject_body_mismatch" in vdm.validation_reasons


@pytest.mark.asyncio
async def test_subject_body_alignment_skipped_for_volume(validator, mock_gemini):
    """With >50 leads, alignment check should be skipped."""
    mcs = [_make_mc(domain=f"co{i}.com") for i in range(51)]
    result = await validator.validate(_make_messaged(mcs))
    # All DMs should have alignment_check_skipped_for_volume
    for vc in result.validated_candidates:
        for vdm in vc.validated_dms:
            if vdm.status != "rejected":
                assert "alignment_check_skipped_for_volume" in vdm.validation_reasons
    mock_gemini.generate_json.assert_not_called()


def test_lead_hash_deterministic():
    h1 = hashlib.sha256("acme.com|alice smith".encode()).hexdigest()
    h2 = hashlib.sha256("acme.com|alice smith".encode()).hexdigest()
    assert h1 == h2
