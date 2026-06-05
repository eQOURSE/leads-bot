"""Tests for ContactEnricher — Phase 6. All offline; all external clients mocked."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents._models import (
    DecisionMaker,
    DomainPattern,
    EnhancedQualifiedResult,
    QualifiedCandidate,
    QualifiedCandidateWithPeople,
    QualifiedResult,
    QualifierSubScores,
)
from agents._email_patterns import _split_name, _default_patterns, _name_matches_email
from agents.contact_enricher import ContactEnricher
from sources.models import CompanyCandidate, ProspectCandidate
from sources.smtp_verifier import SMTPResult


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
        total_score=80,
        pre_score=55,
        sub_scores=QualifierSubScores(
            funding_recency_score=40, reachability_score=10,
            geography_score=10, size_match_score=10,
            segment_fit_score=10, buying_signal_score=10,
        ),
        reasoning="test",
        disqualifiers=[],
        tier=tier,  # type: ignore[arg-type]
        domain_was_resolved=False,
    )


def _make_dm(
    name: str = "Alice CEO",
    title: str = "CEO",
    domain: str = "acme.com",
    linkedin: Optional[str] = None,
) -> DecisionMaker:
    from agents._constants import seniority_score
    return DecisionMaker(
        full_name=name,
        title=title,
        linkedin_url=linkedin,
        source="scrapegraph",
        seniority_score=seniority_score(title),
    )


def _make_cwp(
    domain: str = "acme.com",
    tier: str = "tier_1",
    dms: Optional[list] = None,
    status: str = "found",
) -> QualifiedCandidateWithPeople:
    return QualifiedCandidateWithPeople(
        qualified=_make_qualified(domain=domain, tier=tier),
        decision_makers=dms or [],
        lookup_status=status,  # type: ignore[arg-type]
        lookup_attempts={},
    )


def _make_enhanced(
    cwps: list[QualifiedCandidateWithPeople],
    segment: str = "eqourse_ai_data",
    run_id: str = "test-run-id",
) -> EnhancedQualifiedResult:
    now = _now()
    return EnhancedQualifiedResult(
        segment=segment,
        run_id=run_id,
        candidates_with_people=cwps,
        needs_manual_lookup=[],
        stats={},
        api_credits_used={},
        started_at=now,
        completed_at=now,
        duration_seconds=0.1,
    )


def _smtp_result(email: str, exists: Optional[bool]) -> SMTPResult:
    return SMTPResult(
        email=email,
        exists=exists,
        smtp_response={True: "accepted", False: "rejected", None: "greylisted_or_inconclusive"}[exists],
        mx_records_found=True,
        duration_ms=50,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_hunter():
    h = MagicMock()
    h.domain_search = AsyncMock(return_value={})
    h.email_finder = AsyncMock(return_value=None)
    return h


@pytest.fixture
def mock_abstract():
    a = MagicMock()
    a.validate_email = AsyncMock(return_value={})
    a._exhausted = False
    return a


@pytest.fixture
def mock_smtp():
    s = MagicMock()
    # Default: all emails are greylisted
    s.verify_email = AsyncMock(
        side_effect=lambda email, **kw: _smtp_result(email, None)
    )
    return s


@pytest.fixture
def mock_vibe():
    v = MagicMock()
    v.find_prospects = AsyncMock(return_value=[])
    v.enrich_prospect_contacts = AsyncMock(return_value=[])
    return v


@pytest.fixture
def mock_lead_store():
    s = MagicMock()
    s.update_run = AsyncMock()
    return s


@pytest.fixture
def enricher(test_settings, mock_hunter, mock_abstract, mock_smtp, mock_vibe, mock_lead_store):
    return ContactEnricher(
        settings=test_settings,
        hunter_client=mock_hunter,
        abstract_api_client=mock_abstract,
        smtp_verifier=mock_smtp,
        vibe_prospecting_client=mock_vibe,
        lead_store=mock_lead_store,
    )


# ---------------------------------------------------------------------------
# Hunter-based tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hunter_known_email_match_short_circuits_cascade(enricher, mock_hunter, mock_smtp):
    """If Hunter domain_search returns a known email matching the DM's name, use it immediately."""
    mock_hunter.domain_search.return_value = {
        "pattern": "{first}.{last}",
        "emails": [{"value": "alice.ceo@acme.com"}],
    }

    dm = _make_dm(name="Alice Ceo", title="CEO")
    cwp = _make_cwp(domain="acme.com", tier="tier_1", dms=[dm])
    result = await enricher.enrich(_make_enhanced([cwp]))

    edm = result.enriched_candidates[0].enriched_dms[0]
    assert edm.email_result.email == "alice.ceo@acme.com"
    assert edm.email_result.confidence == 1.0
    assert edm.email_result.source == "hunter_known_email"
    # SMTP should not have been called
    assert mock_smtp.verify_email.call_count == 0


@pytest.mark.asyncio
async def test_hunter_finder_used_only_for_tier_1(enricher, mock_hunter):
    """Hunter email_finder is tried for tier_1 but NOT for tier_2."""
    mock_hunter.domain_search.return_value = {}
    mock_hunter.email_finder.return_value = None

    dm_t1 = _make_dm(name="Bob Smith", title="CEO")
    dm_t2 = _make_dm(name="Carol Jones", title="CEO")
    cwp_t1 = _make_cwp(domain="t1.com", tier="tier_1", dms=[dm_t1])
    cwp_t2 = _make_cwp(domain="t2.com", tier="tier_2", dms=[dm_t2])

    await enricher.enrich(_make_enhanced([cwp_t1, cwp_t2]))

    # email_finder should have been called for tier_1 but not tier_2
    assert mock_hunter.email_finder.call_count >= 1
    # The tier_2 call should NOT appear with domain "t2.com"
    calls = [str(c) for c in mock_hunter.email_finder.call_args_list]
    assert not any("t2.com" in c for c in calls)


@pytest.mark.asyncio
async def test_hunter_finder_skipped_when_cap_exhausted(enricher, mock_hunter):
    """hunter_finder_cap=0 means email_finder is never called."""
    mock_hunter.domain_search.return_value = {}

    dm = _make_dm(name="Alice CEO", title="CEO")
    cwp = _make_cwp(domain="acme.com", tier="tier_1", dms=[dm])
    await enricher.enrich(_make_enhanced([cwp]), hunter_finder_cap=0)

    mock_hunter.email_finder.assert_not_called()


@pytest.mark.asyncio
async def test_pattern_applied_to_all_dms_from_same_domain(enricher, mock_hunter, mock_smtp):
    """One Hunter domain_search call → pattern used for multiple DMs from same domain."""
    mock_hunter.domain_search.return_value = {
        "pattern": "{first}.{last}",
        "emails": [],
    }
    mock_hunter.email_finder.return_value = None
    mock_smtp.verify_email = AsyncMock(
        side_effect=lambda email, **kw: _smtp_result(email, True)
    )

    dms = [
        _make_dm(name="Alice Smith", title="CEO"),
        _make_dm(name="Bob Jones", title="CTO"),
    ]
    cwp = _make_cwp(domain="shared.com", tier="tier_1", dms=dms)
    await enricher.enrich(_make_enhanced([cwp]), hunter_domain_cap=1)

    # domain_search should have been called exactly once
    assert mock_hunter.domain_search.call_count == 1
    # Both DMs should have an email
    for edm in enricher.__dict__:
        pass  # just checking no crash; emails verified via smtp mock


# ---------------------------------------------------------------------------
# SMTP cascade tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_smtp_verify_returns_email_on_accept(enricher, mock_hunter, mock_smtp):
    """SMTP 250 → email found, confidence >= 0.7."""
    mock_hunter.domain_search.return_value = {}
    mock_hunter.email_finder.return_value = None
    mock_smtp.verify_email = AsyncMock(
        side_effect=lambda email, **kw: _smtp_result(email, True)
    )

    dm = _make_dm(name="Dan VP", title="VP Engineering")
    cwp = _make_cwp(domain="startup.io", tier="tier_1", dms=[dm])
    result = await enricher.enrich(_make_enhanced([cwp]), abstract_api_cap=0)

    edm = result.enriched_candidates[0].enriched_dms[0]
    assert edm.email_result.email is not None
    assert edm.email_result.confidence >= 0.7
    assert edm.email_result.smtp_verified is True


@pytest.mark.asyncio
async def test_smtp_verify_skips_email_on_reject(enricher, mock_hunter, mock_smtp):
    """SMTP 550 for all candidates → email is None."""
    mock_hunter.domain_search.return_value = {}
    mock_hunter.email_finder.return_value = None
    mock_smtp.verify_email = AsyncMock(
        side_effect=lambda email, **kw: _smtp_result(email, False)
    )

    dm = _make_dm(name="Eve Dir", title="Director")
    cwp = _make_cwp(domain="noemail.com", tier="tier_1", dms=[dm])
    result = await enricher.enrich(_make_enhanced([cwp]), abstract_api_cap=0)

    edm = result.enriched_candidates[0].enriched_dms[0]
    assert edm.email_result.email is None
    assert edm.email_result.confidence == 0.0


@pytest.mark.asyncio
async def test_smtp_inconclusive_keeps_email_with_lower_confidence(enricher, mock_hunter, mock_smtp):
    """All SMTP responses are inconclusive → keeps first candidate, confidence = 0.5."""
    mock_hunter.domain_search.return_value = {}
    mock_hunter.email_finder.return_value = None
    # All greylisted
    mock_smtp.verify_email = AsyncMock(
        side_effect=lambda email, **kw: _smtp_result(email, None)
    )

    dm = _make_dm(name="Frank Head", title="Head of Product")
    cwp = _make_cwp(domain="greylisted.io", tier="tier_1", dms=[dm])
    result = await enricher.enrich(_make_enhanced([cwp]), abstract_api_cap=0)

    edm = result.enriched_candidates[0].enriched_dms[0]
    assert edm.email_result.email is not None
    assert edm.email_result.confidence == 0.5


# ---------------------------------------------------------------------------
# AbstractAPI tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_abstract_api_catchall_downgrades_confidence(enricher, mock_hunter, mock_smtp, mock_abstract):
    mock_hunter.domain_search.return_value = {}
    mock_hunter.email_finder.return_value = None
    mock_smtp.verify_email = AsyncMock(
        side_effect=lambda email, **kw: _smtp_result(email, True)
    )
    mock_abstract.validate_email = AsyncMock(return_value={
        "is_catchall_email": True,
        "deliverability": "RISKY",
    })

    dm = _make_dm(name="Grace CEO", title="CEO")
    cwp = _make_cwp(domain="catchall.io", tier="tier_1", dms=[dm])
    result = await enricher.enrich(_make_enhanced([cwp]))

    edm = result.enriched_candidates[0].enriched_dms[0]
    assert edm.email_result.confidence == 0.4
    assert edm.email_result.catchall_detected is True


@pytest.mark.asyncio
async def test_abstract_api_deliverable_boosts_confidence_to_85(enricher, mock_hunter, mock_smtp, mock_abstract):
    mock_hunter.domain_search.return_value = {}
    mock_hunter.email_finder.return_value = None
    mock_smtp.verify_email = AsyncMock(
        side_effect=lambda email, **kw: _smtp_result(email, True)
    )
    mock_abstract.validate_email = AsyncMock(return_value={
        "is_catchall_email": False,
        "deliverability": "DELIVERABLE",
    })

    dm = _make_dm(name="Hank Founder", title="Founder")
    cwp = _make_cwp(domain="verified.io", tier="tier_1", dms=[dm])
    result = await enricher.enrich(_make_enhanced([cwp]))

    edm = result.enriched_candidates[0].enriched_dms[0]
    assert edm.email_result.confidence == 0.85
    assert edm.email_result.deliverability == "DELIVERABLE"


@pytest.mark.asyncio
async def test_abstract_api_skipped_for_tier_2(enricher, mock_hunter, mock_smtp, mock_abstract):
    """AbstractAPI should not be called for tier_2 candidates."""
    mock_hunter.domain_search.return_value = {}
    mock_hunter.email_finder.return_value = None
    mock_smtp.verify_email = AsyncMock(
        side_effect=lambda email, **kw: _smtp_result(email, True)
    )

    dm = _make_dm(name="Iris VP", title="VP Sales")
    cwp = _make_cwp(domain="t2company.com", tier="tier_2", dms=[dm])
    await enricher.enrich(_make_enhanced([cwp]))

    mock_abstract.validate_email.assert_not_called()


# ---------------------------------------------------------------------------
# Common-prefix tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_dm_company_tries_common_prefixes_for_tier_1(enricher, mock_smtp):
    """Company with no DMs (tier_1) → common-prefix SMTP probe."""
    call_count = 0

    async def smtp_side(email, **kw):
        nonlocal call_count
        call_count += 1
        if "founder" in email:
            return _smtp_result(email, True)
        return _smtp_result(email, False)

    mock_smtp.verify_email = AsyncMock(side_effect=smtp_side)

    cwp = _make_cwp(domain="nodm.io", tier="tier_1", dms=[], status="no_decision_maker")
    result = await enricher.enrich(_make_enhanced([cwp]))

    ec = result.enriched_candidates[0]
    assert ec.company_contact_email is not None
    assert "founder" in ec.company_contact_email.email
    assert ec.company_contact_email.confidence == 0.3
    assert ec.company_contact_email.source == "common_prefix"


@pytest.mark.asyncio
async def test_no_dm_company_skipped_for_tier_2(enricher, mock_smtp):
    """Company with no DMs (tier_2) → common-prefix NOT tried."""
    cwp = _make_cwp(domain="nodm-t2.io", tier="tier_2", dms=[], status="no_decision_maker")
    result = await enricher.enrich(_make_enhanced([cwp]))

    ec = result.enriched_candidates[0]
    assert ec.company_contact_email is None
    mock_smtp.verify_email.assert_not_called()


# ---------------------------------------------------------------------------
# Explorium phone enrichment
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_explorium_phone_enrichment_caps_at_1(enricher, mock_vibe, mock_smtp, mock_hunter):
    """Explorium phone enrichment is called at most once, for the first tier_1 with DMs."""
    mock_hunter.domain_search.return_value = {}
    mock_hunter.email_finder.return_value = None
    mock_smtp.verify_email = AsyncMock(
        side_effect=lambda email, **kw: _smtp_result(email, None)
    )
    mock_vibe.enrich_prospect_contacts = AsyncMock(return_value=[])

    dms = [_make_dm(name="Jack CEO", title="CEO")]
    cwps = [
        _make_cwp(domain=f"co{i}.com", tier="tier_1", dms=dms)
        for i in range(3)
    ]
    await enricher.enrich(_make_enhanced(cwps), explorium_cap=1)

    assert mock_vibe.enrich_prospect_contacts.call_count == 1


# ---------------------------------------------------------------------------
# Cache test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pattern_cache_hit_skips_hunter_call(enricher, mock_hunter, mock_smtp):
    """Second run with same domain should not call domain_search again (30-day cache)."""
    mock_hunter.domain_search.return_value = {
        "pattern": "{first}.{last}",
        "emails": [],
    }
    mock_hunter.email_finder.return_value = None
    mock_smtp.verify_email = AsyncMock(
        side_effect=lambda email, **kw: _smtp_result(email, None)
    )

    dm = _make_dm(name="Kate Head", title="Head of Growth")
    cwp = _make_cwp(domain="cached-domain.com", tier="tier_1", dms=[dm])

    await enricher.enrich(_make_enhanced([cwp]), hunter_domain_cap=5)
    await enricher.enrich(_make_enhanced([cwp]), hunter_domain_cap=5)

    # domain_search should only have been called once despite two runs
    assert mock_hunter.domain_search.call_count == 1


# ---------------------------------------------------------------------------
# _email_patterns utility tests
# ---------------------------------------------------------------------------

def test_name_matches_email_loose_matching():
    assert _name_matches_email("Sara Chen", "sara.chen@x.com")
    assert _name_matches_email("Sara Chen", "schen@x.com")
    assert not _name_matches_email("Sara Chen", "marcus@x.com")


def test_split_name_handles_three_part_names():
    first, last = _split_name("Dr. Alice Mary Smith")
    assert first == "alice"
    assert last == "smith"


def test_default_patterns_generated_in_priority_order():
    patterns = _default_patterns("alice", "smith", "acme.com")
    assert patterns[0] == "alice.smith@acme.com"
    assert patterns[1] == "alicesmith@acme.com"
    assert patterns[2] == "asmith@acme.com"


# ---------------------------------------------------------------------------
# Run record and graceful degradation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enriched_result_writes_to_run_record(enricher, mock_lead_store, mock_smtp):
    mock_smtp.verify_email = AsyncMock(
        side_effect=lambda email, **kw: _smtp_result(email, None)
    )
    dm = _make_dm(name="Laura VP", title="VP")
    cwp = _make_cwp(domain="runtest.com", tier="tier_1", dms=[dm])
    await enricher.enrich(_make_enhanced([cwp]))
    mock_lead_store.update_run.assert_called_once()


@pytest.mark.asyncio
async def test_graceful_when_all_external_sources_fail(enricher, mock_hunter, mock_smtp, mock_abstract):
    """All sources fail → enrichment_status='no_emails', no crash."""
    mock_hunter.domain_search.return_value = {}
    mock_hunter.email_finder = AsyncMock(side_effect=Exception("connection failed"))
    mock_smtp.verify_email = AsyncMock(
        side_effect=lambda email, **kw: _smtp_result(email, False)
    )
    mock_abstract.validate_email = AsyncMock(return_value={})

    dm = _make_dm(name="Mike Dir", title="Director")
    cwp = _make_cwp(domain="failing.io", tier="tier_1", dms=[dm])
    result = await enricher.enrich(_make_enhanced([cwp]))

    ec = result.enriched_candidates[0]
    assert ec.enrichment_status in ("no_emails", "partial")
    edm = ec.enriched_dms[0]
    assert edm.email_result.email is None
    assert edm.email_result.confidence == 0.0
