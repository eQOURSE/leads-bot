"""Tests for Personalizer — Phase 7. All offline; all clients mocked."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents._models import (
    EnrichedCandidate,
    EnrichedResult,
    EmailResult,
    EnrichedDecisionMaker,
    DecisionMaker,
    PersonalizationContext,
    QualifiedCandidate,
    QualifiedCandidateWithPeople,
    QualifierSubScores,
)
from agents.personalizer import Personalizer
from sources.models import CompanyCandidate, NewsItem


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


def _make_dm(title: str = "CEO") -> DecisionMaker:
    from agents._constants import seniority_score as ss
    return DecisionMaker(
        full_name="Alice Smith", title=title,
        source="scrapegraph", seniority_score=ss(title),
    )


def _make_edm(email: Optional[str] = "alice@acme.com", confidence: float = 0.85):
    return EnrichedDecisionMaker(
        decision_maker=_make_dm(),
        email_result=EmailResult(
            email=email,
            confidence=confidence,
            source="pattern+smtp",
            smtp_verified=True,
        ),
    )


def _make_cwp(domain: str = "acme.com", tier: str = "tier_1") -> QualifiedCandidateWithPeople:
    return QualifiedCandidateWithPeople(
        qualified=_make_qualified(domain=domain, tier=tier),
        decision_makers=[_make_dm()],
        lookup_status="found",
        lookup_attempts={},
    )


def _make_ec(domain: str = "acme.com", tier: str = "tier_1", email: Optional[str] = "alice@acme.com") -> EnrichedCandidate:
    return EnrichedCandidate(
        candidate_with_people=_make_cwp(domain=domain, tier=tier),
        enriched_dms=[_make_edm(email=email)],
        enrichment_status="full",
    )


def _make_enriched(
    ecs: list[EnrichedCandidate],
    segment: str = "eqourse_ai_data",
    run_id: str = "run-id",
) -> EnrichedResult:
    now = _now()
    return EnrichedResult(
        segment=segment, run_id=run_id,
        enriched_candidates=ecs,
        stats={}, api_credits_used={},
        started_at=now, completed_at=now, duration_seconds=0.1,
    )


def _make_hook_raw():
    return MagicMock(
        company_one_liner="Acme builds AI tutoring tools.",
        recent_milestone="Raised $3M seed on Jan 2026",
        pain_hypothesis_specific="Their instructors spend 4h/day on admin.",
        why_now_hook="Saw the Jan 2026 seed — the timing is right for automation.",
        personalization_quality="high",
    )


@pytest.fixture
def mock_icp():
    icp = MagicMock()
    icp.value_prop_one_liner = "AI-powered course automation"
    icp.outreach_angle.pain_hypothesis = "Instructors waste time on manual admin"
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
    g.generate_json = AsyncMock(return_value=_make_hook_raw())
    return g


@pytest.fixture
def mock_scrapegraph():
    sg = MagicMock()
    sg.extract_recent_news = AsyncMock(return_value={"announcements": []})
    return sg


@pytest.fixture
def mock_newsdata():
    nd = MagicMock()
    nd.search_company_news = AsyncMock(return_value=[])
    return nd


@pytest.fixture
def mock_lead_store():
    s = MagicMock()
    s.update_run = AsyncMock()
    return s


@pytest.fixture
def personalizer(test_settings, mock_icp_strategist, mock_gemini, mock_scrapegraph, mock_newsdata, mock_lead_store):
    return Personalizer(
        settings=test_settings,
        icp_strategist=mock_icp_strategist,
        gemini_agent=mock_gemini,
        scrapegraph_client=mock_scrapegraph,
        newsdata_client=mock_newsdata,
        lead_store=mock_lead_store,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_builds_hook_for_company_with_recent_news(personalizer, mock_scrapegraph):
    mock_scrapegraph.extract_recent_news.return_value = {
        "announcements": [{"type": "funding", "date": "2026-01", "summary": "Raised $3M"}]
    }
    ec = _make_ec(domain="acme.com")
    result = await personalizer.build_hooks_for_enriched_result(_make_enriched([ec]))

    assert "acme.com" in result
    ctx = result["acme.com"]
    assert isinstance(ctx, PersonalizationContext)
    assert ctx.why_now_hook


@pytest.mark.asyncio
async def test_skips_company_with_no_email_dms(personalizer, mock_scrapegraph):
    """Companies where all DMs have no email should not be in the result."""
    ec = _make_ec(domain="noemail.com", email=None)
    result = await personalizer.build_hooks_for_enriched_result(_make_enriched([ec]))

    assert "noemail.com" not in result
    mock_scrapegraph.extract_recent_news.assert_not_called()


@pytest.mark.asyncio
async def test_processes_tier_1_before_tier_2(personalizer):
    """tier_1 domains should appear before tier_2 in processing order."""
    call_order = []

    async def track_sg(url, **kw):
        domain = url.replace("https://", "")
        call_order.append(domain)
        return {}

    personalizer.scrapegraph.extract_recent_news = AsyncMock(side_effect=track_sg)

    ec_t1 = _make_ec(domain="tier1co.com", tier="tier_1")
    ec_t2 = _make_ec(domain="tier2co.com", tier="tier_2")

    await personalizer.build_hooks_for_enriched_result(_make_enriched([ec_t2, ec_t1]))

    # tier_1 should have been processed first
    if call_order:
        assert call_order[0] == "tier1co.com"


@pytest.mark.asyncio
async def test_cache_hit_skips_scrapegraph_and_newsdata(personalizer, mock_scrapegraph, mock_newsdata):
    """Second call with same domain uses cache and skips both API clients."""
    ec = _make_ec(domain="cached.com")
    enriched = _make_enriched([ec])

    await personalizer.build_hooks_for_enriched_result(enriched)
    mock_scrapegraph.extract_recent_news.reset_mock()
    mock_newsdata.search_company_news.reset_mock()

    # Second call — same domain should be cached
    await personalizer.build_hooks_for_enriched_result(enriched)

    assert mock_scrapegraph.extract_recent_news.call_count == 0
    assert mock_newsdata.search_company_news.call_count == 0


@pytest.mark.asyncio
async def test_scrapegraph_cap_enforced(personalizer):
    """scrapegraph_cap_per_run=1 means only one SG call across all companies."""
    call_count = 0

    async def count_sg(url, **kw):
        nonlocal call_count
        call_count += 1
        return {}

    personalizer.scrapegraph.extract_recent_news = AsyncMock(side_effect=count_sg)

    ecs = [_make_ec(domain=f"co{i}.com") for i in range(3)]
    await personalizer.build_hooks_for_enriched_result(
        _make_enriched(ecs), scrapegraph_cap_per_run=1
    )

    assert call_count <= 1


@pytest.mark.asyncio
async def test_newsdata_cap_enforced(personalizer):
    """newsdata_cap_per_run=1 means only one NewsData call."""
    call_count = 0

    async def count_nd(name, **kw):
        nonlocal call_count
        call_count += 1
        return []

    personalizer.newsdata.search_company_news = AsyncMock(side_effect=count_nd)

    ecs = [_make_ec(domain=f"nd{i}.com") for i in range(3)]
    await personalizer.build_hooks_for_enriched_result(
        _make_enriched(ecs), newsdata_cap_per_run=1
    )

    assert call_count <= 1


@pytest.mark.asyncio
async def test_low_quality_hook_when_no_specific_data(personalizer, mock_gemini):
    """When Gemini says quality=low, it should be preserved in the context."""
    low_hook = _make_hook_raw()
    low_hook.recent_milestone = None
    low_hook.personalization_quality = "low"
    mock_gemini.generate_json = AsyncMock(return_value=low_hook)

    ec = _make_ec(domain="lowinfo.com")
    result = await personalizer.build_hooks_for_enriched_result(_make_enriched([ec]))

    assert result["lowinfo.com"].personalization_quality == "low"


@pytest.mark.asyncio
async def test_gemini_failure_returns_fallback_context(personalizer, mock_gemini):
    """Gemini returning None should produce a fallback PersonalizationContext."""
    mock_gemini.generate_json = AsyncMock(return_value=None)

    ec = _make_ec(domain="geminifail.com")
    result = await personalizer.build_hooks_for_enriched_result(_make_enriched([ec]))

    assert "geminifail.com" in result
    ctx = result["geminifail.com"]
    assert ctx.personalization_quality == "low"
    assert ctx.why_now_hook  # fallback always has some text


@pytest.mark.asyncio
async def test_dedupes_by_domain_across_candidates(personalizer, mock_scrapegraph):
    """Same domain appearing twice in enriched_candidates should only trigger one API call."""
    ec1 = _make_ec(domain="dupco.com")
    ec2 = _make_ec(domain="dupco.com")  # same domain
    call_count = 0

    async def count_sg(url, **kw):
        nonlocal call_count
        call_count += 1
        return {}

    personalizer.scrapegraph.extract_recent_news = AsyncMock(side_effect=count_sg)

    result = await personalizer.build_hooks_for_enriched_result(_make_enriched([ec1, ec2]))

    assert len(result) == 1  # one unique domain
    assert call_count <= 1
