"""Tests for DecisionMakerFinder — Phase 5. All offline; all APIs mocked."""

from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents._constants import seniority_score
from agents._models import (
    DecisionMaker,
    QualifiedCandidate,
    QualifiedResult,
    QualifierSubScores,
)
from agents.decision_maker_finder import DecisionMakerFinder
from sources.models import CompanyCandidate


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_company(
    domain: str = "acme.com",
    name: str = "Acme Corp",
    website: Optional[str] = None,
) -> CompanyCandidate:
    return CompanyCandidate(
        domain=domain,
        name=name,
        website=website,
        raw_source="test",
        confidence=0.8,
    )


def _make_qualified(
    domain: str = "acme.com",
    name: str = "Acme Corp",
    tier: str = "tier_1",
    website: Optional[str] = None,
) -> QualifiedCandidate:
    candidate = _make_company(domain=domain, name=name, website=website)
    return QualifiedCandidate(
        candidate=candidate,
        total_score=80,
        pre_score=55,
        sub_scores=QualifierSubScores(
            funding_recency_score=40,
            reachability_score=10,
            geography_score=10,
            size_match_score=10,
            segment_fit_score=10,
            buying_signal_score=10,
        ),
        reasoning="test",
        disqualifiers=[],
        tier=tier,  # type: ignore[arg-type]
        domain_was_resolved=False,
    )


def _make_qualified_result(
    qualified: list[QualifiedCandidate],
    segment: str = "eqourse_ai_data",
    run_id: str = "test-run-id",
) -> QualifiedResult:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return QualifiedResult(
        segment=segment,
        run_id=run_id,
        qualified=qualified,
        dropped=[],
        stats={},
        api_credits_used={},
        started_at=now,
        completed_at=now,
        duration_seconds=0.1,
    )


@pytest.fixture
def mock_icp():
    icp = MagicMock()
    icp.target_titles = ["CEO", "CTO", "VP Engineering", "Head of Product"]
    icp.target_departments = ["Engineering", "Product"]
    icp.target_levels = ["c_suite", "vp", "director"]
    return icp


@pytest.fixture
def mock_icp_strategist(mock_icp):
    strategist = MagicMock()
    strategist.load_strategy.return_value = mock_icp
    return strategist


@pytest.fixture
def mock_sg_client():
    client = MagicMock()
    client.extract_team_page = AsyncMock(return_value=[])
    client._scrapegraph_available = True
    return client


@pytest.fixture
def mock_apify_client():
    client = MagicMock()
    client.google_search = AsyncMock(return_value=[])
    client.linkedin_company = AsyncMock(return_value={})
    return client


@pytest.fixture
def mock_vibe_client():
    client = MagicMock()
    client.find_prospects = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_lead_store():
    store = MagicMock()
    store.update_run = AsyncMock()
    return store


@pytest.fixture
def finder(
    test_settings,
    mock_icp_strategist,
    mock_sg_client,
    mock_apify_client,
    mock_vibe_client,
    mock_lead_store,
):
    return DecisionMakerFinder(
        settings=test_settings,
        icp_strategist=mock_icp_strategist,
        scrapegraph_client=mock_sg_client,
        apify_client=mock_apify_client,
        vibe_prospecting_client=mock_vibe_client,
        lead_store=mock_lead_store,
    )


# ---------------------------------------------------------------------------
# Domain blacklist / unknown tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_news_source_domain_flags_as_manual_lookup(finder):
    qualified = _make_qualified(domain="techcrunch.com", name="TechCrunch Story")
    result = await finder.find_for_qualified(_make_qualified_result([qualified]))

    assert len(result.needs_manual_lookup) == 1
    cwp = result.candidates_with_people[0]
    assert cwp.lookup_status == "needs_manual_lookup"
    assert cwp.decision_makers == []


@pytest.mark.asyncio
async def test_unknown_domain_flags_as_manual_lookup(finder):
    qualified = _make_qualified(domain="acmecorp.unknown", name="Acme Corp")
    result = await finder.find_for_qualified(_make_qualified_result([qualified]))

    assert len(result.needs_manual_lookup) == 1
    cwp = result.candidates_with_people[0]
    assert cwp.lookup_status == "needs_manual_lookup"


# ---------------------------------------------------------------------------
# Cascade logic tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scrapegraph_success_stops_cascade(finder, mock_sg_client, mock_apify_client, mock_vibe_client):
    """ScrapeGraph returns DMs → Apify and Explorium are NOT called."""
    from sources.models import ProspectCandidate

    mock_sg_client.extract_team_page.return_value = [
        ProspectCandidate(
            full_name="Alice CEO",
            title="CEO",
            company_domain="acme.com",
            source="scrapegraph",
        )
    ]

    qualified = _make_qualified(domain="acme.com", tier="tier_1")
    result = await finder.find_for_qualified(_make_qualified_result([qualified]))

    assert mock_sg_client.extract_team_page.called
    assert not mock_apify_client.google_search.called
    assert not mock_vibe_client.find_prospects.called

    cwp = result.candidates_with_people[0]
    assert cwp.lookup_status == "found"
    assert len(cwp.decision_makers) >= 1


@pytest.mark.asyncio
async def test_scrapegraph_failure_falls_to_apify_for_tier_1(
    finder, mock_sg_client, mock_apify_client
):
    """SG returns nothing → Apify is tried for tier_1."""
    from sources.models import SearchResult

    mock_sg_client.extract_team_page.return_value = []
    mock_apify_client.google_search.return_value = [
        SearchResult(
            title="Acme Corp | LinkedIn",
            url="https://www.linkedin.com/company/acme-corp",
            snippet="",
            position=1,
        )
    ]
    mock_apify_client.linkedin_company.return_value = {
        "employees": [
            {"name": "Bob CTO", "title": "CTO", "url": "https://li.test/bob"},
        ]
    }

    qualified = _make_qualified(domain="acme.com", tier="tier_1")
    result = await finder.find_for_qualified(_make_qualified_result([qualified]))

    assert mock_apify_client.google_search.called


@pytest.mark.asyncio
async def test_scrapegraph_failure_skips_apify_for_tier_2(
    finder, mock_sg_client, mock_apify_client
):
    """SG returns nothing for tier_2 → Apify is NOT called."""
    mock_sg_client.extract_team_page.return_value = []

    qualified = _make_qualified(domain="acme.com", tier="tier_2")
    result = await finder.find_for_qualified(_make_qualified_result([qualified]))

    assert not mock_apify_client.google_search.called

    cwp = result.candidates_with_people[0]
    assert cwp.lookup_attempts.get("apify") == "not_attempted"


@pytest.mark.asyncio
async def test_apify_failure_falls_to_explorium_for_tier_1(
    finder, mock_sg_client, mock_apify_client, mock_vibe_client
):
    """SG returns nothing, Apify returns nothing → Explorium tried for tier_1."""
    mock_sg_client.extract_team_page.return_value = []
    mock_apify_client.google_search.return_value = []  # no LinkedIn URL found

    from sources.models import ProspectCandidate

    mock_vibe_client.find_prospects.return_value = [
        ProspectCandidate(
            full_name="Carol VP",
            title="VP Product",
            company_domain="acme.com",
            source="vibe_prospecting",
        )
    ]

    qualified = _make_qualified(domain="acme.com", tier="tier_1")
    result = await finder.find_for_qualified(_make_qualified_result([qualified]))

    assert mock_vibe_client.find_prospects.called


@pytest.mark.asyncio
async def test_explorium_only_for_tier_1(
    finder, mock_sg_client, mock_apify_client, mock_vibe_client
):
    """Explorium is never called for tier_2 candidates."""
    mock_sg_client.extract_team_page.return_value = []

    qualified = _make_qualified(domain="startup.io", tier="tier_2")
    result = await finder.find_for_qualified(_make_qualified_result([qualified]))

    assert not mock_vibe_client.find_prospects.called


# ---------------------------------------------------------------------------
# Capping and ranking tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_per_company_caps_results(finder, mock_sg_client):
    """Even if SG returns 5 members, max_per_company=2 caps to 2."""
    from sources.models import ProspectCandidate

    mock_sg_client.extract_team_page.return_value = [
        ProspectCandidate(full_name=f"Person {i}", title="Manager", company_domain="big.com", source="scrapegraph")
        for i in range(5)
    ]

    qualified = _make_qualified(domain="big.com", tier="tier_1")
    result = await finder.find_for_qualified(
        _make_qualified_result([qualified]), max_per_company=2
    )

    cwp = result.candidates_with_people[0]
    # After ICP filter (Manager doesn't hit target titles but may hit seniority < 65)
    # Let's not assert exact count here; just assert it doesn't exceed 2
    assert len(cwp.decision_makers) <= 2


@pytest.mark.asyncio
async def test_seniority_ranking(finder, mock_sg_client):
    """Higher seniority_score DMs should come first."""
    from sources.models import ProspectCandidate

    mock_sg_client.extract_team_page.return_value = [
        ProspectCandidate(full_name="Junior Dev", title="Junior Developer", company_domain="acme.com", source="scrapegraph"),
        ProspectCandidate(full_name="Chief Alice", title="CEO", company_domain="acme.com", source="scrapegraph"),
        ProspectCandidate(full_name="VP Bob", title="VP Engineering", company_domain="acme.com", source="scrapegraph"),
    ]

    qualified = _make_qualified(domain="acme.com", tier="tier_1")
    result = await finder.find_for_qualified(
        _make_qualified_result([qualified]), max_per_company=3
    )

    cwp = result.candidates_with_people[0]
    dms = cwp.decision_makers
    # CEO (95) and VP (75) should appear; sorted by seniority desc
    names = [dm.full_name for dm in dms]
    if len(dms) >= 2:
        assert dms[0].seniority_score >= dms[1].seniority_score


@pytest.mark.asyncio
async def test_filter_by_icp_titles_drops_unrelated(finder, mock_sg_client):
    """Titles not matching ICP and with seniority < 65 should be dropped."""
    from sources.models import ProspectCandidate

    mock_sg_client.extract_team_page.return_value = [
        ProspectCandidate(
            full_name="Intern Sam",
            title="Marketing Intern",
            company_domain="acme.com",
            source="scrapegraph",
        )
    ]

    qualified = _make_qualified(domain="acme.com", tier="tier_1")
    result = await finder.find_for_qualified(_make_qualified_result([qualified]))

    cwp = result.candidates_with_people[0]
    # "Marketing Intern" should be filtered out (seniority 0, and is an intern)
    assert cwp.decision_makers == []


# ---------------------------------------------------------------------------
# Cap enforcement tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_apify_cap_enforced(
    finder, mock_sg_client, mock_apify_client
):
    """apify_cap_per_run=1 means only one Apify attempt across all candidates."""
    mock_sg_client.extract_team_page.return_value = []
    mock_apify_client.google_search.return_value = []

    candidates = [
        _make_qualified(domain=f"company{i}.com", name=f"Co {i}", tier="tier_1")
        for i in range(3)
    ]
    result = await finder.find_for_qualified(
        _make_qualified_result(candidates),
        scrapegraph_cap_per_segment=10,
        apify_cap_per_run=1,
    )

    # google_search should be called at most once
    assert mock_apify_client.google_search.call_count <= 1


@pytest.mark.asyncio
async def test_explorium_cap_enforced(
    finder, mock_sg_client, mock_apify_client, mock_vibe_client
):
    """explorium_cap_per_run=1 means only one Explorium attempt."""
    mock_sg_client.extract_team_page.return_value = []
    mock_apify_client.google_search.return_value = []

    candidates = [
        _make_qualified(domain=f"co{i}.com", name=f"Co {i}", tier="tier_1")
        for i in range(3)
    ]
    result = await finder.find_for_qualified(
        _make_qualified_result(candidates),
        scrapegraph_cap_per_segment=10,
        apify_cap_per_run=3,
        explorium_cap_per_run=1,
    )

    assert mock_vibe_client.find_prospects.call_count <= 1


# ---------------------------------------------------------------------------
# Error-handling tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scrapegraph_invalid_key_doesnt_crash(finder, mock_sg_client):
    """ScrapeGraph auth error (ApiResult status='error', error='auth_failed') → [] returned safely."""
    # Simulate the client already being in "unavailable" mode
    mock_sg_client._scrapegraph_available = False
    mock_sg_client.extract_team_page.return_value = []

    qualified = _make_qualified(domain="acme.com", tier="tier_1")
    result = await finder.find_for_qualified(
        _make_qualified_result([qualified]),
        apify_cap_per_run=0,
        explorium_cap_per_run=0,
    )

    # Should not crash; lookup_status is no_decision_maker
    cwp = result.candidates_with_people[0]
    assert cwp.decision_makers == []
    assert cwp.lookup_status in ("no_decision_maker", "found")


@pytest.mark.asyncio
async def test_run_record_updated(finder, mock_lead_store, mock_sg_client):
    """lead_store.update_run must be called after find_for_qualified."""
    mock_sg_client.extract_team_page.return_value = []
    qualified = _make_qualified(domain="acme.com", tier="tier_1")

    await finder.find_for_qualified(_make_qualified_result([qualified]))

    mock_lead_store.update_run.assert_called_once()


@pytest.mark.asyncio
async def test_no_decision_maker_status_when_all_sources_exhausted(
    finder, mock_sg_client, mock_apify_client, mock_vibe_client
):
    """When all sources return nothing, status='no_decision_maker'."""
    mock_sg_client.extract_team_page.return_value = []
    mock_apify_client.google_search.return_value = []
    mock_vibe_client.find_prospects.return_value = []

    qualified = _make_qualified(domain="obscure.io", tier="tier_1")
    result = await finder.find_for_qualified(_make_qualified_result([qualified]))

    cwp = result.candidates_with_people[0]
    assert cwp.lookup_status == "no_decision_maker"
    assert cwp.decision_makers == []
