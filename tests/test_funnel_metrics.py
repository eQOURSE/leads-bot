"""Phase 11 — tests for compute_funnel_metrics and hunter multi-source integration."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from orchestrator.nodes import compute_funnel_metrics
from orchestrator.state import make_initial_state


def test_compute_funnel_metrics_empty_state():
    state = make_initial_state("tutrain", "r1", 30)
    m = compute_funnel_metrics(state)
    assert m["funnel_drop_off"]["hunted_raw"] == 0
    assert m["funnel_drop_off"]["ready_to_send"] == 0
    assert "source_contributions" in m


def test_compute_funnel_metrics_with_results():
    from agents._models import HuntResult, QualifiedResult, ValidatedResult
    from sources.models import CompanyCandidate
    from datetime import date

    state = make_initial_state("tutrain", "r2", 30)

    candidates = [
        CompanyCandidate(domain="real.com", name="Real", raw_source="crunchbase", confidence=0.85),
        CompanyCandidate(domain="slug.unknown", name="Slug", raw_source="rss_feeds", confidence=0.5),
    ]
    state["hunt_result"] = HuntResult(
        segment="tutrain", run_id="r2", candidates=candidates,
        source_counts={"rss": 1, "serpapi": 0, "newsdata": 0, "crunchbase": 1, "wellfound": 0},
        merged_count=2, after_filter=2, after_dedupe=2, enriched_count=0,
        api_credits_used={}, errors=[],
        started_at=datetime.utcnow(), completed_at=datetime.utcnow(), duration_seconds=1.0,
    )
    state["qualified_result"] = QualifiedResult(
        segment="tutrain", run_id="r2", qualified=[], dropped=[],
        stats={"pre_score_filtered": 1}, api_credits_used={},
        started_at=datetime.utcnow(), completed_at=datetime.utcnow(), duration_seconds=1.0,
    )
    state["_hunter_metrics"] = {
        "article_link_resolution_rate": 0.5,
        "apify_discovery_spend_estimate_usd": 2.0,
        "source_contributions": {"rss": 1, "crunchbase": 1},
    }
    state["_gemini_metrics"] = {"retry_count": 2, "fallback_count": 1}

    m = compute_funnel_metrics(state)
    assert m["funnel_drop_off"]["hunted_raw"] == 2
    assert m["funnel_drop_off"]["after_domain_resolution"] == 1  # only real.com
    assert m["article_link_resolution_rate"] == 0.5
    assert m["gemini_retry_count"] == 2
    assert m["gemini_fallback_count"] == 1
    assert m["apify_spend_estimate_usd"] == 2.0


@pytest.mark.asyncio
async def test_hunter_integrates_crunchbase_and_wellfound(test_settings):
    """Hunter.hunt should call the new sources and include them in source_counts."""
    from agents.company_hunter import CompanyHunter
    from sources.models import CompanyCandidate

    test_settings.ENABLE_CRUNCHBASE_DISCOVERY = True
    test_settings.ENABLE_WELLFOUND_DISCOVERY = True

    icp_strategist = MagicMock()
    icp = MagicMock()
    icp.target_industries.naics_codes = ["541512"]
    icp.target_industries.industry_keywords = ["ai"]
    icp.target_company_profile.funding_stages = ["Seed", "Series A"]
    icp.target_company_profile.funding_recency_days = 240
    icp.target_company_profile.geographies.countries = ["US"]
    icp.target_company_profile.founded_after_year = 2022
    icp_strategist.load_strategy.return_value = icp

    lead_store = MagicMock()
    lead_store.create_run = AsyncMock(return_value="run-cb")
    lead_store.update_run = AsyncMock()
    lead_store.mark_domains_seen = AsyncMock()
    lead_store.get_seen_domains_within = AsyncMock(return_value=set())

    rss = MagicMock()
    rss.reset_resolution_metrics = MagicMock()
    rss.article_link_resolution_rate = 0.6

    crunchbase = MagicMock()
    crunchbase.search_recent_funding = AsyncMock(return_value=[
        CompanyCandidate(domain=f"cb{i}.com", name=f"CB {i}", raw_source="crunchbase", confidence=0.85)
        for i in range(6)
    ])
    wellfound = MagicMock()
    wellfound.search_recent_startups = AsyncMock(return_value=[
        CompanyCandidate(domain=f"wf{i}.com", name=f"WF {i}", raw_source="wellfound", confidence=0.8)
        for i in range(5)
    ])

    hunter = CompanyHunter(
        test_settings, icp_strategist,
        rss_client=rss,
        serpapi_client=MagicMock(),
        newsdata_client=MagicMock(),
        companies_api_client=MagicMock(),
        lead_store=lead_store,
        crunchbase_client=crunchbase,
        wellfound_client=wellfound,
    )

    # Stub the RSS/SerpAPI/NewsData sub-hunts to return empty (focus on new sources).
    hunter._hunt_via_rss = AsyncMock(return_value=[])
    hunter._hunt_via_serpapi = AsyncMock(return_value=[])
    hunter._hunt_via_newsdata = AsyncMock(return_value=[])
    hunter._enrich_with_firmographics = AsyncMock(side_effect=lambda c, n: c)

    result = await hunter.hunt("eqourse_ai_data", target_count=30)

    crunchbase.search_recent_funding.assert_awaited_once()
    wellfound.search_recent_startups.assert_awaited_once()
    assert result.source_counts["crunchbase"] == 6
    assert result.source_counts["wellfound"] == 5
    # Both sets of candidates present in the merged output.
    names = {c.name for c in result.candidates}
    assert "CB 0" in names and "WF 0" in names
