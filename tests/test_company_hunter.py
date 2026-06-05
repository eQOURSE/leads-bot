"""Tests for CompanyHunter. All external calls are mocked; no real network."""

from __future__ import annotations

import re
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.company_hunter import CompanyHunter
from agents.icp_strategist import IcpStrategist
from sources.models import CompanyCandidate, NewsItem, SearchResult
from sources._utils import normalize_domain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    domain: str,
    name: str,
    *,
    raw_source: str = "rss",
    confidence: float = 0.6,
    funding_date: date | None = None,
    funding_stage: str | None = "seed",
    description: str | None = None,
    size_range: str | None = None,
    hq_country: str | None = None,
) -> CompanyCandidate:
    return CompanyCandidate(
        domain=domain,
        name=name,
        raw_source=raw_source,
        confidence=confidence,
        funding_date=funding_date,
        funding_stage=funding_stage,
        description=description,
        size_range=size_range,
        hq_country=hq_country,
    )


def _recent_date(days_ago: int = 10) -> date:
    return date.today() - timedelta(days=days_ago)


def _make_news(title: str, snippet: str = "") -> NewsItem:
    from datetime import datetime

    return NewsItem(
        title=title,
        url="https://example.com/news",
        published_at=datetime.utcnow(),
        source_name="test",
        snippet=snippet,
    )


def _mock_lead_store() -> MagicMock:
    store = MagicMock()
    store.create_run = AsyncMock(return_value="test-run-id")
    store.update_run = AsyncMock()
    store.mark_domains_seen = AsyncMock()
    store.get_seen_domains_within = AsyncMock(return_value=set())
    return store


def _make_hunter(test_settings, *, lead_store=None) -> CompanyHunter:
    strategist = IcpStrategist(test_settings)

    rss = MagicMock()
    rss.fetch_recent_funding = AsyncMock(return_value=[])
    rss.extract_company_from_headline = AsyncMock(return_value=None)

    serp = MagicMock()
    serp.search = AsyncMock(return_value=[])

    news = MagicMock()
    news.search_funding_news = AsyncMock(return_value=[])

    ca = MagicMock()
    ca.enrich_by_domain = AsyncMock(return_value=None)

    return CompanyHunter(
        settings=test_settings,
        icp_strategist=strategist,
        rss_client=rss,
        serpapi_client=serp,
        newsdata_client=news,
        companies_api_client=ca,
        lead_store=lead_store or _mock_lead_store(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hunt_happy_path_returns_candidates(test_settings):
    """All 3 sources return data; hunt produces a non-empty result."""
    hunter = _make_hunter(test_settings)

    # RSS returns one item that will be converted to a candidate
    news_item = _make_news("EdTech startup raises $5M seed round", "online learning platform")
    hunter.rss_client.fetch_recent_funding = AsyncMock(return_value=[news_item])
    hunter.rss_client.extract_company_from_headline = AsyncMock(
        return_value=_make_candidate(
            "learnco.com", "LearnCo", funding_date=_recent_date(5), raw_source="rss"
        )
    )

    # SerpAPI returns empty (Gemini batch mocked at hunter level)
    hunter.serpapi_client.search = AsyncMock(return_value=[])

    # NewsData returns nothing
    hunter.newsdata_client.search_funding_news = AsyncMock(return_value=[])

    result = await hunter.hunt("tutrain", target_count=10, enrichment_top_n=0)

    assert len(result.candidates) >= 1
    assert result.run_id == "test-run-id"
    assert result.source_counts["rss"] >= 1


@pytest.mark.asyncio
async def test_one_source_failure_doesnt_kill_run(test_settings):
    """If RSS raises, the run completes using SerpAPI + NewsData data."""
    hunter = _make_hunter(test_settings)

    # RSS raises
    hunter.rss_client.fetch_recent_funding = AsyncMock(
        side_effect=RuntimeError("feed timeout")
    )

    # SerpAPI returns a candidate (bypass Gemini batch via patching)
    serp_result = SearchResult(title="AI startup raises $3M", url="https://tc.com/ai", snippet="AI funding", position=1)
    hunter.serpapi_client.search = AsyncMock(return_value=[serp_result])

    with patch.object(
        hunter._gemini, "batch_generate_json",
        new=AsyncMock(return_value=[
            None  # Gemini returns None for this prompt (no candidate extracted)
        ]),
    ):
        result = await hunter.hunt("tutrain", target_count=10, enrichment_top_n=0)

    assert any("rss" in e.lower() for e in result.errors)
    # Run completed despite the failure
    assert result.run_id == "test-run-id"


@pytest.mark.asyncio
async def test_merge_dedupes_by_normalized_domain(test_settings):
    """www.acme.com and acme.com normalize to the same key and merge into one."""
    hunter = _make_hunter(test_settings)

    rss = [_make_candidate("www.acme.com", "Acme", raw_source="rss", funding_date=_recent_date(3))]
    serp = [_make_candidate("acme.com", "Acme Corp", raw_source="serpapi", funding_date=_recent_date(3))]

    merged = hunter._merge_candidates(rss, serp, [])

    assert len(merged) == 1
    assert merged[0].domain == "acme.com"


@pytest.mark.asyncio
async def test_merge_boosts_confidence_for_multi_source(test_settings):
    """Company seen in both RSS and SerpAPI gets +0.2 confidence boost."""
    hunter = _make_hunter(test_settings)

    rss = [_make_candidate("beta.com", "Beta", raw_source="rss", confidence=0.6, funding_date=_recent_date(3))]
    serp = [_make_candidate("beta.com", "Beta", raw_source="serpapi", confidence=0.7, funding_date=_recent_date(3))]

    merged = hunter._merge_candidates(rss, serp, [])

    assert len(merged) == 1
    assert merged[0].confidence >= 0.8
    assert "rss" in merged[0].raw_source
    assert "serpapi" in merged[0].raw_source


@pytest.mark.asyncio
async def test_icp_filter_drops_old_funding(test_settings):
    """Candidate with funding_date older than recency_days is dropped."""
    hunter = _make_hunter(test_settings)
    icp = IcpStrategist(test_settings).load_strategy("tutrain")

    # funding_date well outside the 240-day window
    old = _make_candidate("old.com", "OldCo", funding_date=date(2020, 1, 1))
    recent = _make_candidate("new.com", "NewCo", funding_date=_recent_date(30))

    filtered = hunter._apply_icp_filters([old, recent], icp)

    domains = [c.domain for c in filtered]
    assert "old.com" not in domains
    assert "new.com" in domains


@pytest.mark.asyncio
async def test_icp_filter_drops_negative_signal_match(test_settings):
    """Candidate whose description matches a negative signal is dropped."""
    hunter = _make_hunter(test_settings)
    icp = IcpStrategist(test_settings).load_strategy("tutrain")

    # "Brick-and-mortar tutoring center" is a tutrain negative signal
    bad = _make_candidate(
        "tutor.com", "TutorPlus",
        description="We operate brick-and-mortar tutoring centers in 30 cities",
        funding_date=_recent_date(20),
    )
    good = _make_candidate(
        "edco.com", "EdCo",
        description="Online learning platform for K-12 students",
        funding_date=_recent_date(20),
    )

    filtered = hunter._apply_icp_filters([bad, good], icp)

    domains = [c.domain for c in filtered]
    assert "tutor.com" not in domains
    assert "edco.com" in domains


@pytest.mark.asyncio
async def test_dedupe_skips_recently_seen_domains(test_settings):
    """Domains seen within the window are removed from candidates."""
    store = _mock_lead_store()
    store.get_seen_domains_within = AsyncMock(return_value={"acme.com", "beta.com"})

    hunter = _make_hunter(test_settings, lead_store=store)

    candidates = [
        _make_candidate("acme.com", "Acme"),
        _make_candidate("gamma.com", "Gamma"),
    ]

    survivors = await hunter._dedupe_against_seen(candidates, 30, bypass=False)

    domains = [c.domain for c in survivors]
    assert "acme.com" not in domains
    assert "gamma.com" in domains


@pytest.mark.asyncio
async def test_bypass_dedupe_flag_works(test_settings):
    """bypass=True returns all candidates regardless of seen_domains."""
    store = _mock_lead_store()
    store.get_seen_domains_within = AsyncMock(return_value={"acme.com", "beta.com"})

    hunter = _make_hunter(test_settings, lead_store=store)

    candidates = [
        _make_candidate("acme.com", "Acme"),
        _make_candidate("beta.com", "Beta"),
    ]

    survivors = await hunter._dedupe_against_seen(candidates, 30, bypass=True)

    assert len(survivors) == 2
    # get_seen_domains_within should never be called when bypassing
    store.get_seen_domains_within.assert_not_called()


@pytest.mark.asyncio
async def test_enrichment_only_calls_companies_api_for_missing_fields(test_settings):
    """Candidates that already have size_range and hq_country skip enrichment."""
    ca = MagicMock()
    ca.enrich_by_domain = AsyncMock(return_value=None)

    hunter = _make_hunter(test_settings)
    hunter.companies_api_client = ca

    already_enriched = _make_candidate(
        "full.com", "Full", size_range="11-50", hq_country="US"
    )
    needs_enrichment = _make_candidate(
        "partial.com", "Partial", size_range=None, hq_country=None
    )

    await hunter._enrich_with_firmographics([already_enriched, needs_enrichment], cap=5)

    # Only the partial one should trigger a real API call
    assert ca.enrich_by_domain.call_count == 1
    called_domain = ca.enrich_by_domain.call_args[0][0]
    assert "partial" in called_domain


@pytest.mark.asyncio
async def test_enrichment_respects_top_n_cap(test_settings):
    """Only enrichment_top_n calls are made even if more candidates lack fields."""
    ca = MagicMock()
    ca.enrich_by_domain = AsyncMock(return_value=None)

    hunter = _make_hunter(test_settings)
    hunter.companies_api_client = ca

    # 5 candidates all missing firmographics, cap = 2
    candidates = [
        _make_candidate(f"co{i}.com", f"Co{i}", size_range=None)
        for i in range(5)
    ]

    await hunter._enrich_with_firmographics(candidates, cap=2)

    assert ca.enrich_by_domain.call_count == 2


@pytest.mark.asyncio
async def test_run_record_written_to_db(test_settings):
    """create_run and update_run are both called during a hunt."""
    store = _mock_lead_store()
    hunter = _make_hunter(test_settings, lead_store=store)

    await hunter.hunt("tutrain", target_count=5, enrichment_top_n=0)

    store.create_run.assert_called_once_with("tutrain")
    store.update_run.assert_called_once()
    call_kwargs = store.update_run.call_args[1]
    assert "completed_at" in call_kwargs
    assert call_kwargs["status"] == "completed"


@pytest.mark.asyncio
async def test_normalized_domain_handles_edge_cases(test_settings):
    """normalize_domain correctly strips www, protocol, and paths."""
    cases = [
        ("http://www.foo.com/path", "foo.com"),
        ("FOO.COM", "foo.com"),
        ("www.foo.co.uk", "foo.co.uk"),
        ("https://blog.example.org/article", "example.org"),
    ]
    for raw, expected in cases:
        result = normalize_domain(raw)
        assert result == expected, f"normalize_domain({raw!r}) = {result!r}, expected {expected!r}"


@pytest.mark.asyncio
async def test_serpapi_query_construction(test_settings):
    """SerpAPI search query contains the ICP primary keyword and a date filter."""
    hunter = _make_hunter(test_settings)
    icp = IcpStrategist(test_settings).load_strategy("tutrain")

    captured_query: list[str] = []

    async def mock_search(query, **kwargs):
        captured_query.append(query)
        return []

    hunter.serpapi_client.search = mock_search

    with patch.object(
        hunter._gemini, "batch_generate_json", new=AsyncMock(return_value=[])
    ):
        await hunter._hunt_via_serpapi(icp)

    assert captured_query, "search() was never called"
    q = captured_query[0]
    primary_kw = icp.target_industries.industry_keywords[0]
    assert primary_kw in q
    # Date filter present (after:YYYY-MM-DD)
    assert re.search(r"after:\d{4}-\d{2}-\d{2}", q), f"No date filter in: {q}"


@pytest.mark.asyncio
async def test_newsdata_keywords_built_from_icp(test_settings):
    """search_funding_news receives keywords including the first 3 ICP industry_keywords."""
    hunter = _make_hunter(test_settings)
    icp = IcpStrategist(test_settings).load_strategy("tutrain")

    captured_kwargs: list[dict] = []

    async def mock_search_funding_news(**kwargs):
        captured_kwargs.append(kwargs)
        return []

    hunter.newsdata_client.search_funding_news = mock_search_funding_news

    await hunter._hunt_via_newsdata(icp)

    assert captured_kwargs, "search_funding_news() was never called"
    kws = captured_kwargs[0].get("keywords", [])
    icp_kws = icp.target_industries.industry_keywords[:3]
    for kw in icp_kws:
        assert kw in kws, f"ICP keyword {kw!r} missing from newsdata keywords: {kws}"
