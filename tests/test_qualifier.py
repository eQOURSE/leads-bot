"""Tests for the Qualifier agent. All offline — no real network or Gemini."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents._models import (
    BatchScoringResponse,
    GeminiScoringResult,
    HuntResult,
)
from agents.icp_strategist import IcpStrategist
from agents.qualifier import Qualifier, _GEMINI_BATCH_SIZE, _PRE_SCORE_THRESHOLD
from sources.models import CompanyCandidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today() -> date:
    return date.today()


def _days_ago(n: int) -> date:
    return _today() - timedelta(days=n)


def _make_candidate(
    domain: str = "acme.com",
    name: str = "Acme",
    *,
    funding_date: date | None = None,
    description: str | None = None,
    hq_country: str | None = None,
    size_range: str | None = None,
    funding_stage: str | None = "seed",
    industry: str | None = None,
    website: str | None = None,
) -> CompanyCandidate:
    return CompanyCandidate(
        domain=domain,
        name=name,
        raw_source="test",
        confidence=0.6,
        funding_date=funding_date,
        description=description,
        hq_country=hq_country,
        size_range=size_range,
        funding_stage=funding_stage,
        industry=industry,
        website=website,
    )


def _make_hunt_result(
    candidates: list[CompanyCandidate],
    segment: str = "tutrain",
    run_id: str = "run-test",
) -> HuntResult:
    now = datetime.utcnow()
    return HuntResult(
        segment=segment,
        run_id=run_id,
        candidates=candidates,
        source_counts={"rss": len(candidates), "serpapi": 0, "newsdata": 0},
        merged_count=len(candidates),
        after_filter=len(candidates),
        after_dedupe=len(candidates),
        enriched_count=0,
        api_credits_used={},
        errors=[],
        started_at=now,
        completed_at=now,
        duration_seconds=0.5,
    )


def _mock_lead_store() -> MagicMock:
    store = MagicMock()
    store.update_run = AsyncMock()
    return store


def _make_qualifier(test_settings, *, gemini_results=None, ca_result=None) -> Qualifier:
    strategist = IcpStrategist(test_settings)
    store = _mock_lead_store()

    gemini = MagicMock()
    if gemini_results is not None:
        response = BatchScoringResponse(results=gemini_results)
        gemini.generate_json = AsyncMock(return_value=response)
    else:
        # Default: all candidates get decent scores, as an AsyncMock so
        # assert_not_called() works in tests that need it.
        async def _default_impl(prompt, schema, **kwargs):
            import re as _re
            indices = [int(m) for m in _re.findall(r"\[(\d+)\]", prompt)]
            results = [
                GeminiScoringResult(
                    candidate_index=i,
                    segment_fit_score=8,
                    buying_signal_score=6,
                    reasoning="Default test scoring",
                    disqualifiers=[],
                )
                for i in indices
            ]
            return BatchScoringResponse(results=results)

        gemini.generate_json = AsyncMock(side_effect=_default_impl)

    ca = MagicMock()
    ca.search_by_filters = AsyncMock(return_value=[])
    ca.enrich_by_domain = AsyncMock(return_value=ca_result)

    return Qualifier(
        settings=test_settings,
        icp_strategist=strategist,
        gemini_agent=gemini,
        companies_api_client=ca,
        lead_store=store,
    )


# ---------------------------------------------------------------------------
# Pre-scoring tests
# ---------------------------------------------------------------------------

def test_pre_score_funding_recency_buckets(test_settings):
    qualifier = _make_qualifier(test_settings)
    icp = IcpStrategist(test_settings).load_strategy("tutrain")

    cases = [
        (None, 0),
        (_days_ago(30), 40),   # <= 90 days
        (_days_ago(120), 30),  # <= 180 days
        (_days_ago(200), 20),  # <= 240 days
        (_days_ago(300), 0),   # > 240 days
    ]
    for funding_date, expected_fr in cases:
        c = _make_candidate(funding_date=funding_date)
        score, sub = qualifier._pre_score(c, icp)
        assert sub["funding_recency"] == expected_fr, (
            f"funding_date={funding_date} expected fr={expected_fr}, got {sub['funding_recency']}"
        )


def test_pre_score_negative_signal_auto_drops(test_settings):
    qualifier = _make_qualifier(test_settings)
    icp = IcpStrategist(test_settings).load_strategy("tutrain")

    # "brick-and-mortar tutoring center" is in tutrain negative_signals
    c = _make_candidate(
        description="We operate brick-and-mortar tutoring centers in 50 cities",
        funding_date=_days_ago(30),
    )
    score, sub = qualifier._pre_score(c, icp)
    assert score == 0
    assert sub.get("auto_drop_reason") == "negative_signal_match"


def test_pre_score_geography_match(test_settings):
    qualifier = _make_qualifier(test_settings)
    icp = IcpStrategist(test_settings).load_strategy("tutrain")

    us = _make_candidate(hq_country="United States", funding_date=_days_ago(30))
    unknown = _make_candidate(hq_country=None, funding_date=_days_ago(30))
    foreign = _make_candidate(hq_country="Germany", funding_date=_days_ago(30))

    _, sub_us = qualifier._pre_score(us, icp)
    _, sub_unk = qualifier._pre_score(unknown, icp)
    _, sub_foreign = qualifier._pre_score(foreign, icp)

    assert sub_us["geography"] == 10
    assert sub_unk["geography"] == 5
    assert sub_foreign["geography"] == 0


def test_pre_score_size_match(test_settings):
    qualifier = _make_qualifier(test_settings)
    icp = IcpStrategist(test_settings).load_strategy("tutrain")

    match_c = _make_candidate(size_range="51-200", funding_date=_days_ago(30))
    none_c = _make_candidate(size_range=None, funding_date=_days_ago(30))
    miss_c = _make_candidate(size_range="1000+", funding_date=_days_ago(30))

    _, sub_match = qualifier._pre_score(match_c, icp)
    _, sub_none = qualifier._pre_score(none_c, icp)
    _, sub_miss = qualifier._pre_score(miss_c, icp)

    assert sub_match["size_match"] == 10
    assert sub_none["size_match"] == 5
    assert sub_miss["size_match"] == 0


@pytest.mark.asyncio
async def test_pre_score_skips_below_40(test_settings):
    """Candidates below pre_score threshold never trigger a Gemini call."""
    qualifier = _make_qualifier(test_settings)
    # Candidate with old funding (0) + no domain real (0) + no country (5) + no size (5) = 10
    c = _make_candidate(
        domain="old.unknown",
        funding_date=_days_ago(300),  # too old → fr=0
        hq_country=None,              # geo=5
        size_range=None,              # size=5
    )
    hunt = _make_hunt_result([c])
    result = await qualifier.qualify(hunt)

    qualifier.gemini.generate_json.assert_not_called()
    assert all(d["total_score"] < _PRE_SCORE_THRESHOLD for d in result.dropped)


# ---------------------------------------------------------------------------
# Gemini batch tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_called_in_batches_of_10(test_settings):
    """23 surviving candidates → ceil(23/10) = 3 Gemini calls."""
    # Build 23 candidates that all pass pre-score (recent funding, US, matching size)
    candidates = [
        _make_candidate(
            domain=f"co{i}.com",
            name=f"Company {i}",
            funding_date=_days_ago(30),
            hq_country="United States",
            size_range="51-200",
            description="Online learning platform for K-12 students",
        )
        for i in range(23)
    ]
    qualifier = _make_qualifier(test_settings)
    hunt = _make_hunt_result(candidates)

    await qualifier.qualify(hunt)

    assert qualifier.gemini.generate_json.call_count == 3  # ceil(23/10)


@pytest.mark.asyncio
async def test_gemini_failure_returns_zero_scores(test_settings):
    """If Gemini returns None, qualify() still completes — no crash."""
    qualifier = _make_qualifier(test_settings)
    qualifier.gemini.generate_json = AsyncMock(return_value=None)

    c = _make_candidate(
        domain="alive.com",
        funding_date=_days_ago(30),
        hq_country="United States",
        size_range="51-200",
        description="Online learning platform for K-12 students",
    )
    hunt = _make_hunt_result([c])
    # Must not raise regardless of Gemini failure
    result = await qualifier.qualify(hunt)
    # Result is well-formed
    assert isinstance(result.qualified, list)
    assert isinstance(result.dropped, list)


# ---------------------------------------------------------------------------
# Domain resolution tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_domain_resolution_only_for_unknown_high_scorers(test_settings):
    """Real domain → no resolution. .unknown + pre<60 → no resolution. .unknown + pre>=60 → yes."""
    qualifier = _make_qualifier(test_settings)
    ca = qualifier.companies_api_client

    # Real domain — no resolution needed
    real = _make_candidate("acme.com", "Acme", funding_date=_days_ago(30), hq_country="US", size_range="51-200")
    # .unknown + pre_score will be < 60 (old funding, no country)
    unknown_low = _make_candidate("lowco.unknown", "LowCo", funding_date=_days_ago(300))
    # .unknown + pre_score >= 60 (recent funding, US, matching size, description)
    unknown_high = _make_candidate(
        "highco.unknown", "HighCo",
        funding_date=_days_ago(30),
        hq_country="United States",
        size_range="51-200",
        description="Online EdTech platform for K-12 students",
    )

    hunt = _make_hunt_result([real, unknown_low, unknown_high])
    await qualifier.qualify(hunt, domain_resolution_cap=5)

    # search_by_filters called only for the high-scoring unknown domain
    assert ca.search_by_filters.call_count == 1


@pytest.mark.asyncio
async def test_domain_resolution_cap_enforced(test_settings):
    """Only domain_resolution_cap domains are resolved even if more qualify."""
    qualifier = _make_qualifier(test_settings)
    ca = qualifier.companies_api_client

    # 8 candidates all with .unknown domains and high pre-scores
    candidates = [
        _make_candidate(
            f"co{i}.unknown", f"Company {i}",
            funding_date=_days_ago(30),
            hq_country="United States",
            size_range="51-200",
            description="Online learning platform for K-12 students",
        )
        for i in range(8)
    ]
    hunt = _make_hunt_result(candidates)
    await qualifier.qualify(hunt, domain_resolution_cap=3)

    assert ca.search_by_filters.call_count == 3


@pytest.mark.asyncio
async def test_domain_resolution_failure_drops_candidate(test_settings):
    """When resolution fails (returns None), candidate ends up in dropped."""
    qualifier = _make_qualifier(test_settings)
    qualifier.companies_api_client.search_by_filters = AsyncMock(return_value=[])

    unknown = _make_candidate(
        "ghost.unknown", "Ghost Inc",
        funding_date=_days_ago(30),
        hq_country="United States",
        size_range="51-200",
        description="Online EdTech platform for K-12 students",
    )
    hunt = _make_hunt_result([unknown])
    result = await qualifier.qualify(hunt, domain_resolution_cap=5)

    dropped_names = [d["candidate_name"] for d in result.dropped]
    qualified_names = [q.candidate.name for q in result.qualified]  # type: ignore[union-attr]
    assert "Ghost Inc" in dropped_names
    assert "Ghost Inc" not in qualified_names


# ---------------------------------------------------------------------------
# Tier assignment tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tier_assignment_uses_icp_thresholds(test_settings):
    """score >= tier_1_above → tier_1; tier_2_above <= score < tier_1 → tier_2; below → dropped."""
    # tutrain thresholds: auto_drop=70, tier_2=70, tier_1=85
    # c1: fr=40(<=90d) + reach=10(real dom+website+desc) + geo=10(US) + size=10 = 70
    #     Gemini: seg=15, buy=14 → total=99 → tier_1
    # c2: fr=40 + reach=5(real dom, no website) + geo=10(US) + size=10 = 65
    #     Gemini: seg=8, buy=5 → total=78 → tier_2 (70<=78<85)
    # c3: fr=40 + reach=5 + geo=10 + size=10 = 65
    #     Gemini: seg=0, buy=0 → total=65 → dropped (<70)

    call_counter = {"n": 0}

    async def gemini_by_call_order(prompt, schema, **kwargs):
        import re as _re
        indices = [int(m) for m in _re.findall(r"\[(\d+)\]", prompt)]
        # All 3 candidates scored in one batch; map by their global index
        score_map = {
            1: (15, 14),  # c1 → tier_1
            2: (8, 5),    # c2 → tier_2
            3: (0, 0),    # c3 → dropped
        }
        results = []
        for i in indices:
            sf, bs = score_map.get(i, (0, 0))
            results.append(GeminiScoringResult(
                candidate_index=i, segment_fit_score=sf,
                buying_signal_score=bs, reasoning="test", disqualifiers=[],
            ))
        return BatchScoringResponse(results=results)

    qualifier = _make_qualifier(test_settings)
    qualifier.gemini.generate_json = AsyncMock(side_effect=gemini_by_call_order)

    c1 = _make_candidate("co1.com", "Co1", funding_date=_days_ago(30),
                         hq_country="United States", size_range="51-200",
                         description="EdTech platform K-12", website="https://co1.com")
    c2 = _make_candidate("co2.com", "Co2", funding_date=_days_ago(30),
                         hq_country="United States", size_range="51-200")
    c3 = _make_candidate("co3.com", "Co3", funding_date=_days_ago(30),
                         hq_country="United States", size_range="51-200")

    hunt = _make_hunt_result([c1, c2, c3])
    result = await qualifier.qualify(hunt)

    tiers = {q.candidate.name: q.tier for q in result.qualified}  # type: ignore[union-attr]
    dropped_names = {d["candidate_name"] for d in result.dropped}

    assert tiers.get("Co1") == "tier_1", f"Co1 tiers={tiers}"
    assert tiers.get("Co2") == "tier_2", f"Co2 tiers={tiers}, dropped={dropped_names}"
    assert "Co3" in dropped_names, f"Co3 not in dropped: {dropped_names}"


@pytest.mark.asyncio
async def test_qualified_result_writes_to_run_record(test_settings):
    """update_run is called with qualified_count after qualify()."""
    qualifier = _make_qualifier(test_settings)

    c = _make_candidate(
        "acme.com", "Acme",
        funding_date=_days_ago(30),
        hq_country="United States",
        size_range="51-200",
    )
    hunt = _make_hunt_result([c])
    await qualifier.qualify(hunt)

    qualifier.lead_store.update_run.assert_called()
    call_kwargs = qualifier.lead_store.update_run.call_args[1]
    assert "qualified_count" in call_kwargs


@pytest.mark.asyncio
async def test_run_with_zero_candidates_returns_empty_result(test_settings):
    """Empty HuntResult produces empty QualifiedResult without any Gemini calls."""
    qualifier = _make_qualifier(test_settings)
    hunt = _make_hunt_result([])
    result = await qualifier.qualify(hunt)

    assert result.qualified == []
    qualifier.gemini.generate_json.assert_not_called()


@pytest.mark.asyncio
async def test_disqualifiers_from_gemini_drop_candidate(test_settings):
    """A candidate whose Gemini response includes disqualifiers ends up in dropped."""
    async def gemini_with_disqualifier(prompt, schema, **kwargs):
        import re
        indices = [int(m) for m in re.findall(r"\[(\d+)\]", prompt)]
        results = [
            GeminiScoringResult(
                candidate_index=i,
                segment_fit_score=10,
                buying_signal_score=10,
                reasoning="Looks good but...",
                disqualifiers=["Enterprise corporate L&D only"],
            )
            for i in indices
        ]
        return BatchScoringResponse(results=results)

    qualifier = _make_qualifier(test_settings)
    qualifier.gemini.generate_json = gemini_with_disqualifier

    c = _make_candidate(
        "acme.com", "Acme Corp",
        funding_date=_days_ago(30),
        hq_country="United States",
        size_range="51-200",
    )
    hunt = _make_hunt_result([c])
    result = await qualifier.qualify(hunt)

    qualified_names = [q.candidate.name for q in result.qualified]  # type: ignore[union-attr]
    dropped_names = [d["candidate_name"] for d in result.dropped]
    assert "Acme Corp" not in qualified_names
    assert "Acme Corp" in dropped_names
