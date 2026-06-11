"""Phase 13 — tests for the three surgical fixes.

Part A: Sheets Run History columns are populated from funnel metrics
        (not hardcoded zeros) and the status label reflects real numbers.
Part B: Pre-score baseline bonus lets a sparse-firmographic candidate clear 40,
        and the run-level prescore_baseline_bonus_applied metric is recorded.
Part C: DM-finder records a specific reason per source in lookup_attempts when
        no decision-maker is found.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents._models import HuntResult, QualifiedCandidate, QualifierSubScores, QualifiedResult
from agents.icp_strategist import IcpStrategist
from agents.qualifier import Qualifier, _PRE_SCORE_THRESHOLD
from agents.decision_maker_finder import DecisionMakerFinder, _classify_error
from orchestrator.nodes import (
    _build_run_summary,
    _build_empty_run_summary,
    _run_status,
    compute_funnel_metrics,
)
from sources.models import CompanyCandidate


# ===========================================================================
# Helpers
# ===========================================================================

def _days_ago(n: int) -> date:
    return date.today() - timedelta(days=n)


def _make_hunt_result(candidates, segment="eqourse_ai_data", run_id="run-13"):
    now = datetime.now(timezone.utc)
    return HuntResult(
        segment=segment,
        run_id=run_id,
        candidates=candidates,
        source_counts={"rss": 20, "serpapi": 8, "newsdata": 1},
        merged_count=14,
        after_filter=14,
        after_dedupe=13,
        enriched_count=0,
        api_credits_used={},
        errors=[],
        started_at=now,
        completed_at=now,
        duration_seconds=0.5,
    )


def _make_candidate(domain="eqourse.com", **kw) -> CompanyCandidate:
    base = dict(domain=domain, name="Acme AI", raw_source="test", confidence=0.7)
    base.update(kw)
    return CompanyCandidate(**base)


# ===========================================================================
# PART A — Sheets Run History column writes + status labels
# ===========================================================================

def test_run_status_empty_when_nothing_hunted():
    funnel = {"funnel_drop_off": {"hunted_raw": 0, "ready_to_send": 0}}
    assert _run_status(funnel, {}) == "empty_run"


def test_run_status_failed_when_nothing_hunted_but_node_error():
    funnel = {"funnel_drop_off": {"hunted_raw": 0, "ready_to_send": 0}}
    assert _run_status(funnel, {"hunt": "boom"}) == "failed"


def test_run_status_completed_no_leads_when_hunted_but_no_ready():
    funnel = {"funnel_drop_off": {"hunted_raw": 29, "ready_to_send": 0}}
    assert _run_status(funnel, {}) == "completed_no_leads"


def test_run_status_success_when_ready_to_send():
    funnel = {"funnel_drop_off": {"hunted_raw": 29, "ready_to_send": 2}}
    assert _run_status(funnel, {}) == "success"


def test_empty_run_summary_populates_visible_columns_from_funnel():
    """A no-lead run (June 11 scenario) must show the real hunted/qualified
    numbers in the visible columns, not zeros."""
    candidates = [_make_candidate(funding_date=_days_ago(30)) for _ in range(13)]
    hunt = _make_hunt_result(candidates)

    # Fake a qualifier result with 1 candidate clearing Gemini.
    now = datetime.now(timezone.utc)
    qual = QualifiedResult(
        segment="eqourse_ai_data",
        run_id="run-13",
        qualified=[],  # none survive to "qualified" list in this no-lead case
        dropped=[],
        stats={"pre_score_filtered": 8},
        api_credits_used={},
        started_at=now,
        completed_at=now,
        duration_seconds=0.1,
    )

    state = {
        "segment": "eqourse_ai_data",
        "run_id": "run-13",
        "hunt_result": hunt,
        "qualified_result": qual,
    }

    summary = _build_empty_run_summary(state)

    # hunted_raw = 20+8+1 = 29
    assert summary["candidates_hunted"] == 29
    # Visible columns must come from the funnel, and the status must reflect
    # that discovery DID find candidates (not "empty_run").
    assert summary["status"] == "completed_no_leads"
    assert summary["ready_to_send"] == 0
    # The full metrics dict is preserved for the Funnel Metrics column.
    assert "funnel_drop_off" in summary["metrics"]


def test_run_summary_maps_funnel_to_columns():
    candidates = [_make_candidate(funding_date=_days_ago(30)) for _ in range(13)]
    hunt = _make_hunt_result(candidates)
    state = {"segment": "eqourse_ai_data", "run_id": "run-13", "hunt_result": hunt}

    validated = MagicMock()
    validated.duration_seconds = 3.2
    validated.run_id = "run-13"
    validated.segment = "eqourse_ai_data"
    validated.api_credits_used = {"gemini": 2}

    summary = _build_run_summary(validated, {"appended": 0}, state)

    assert summary["candidates_hunted"] == 29
    assert summary["run_id"] == "run-13"
    # api_credits column holds the credits dict, metrics holds funnel JSON.
    assert "funnel_drop_off" in summary["metrics"]


# ===========================================================================
# PART B — Pre-score baseline bonus
# ===========================================================================

def _make_qualifier(test_settings):
    strategist = IcpStrategist(test_settings)
    store = MagicMock()
    store.update_run = AsyncMock()
    gemini = MagicMock()
    ca = MagicMock()
    ca.search_by_filters = AsyncMock(return_value=[])
    return Qualifier(
        settings=test_settings,
        icp_strategist=strategist,
        gemini_agent=gemini,
        companies_api_client=ca,
        lead_store=store,
    )


def test_prescore_baseline_bonus_for_sparse_firmographics(test_settings):
    """Real domain + recent funding + US country but NO size_range should clear
    the 40 gate thanks to the +10 baseline bonus."""
    qualifier = _make_qualifier(test_settings)
    icp = IcpStrategist(test_settings).load_strategy("eqourse_ai_data")

    c = _make_candidate(
        domain="realstartup.com",
        funding_date=_days_ago(30),
        hq_country="United States",
        size_range=None,
        website=None,
        description=None,
    )
    score, sub = qualifier._pre_score(c, icp)

    assert sub["baseline_bonus_applied"] is True
    assert score >= _PRE_SCORE_THRESHOLD  # clears 40


def test_prescore_no_baseline_when_news_domain(test_settings):
    """A news-source domain is not a 'real domain' → no baseline bonus."""
    qualifier = _make_qualifier(test_settings)
    icp = IcpStrategist(test_settings).load_strategy("eqourse_ai_data")

    c = _make_candidate(
        domain="techcrunch.com",
        funding_date=_days_ago(30),
        hq_country="United States",
    )
    _, sub = qualifier._pre_score(c, icp)
    assert sub["baseline_bonus_applied"] is False


def test_prescore_no_baseline_when_unknown_domain(test_settings):
    qualifier = _make_qualifier(test_settings)
    icp = IcpStrategist(test_settings).load_strategy("eqourse_ai_data")

    c = _make_candidate(
        domain="some-company.unknown",
        funding_date=_days_ago(30),
        hq_country="United States",
    )
    _, sub = qualifier._pre_score(c, icp)
    assert sub["baseline_bonus_applied"] is False


@pytest.mark.asyncio
async def test_run_level_baseline_bonus_metric(test_settings):
    """qualify() must expose prescore_baseline_bonus_applied in stats."""
    from agents._models import BatchScoringResponse, GeminiScoringResult

    qualifier = _make_qualifier(test_settings)

    async def _score(prompt, schema, **kw):
        import re
        idx = [int(m) for m in re.findall(r"\[(\d+)\]", prompt)]
        return BatchScoringResponse(results=[
            GeminiScoringResult(
                candidate_index=i, segment_fit_score=5, buying_signal_score=5,
                reasoning="x", disqualifiers=[],
            ) for i in idx
        ])

    qualifier.gemini.generate_json = AsyncMock(side_effect=_score)

    candidates = [
        _make_candidate(domain="realstartup.com", funding_date=_days_ago(20),
                        hq_country="United States"),
        _make_candidate(domain="another.com", name="Another", funding_date=_days_ago(40),
                        hq_country="United States"),
    ]
    hunt = _make_hunt_result(candidates)
    result = await qualifier.qualify(hunt)

    assert "prescore_baseline_bonus_applied" in result.stats
    assert result.stats["prescore_baseline_bonus_applied"] == 2


# ===========================================================================
# PART C — DM-finder lookup_attempts enrichment
# ===========================================================================

def test_classify_error_codes():
    assert _classify_error(Exception("HTTP 403 Forbidden")) == "auth_failed"
    assert _classify_error(Exception("401 Unauthorized")) == "auth_failed"
    assert _classify_error(Exception("got 404 not found")) == "url_not_found"
    assert _classify_error(Exception("429 Too Many Requests")) == "rate_limited"
    assert _classify_error(Exception("connection timed out")) == "timeout"
    assert _classify_error(Exception("weird boom")) == "error"


def _dm_qualified(domain="acme.com", tier="tier_1"):
    candidate = CompanyCandidate(
        domain=domain, name="Acme Corp", raw_source="test", confidence=0.8,
    )
    return QualifiedCandidate(
        candidate=candidate,
        total_score=80,
        pre_score=55,
        sub_scores=QualifierSubScores(
            funding_recency_score=40, reachability_score=10, geography_score=10,
            size_match_score=10, segment_fit_score=10, buying_signal_score=10,
        ),
        reasoning="test",
        disqualifiers=[],
        tier=tier,
        domain_was_resolved=False,
    )


def _dm_qualified_result(qualified):
    now = datetime.now(timezone.utc)
    return QualifiedResult(
        segment="eqourse_ai_data", run_id="run-13", qualified=qualified,
        dropped=[], stats={}, api_credits_used={},
        started_at=now, completed_at=now, duration_seconds=0.1,
    )


@pytest.mark.asyncio
async def test_dm_finder_records_specific_reasons_per_source(test_settings):
    """All three sources fail with distinct errors → lookup_attempts must record
    a specific reason for each, and status is no_decision_maker."""
    icp = MagicMock()
    icp.target_titles = ["CEO", "CTO"]
    icp.target_departments = ["Engineering"]
    icp.target_levels = ["c_suite"]
    strategist = MagicMock()
    strategist.load_strategy.return_value = icp

    sg = MagicMock()
    sg.extract_team_page = AsyncMock(side_effect=Exception("HTTP 403 Forbidden"))
    apify = MagicMock()
    apify.google_search = AsyncMock(side_effect=Exception("request timed out"))
    apify.linkedin_company = AsyncMock(return_value={})
    vibe = MagicMock()
    vibe.find_prospects = AsyncMock(side_effect=Exception("429 too many requests"))
    store = MagicMock()
    store.update_run = AsyncMock()

    finder = DecisionMakerFinder(
        settings=test_settings,
        icp_strategist=strategist,
        scrapegraph_client=sg,
        apify_client=apify,
        vibe_prospecting_client=vibe,
        lead_store=store,
    )

    qualified = _dm_qualified(domain="acme.com", tier="tier_1")
    result = await finder.find_for_qualified(_dm_qualified_result([qualified]))

    cwp = result.candidates_with_people[0]
    assert cwp.lookup_status == "no_decision_maker"
    assert cwp.lookup_attempts.get("scrapegraph") == "auth_failed"
    assert cwp.lookup_attempts.get("apify") == "timeout"
    assert cwp.lookup_attempts.get("explorium") == "rate_limited"


@pytest.mark.asyncio
async def test_dm_finder_fetched_but_empty(test_settings):
    """ScrapeGraph returns successfully but with zero people → fetched_but_empty
    (tier_2 so apify/explorium are not attempted)."""
    icp = MagicMock()
    icp.target_titles = ["CEO"]
    icp.target_departments = []
    icp.target_levels = ["c_suite"]
    strategist = MagicMock()
    strategist.load_strategy.return_value = icp

    sg = MagicMock()
    sg.extract_team_page = AsyncMock(return_value=[])
    apify = MagicMock()
    apify.google_search = AsyncMock(return_value=[])
    apify.linkedin_company = AsyncMock(return_value={})
    vibe = MagicMock()
    vibe.find_prospects = AsyncMock(return_value=[])
    store = MagicMock()
    store.update_run = AsyncMock()

    finder = DecisionMakerFinder(
        settings=test_settings,
        icp_strategist=strategist,
        scrapegraph_client=sg,
        apify_client=apify,
        vibe_prospecting_client=vibe,
        lead_store=store,
    )

    qualified = _dm_qualified(domain="acme.com", tier="tier_2")
    result = await finder.find_for_qualified(_dm_qualified_result([qualified]))

    cwp = result.candidates_with_people[0]
    assert cwp.lookup_attempts.get("scrapegraph") == "fetched_but_empty"
    assert cwp.lookup_attempts.get("apify") == "not_attempted"
