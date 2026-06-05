"""Phase 9 — Orchestrator integration tests.

All agents are mocked via AgentRegistry. Tests verify:
  - Happy-path: all nodes complete
  - Skip logic: empty hunt, empty qualify, no DMs, no emails
  - Error resilience: one agent raises, pipeline continues
  - Checkpoint persistence: run, restore, verify state
  - Concurrent execution: asyncio.gather timing
  - Telegram sent once (not 3 times)
  - Per-segment dispatch skips Telegram
  - final_status computed correctly
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_settings(tmp_path):
    """Create an isolated Settings instance for tests."""
    from scripts.init_db import init_db
    db_path = tmp_path / "leads.db"
    init_db(str(db_path))
    from config.settings import Settings
    return Settings(
        _env_file=None,
        GEMINI_API_KEY="test-key",
        HUNTER_API_KEY="test-hunter",
        SCRAPEGRAPH_API_KEY="test-sg",
        SERPAPI_KEY="test-serp",
        NEWSDATA_API_KEY="test-nd",
        COMPANIES_API_TOKEN="test-co",
        APIFY_TOKEN_1="test-apify",
        SQLITE_PATH=str(db_path),
        TELEGRAM_BOT_TOKEN="test-bot",
        TELEGRAM_CHAT_ID="test-chat",
        SHEET_ID="test-sheet-id",
    )


def _make_hunt_result(segment: str = "tutrain", n: int = 2):
    """Build a minimal HuntResult with n candidates."""
    from agents._models import HuntResult
    from sources.models import CompanyCandidate
    from datetime import date

    candidates = [
        CompanyCandidate(
            domain=f"test{i}.com",
            name=f"TestCo {i}",
            description="EdTech platform",
            funding_stage="Seed",
            funding_date=date.today(),
            raw_source="rss",
            confidence=0.8,
        )
        for i in range(n)
    ]
    return HuntResult(
        segment=segment,
        run_id=f"run_{segment}",
        candidates=candidates,
        source_counts={"rss": n},
        merged_count=n,
        after_filter=n,
        after_dedupe=n,
        enriched_count=0,
        api_credits_used={"rss": n},
        errors=[],
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        duration_seconds=1.0,
    )


def _make_qualified_result(hunt_result):
    """Build a minimal QualifiedResult from a HuntResult."""
    from agents._models import (
        QualifiedResult, QualifiedCandidate, QualifierSubScores
    )

    qualified = [
        QualifiedCandidate(
            candidate=c,
            total_score=85,
            pre_score=60,
            sub_scores=QualifierSubScores(
                funding_recency_score=40,
                reachability_score=10,
                geography_score=10,
                size_match_score=0,
                segment_fit_score=15,
                buying_signal_score=10,
            ),
            reasoning="Strong signal",
            disqualifiers=[],
            tier="tier_1",
            domain_was_resolved=False,
        )
        for c in hunt_result.candidates
    ]
    return QualifiedResult(
        segment=hunt_result.segment,
        run_id=hunt_result.run_id,
        qualified=qualified,
        dropped=[],
        stats={"tier_1_count": len(qualified)},
        api_credits_used={"gemini": 1},
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        duration_seconds=1.0,
    )


def _make_enhanced_result(qualified_result):
    """Build a minimal EnhancedQualifiedResult."""
    from agents._models import (
        EnhancedQualifiedResult, QualifiedCandidateWithPeople, DecisionMaker
    )

    dms = [
        DecisionMaker(
            full_name="Jane Doe",
            title="CEO",
            linkedin_url="https://linkedin.com/in/janedoe",
            source="scrapegraph",
            seniority_score=90,
        )
    ]
    cwp = [
        QualifiedCandidateWithPeople(
            qualified=qc,
            decision_makers=dms,
            lookup_status="found",
            lookup_attempts={"scrapegraph": "found_1"},
        )
        for qc in qualified_result.qualified
    ]
    return EnhancedQualifiedResult(
        segment=qualified_result.segment,
        run_id=qualified_result.run_id,
        candidates_with_people=cwp,
        needs_manual_lookup=[],
        stats={"total_dms_found": len(dms)},
        api_credits_used={"scrapegraph": 1},
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        duration_seconds=1.0,
    )


def _make_enriched_result(enhanced_result):
    """Build a minimal EnrichedResult with email."""
    from agents._models import (
        EnrichedResult, EnrichedCandidate, EnrichedDecisionMaker, EmailResult
    )

    enriched_candidates = []
    for cwp in enhanced_result.candidates_with_people:
        enriched_dms = [
            EnrichedDecisionMaker(
                decision_maker=dm,
                email_result=EmailResult(
                    email=f"{dm.full_name.lower().replace(' ', '.')}@test.com",
                    confidence=0.9,
                    source="hunter_finder",
                    smtp_verified=True,
                ),
            )
            for dm in cwp.decision_makers
        ]
        enriched_candidates.append(
            EnrichedCandidate(
                candidate_with_people=cwp,
                enriched_dms=enriched_dms,
                enrichment_status="full",
            )
        )

    return EnrichedResult(
        segment=enhanced_result.segment,
        run_id=enhanced_result.run_id,
        enriched_candidates=enriched_candidates,
        stats={"emails_found": len(enriched_candidates)},
        api_credits_used={"hunter_domain": 1},
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        duration_seconds=1.0,
    )


def _make_messaged_result(enriched_result):
    """Build a minimal MessagedResult."""
    from agents._models import (
        MessagedResult, MessagedCandidate, MessagedDecisionMaker,
        GeneratedMessages
    )

    messages = GeneratedMessages(
        email_subject_a="Quick question about scaling",
        email_subject_b="EdTech tutoring infrastructure",
        email_body="Hi Jane, saw TestCo just raised their Seed round. Congrats on the milestone. We help EdTech platforms like yours deliver live tutoring without hiring ops. Could we show you a 15-min demo? Best, Team | eQOURSE x TUTRAIN",
        linkedin_dm="Hi Jane, congrats on the Seed round! We help EdTech platforms scale live tutoring — worth a quick chat?",
        reply_likelihood=7,
        quality_flags=[],
    )

    messaged_candidates = []
    for ec in enriched_result.enriched_candidates:
        mdms = [
            MessagedDecisionMaker(enriched_dm=edm, messages=messages)
            for edm in ec.enriched_dms
            if edm.email_result.email
        ]
        messaged_candidates.append(
            MessagedCandidate(
                enriched_candidate=ec,
                personalization=None,
                messaged_dms=mdms,
            )
        )

    return MessagedResult(
        segment=enriched_result.segment,
        run_id=enriched_result.run_id,
        messaged_candidates=messaged_candidates,
        stats={"messages_generated": len(messaged_candidates), "total_dms": len(messaged_candidates)},
        api_credits_used={"gemini_writer": len(messaged_candidates)},
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        duration_seconds=1.0,
    )


def _make_validated_result(messaged_result):
    """Build a minimal ValidatedResult."""
    from agents._models import (
        ValidatedResult, ValidatedCandidate, ValidatedDecisionMaker
    )
    import hashlib

    validated_candidates = []
    for mc in messaged_result.messaged_candidates:
        validated_dms = []
        for mdm in mc.messaged_dms:
            edm = mdm.enriched_dm
            dm = edm.decision_maker
            domain = getattr(
                mc.enriched_candidate.candidate_with_people.qualified.candidate,
                "domain", "test.com"
            )
            lead_hash = hashlib.sha256(
                f"{domain}|{dm.full_name.lower()}".encode()
            ).hexdigest()
            validated_dms.append(
                ValidatedDecisionMaker(
                    messaged_dm=mdm,
                    status="ready_to_send",
                    validation_reasons=[],
                    lead_hash=lead_hash,
                )
            )
        validated_candidates.append(
            ValidatedCandidate(messaged_candidate=mc, validated_dms=validated_dms)
        )

    return ValidatedResult(
        segment=messaged_result.segment,
        run_id=messaged_result.run_id,
        validated_candidates=validated_candidates,
        stats={
            "ready_to_send": sum(
                1 for vc in validated_candidates for vdm in vc.validated_dms
                if vdm.status == "ready_to_send"
            ),
            "needs_review": 0,
            "rejected": 0,
        },
        api_credits_used={"gemini_flash": 0},
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        duration_seconds=1.0,
    )


def _make_mock_agents(tmp_path, segment="tutrain"):
    """Return a fully mocked AgentRegistry-like object."""
    hunt_result = _make_hunt_result(segment)
    qualified_result = _make_qualified_result(hunt_result)
    enhanced_result = _make_enhanced_result(qualified_result)
    enriched_result = _make_enriched_result(enhanced_result)
    messaged_result = _make_messaged_result(enriched_result)
    validated_result = _make_validated_result(messaged_result)

    agents = MagicMock()
    agents.icp_strategist.load_strategy.return_value = MagicMock()
    agents.icp_strategist.list_segments.return_value = [
        "tutrain", "eqourse_content", "eqourse_ai_data"
    ]
    agents.company_hunter.hunt = AsyncMock(return_value=hunt_result)
    agents.qualifier.qualify = AsyncMock(return_value=qualified_result)
    agents.dm_finder.find_for_qualified = AsyncMock(return_value=enhanced_result)
    agents.contact_enricher.enrich = AsyncMock(return_value=enriched_result)
    agents.personalizer.build_hooks_for_enriched_result = AsyncMock(return_value={})
    agents.message_writer.write_for_enriched = AsyncMock(return_value=messaged_result)
    agents.validator.validate = AsyncMock(return_value=validated_result)

    # Sinks
    agents.sqlite_writer.write_validated = AsyncMock(
        return_value={"inserted": 1, "skipped_existing": 0}
    )
    agents.sheets_sink.write_leads = AsyncMock(return_value={"appended": 1})
    agents.sheets_sink.write_manual_lookup = AsyncMock(return_value=0)
    agents.sheets_sink.write_run_history = AsyncMock()
    agents.telegram_sink.send_run_digest = AsyncMock(
        return_value={"message_id": 12345, "leads_included": 1}
    )

    return agents, {
        "hunt_result": hunt_result,
        "qualified_result": qualified_result,
        "enhanced_result": enhanced_result,
        "enriched_result": enriched_result,
        "messaged_result": messaged_result,
        "validated_result": validated_result,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_segment_happy_path_completes_all_nodes(tmp_path):
    """All 9 nodes should complete in the happy path."""
    from orchestrator.graph import build_pipeline_graph
    from orchestrator.state import make_initial_state

    agents, data = _make_mock_agents(tmp_path, "tutrain")
    # Use no checkpointer to avoid MagicMock serialization issues in tests
    graph = build_pipeline_graph(agents, checkpointer=None)

    initial = make_initial_state("tutrain", "run-happy-1", target_count=5)
    result = await graph.ainvoke(dict(initial))

    assert result["validated_result"] is not None
    assert "dispatch" in result["nodes_completed"]
    assert result.get("node_errors") == {} or result.get("node_errors") is None or len(result.get("node_errors", {})) == 0


@pytest.mark.asyncio
async def test_run_segment_skips_qualify_when_hunt_returns_empty(tmp_path):
    """When hunt returns 0 candidates, graph should skip to dispatch."""
    from agents._models import HuntResult
    from orchestrator.graph import build_pipeline_graph
    from orchestrator.state import make_initial_state

    agents, _ = _make_mock_agents(tmp_path, "tutrain")
    # Override hunt to return empty
    empty_hunt = HuntResult(
        segment="tutrain", run_id="run-empty",
        candidates=[], source_counts={}, merged_count=0, after_filter=0,
        after_dedupe=0, enriched_count=0, api_credits_used={}, errors=[],
        started_at=datetime.utcnow(), completed_at=datetime.utcnow(), duration_seconds=0.1,
    )
    agents.company_hunter.hunt = AsyncMock(return_value=empty_hunt)

    graph = build_pipeline_graph(agents, checkpointer=None)
    initial = make_initial_state("tutrain", "run-empty-hunt", target_count=5)
    result = await graph.ainvoke(dict(initial))

    # qualify should NOT have been called
    agents.qualifier.qualify.assert_not_called()
    # dispatch should still have run
    assert "dispatch" in result["nodes_completed"]


@pytest.mark.asyncio
async def test_run_segment_skips_enrich_when_no_dms_found(tmp_path):
    """When all DMs come back empty, graph should skip to dispatch from find_dms."""
    from agents._models import EnhancedQualifiedResult, QualifiedCandidateWithPeople
    from orchestrator.graph import build_pipeline_graph
    from orchestrator.state import make_initial_state

    agents, data = _make_mock_agents(tmp_path, "tutrain")

    # Override enhanced_result: all candidates have 0 DMs
    cwp_no_dms = [
        QualifiedCandidateWithPeople(
            qualified=qc,
            decision_makers=[],
            lookup_status="no_decision_maker",
            lookup_attempts={},
        )
        for qc in data["qualified_result"].qualified
    ]
    no_dms_result = EnhancedQualifiedResult(
        segment="tutrain", run_id=data["qualified_result"].run_id,
        candidates_with_people=cwp_no_dms,
        needs_manual_lookup=[],
        stats={"total_dms_found": 0},
        api_credits_used={},
        started_at=datetime.utcnow(), completed_at=datetime.utcnow(), duration_seconds=1.0,
    )
    agents.dm_finder.find_for_qualified = AsyncMock(return_value=no_dms_result)

    graph = build_pipeline_graph(agents, checkpointer=None)
    initial = make_initial_state("tutrain", "run-no-dms", target_count=5)
    result = await graph.ainvoke(dict(initial))

    agents.contact_enricher.enrich.assert_not_called()
    assert "dispatch" in result["nodes_completed"]


@pytest.mark.asyncio
async def test_run_segment_skips_personalize_write_validate_when_no_emails(tmp_path):
    """When enrich returns 0 emails, graph should skip personalize/write/validate."""
    from agents._models import EnrichedResult, EnrichedCandidate, EmailResult, EnrichedDecisionMaker
    from orchestrator.graph import build_pipeline_graph
    from orchestrator.state import make_initial_state

    agents, data = _make_mock_agents(tmp_path, "tutrain")

    # Enriched result with no emails
    no_email_candidates = [
        EnrichedCandidate(
            candidate_with_people=cwp,
            enriched_dms=[
                EnrichedDecisionMaker(
                    decision_maker=dm,
                    email_result=EmailResult(email=None, confidence=0.0, source="not_found"),
                )
                for dm in cwp.decision_makers
            ],
            enrichment_status="no_emails",
        )
        for cwp in data["enhanced_result"].candidates_with_people
    ]
    no_email_enriched = EnrichedResult(
        segment="tutrain", run_id=data["enhanced_result"].run_id,
        enriched_candidates=no_email_candidates,
        stats={"emails_found": 0},
        api_credits_used={},
        started_at=datetime.utcnow(), completed_at=datetime.utcnow(), duration_seconds=1.0,
    )
    agents.contact_enricher.enrich = AsyncMock(return_value=no_email_enriched)

    graph = build_pipeline_graph(agents, checkpointer=None)
    initial = make_initial_state("tutrain", "run-no-emails", target_count=5)
    result = await graph.ainvoke(dict(initial))

    agents.personalizer.build_hooks_for_enriched_result.assert_not_called()
    agents.message_writer.write_for_enriched.assert_not_called()
    agents.validator.validate.assert_not_called()
    assert "dispatch" in result["nodes_completed"]


@pytest.mark.asyncio
async def test_node_error_doesnt_crash_pipeline(tmp_path):
    """A node raising an exception should record the error but not stop the graph."""
    from orchestrator.graph import build_pipeline_graph
    from orchestrator.state import make_initial_state

    agents, _ = _make_mock_agents(tmp_path, "tutrain")
    # Make personalizer raise
    agents.personalizer.build_hooks_for_enriched_result = AsyncMock(
        side_effect=RuntimeError("ScrapeGraph quota exceeded")
    )

    # No checkpointer avoids MagicMock serialization issues
    graph = build_pipeline_graph(agents, checkpointer=None)
    initial = make_initial_state("tutrain", "run-error-resilience", target_count=5)
    result = await graph.ainvoke(dict(initial))

    # Pipeline should still reach dispatch despite personalizer error
    assert "dispatch" in result["nodes_completed"]
    # personalizer error → write_messages still ran with empty personalization_map


@pytest.mark.asyncio
async def test_state_persists_via_checkpointer(tmp_path):
    """Run with a checkpointer, verify state can be read back from the checkpoint."""
    from langgraph.checkpoint.memory import MemorySaver
    from orchestrator.graph import build_pipeline_graph
    from orchestrator.state import make_initial_state
    from agents._models import HuntResult

    agents, _ = _make_mock_agents(tmp_path, "tutrain")
    # Use empty hunt so only HuntResult (Pydantic) goes into state — no MagicMock values
    empty_hunt = HuntResult(
        segment="tutrain", run_id="run-cp",
        candidates=[], source_counts={}, merged_count=0, after_filter=0,
        after_dedupe=0, enriched_count=0, api_credits_used={}, errors=[],
        started_at=datetime.utcnow(), completed_at=datetime.utcnow(), duration_seconds=0.1,
    )
    agents.company_hunter.hunt = AsyncMock(return_value=empty_hunt)
    # Make icp_strategist return None (icp_strategy field = None is serializable)
    agents.icp_strategist.load_strategy.return_value = None

    checkpointer = MemorySaver()
    graph = build_pipeline_graph(agents, checkpointer)

    thread_id = "run-checkpoint-test"
    initial = make_initial_state("tutrain", thread_id, target_count=5)
    await graph.ainvoke(dict(initial), config={"configurable": {"thread_id": thread_id}})

    # Verify checkpoint was stored
    state_snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    assert state_snapshot is not None
    saved_values = state_snapshot.values
    assert saved_values.get("segment") == "tutrain"
    assert "dispatch" in (saved_values.get("nodes_completed") or [])


@pytest.mark.asyncio
async def test_run_all_segments_runs_concurrently(tmp_path):
    """asyncio.gather should run segments in parallel — total time < sum of individual times."""
    import asyncio
    from orchestrator.runner import PipelineRunner

    settings = _make_settings(tmp_path)

    async def slow_segment(segment: str, target_count: int = 30, thread_id=None):
        await asyncio.sleep(0.1)  # simulate 0.1s per segment
        from orchestrator.state import make_initial_state
        state = make_initial_state(segment, f"{segment}_test", target_count)
        state["final_status"] = "success"
        state["nodes_completed"] = ["dispatch"]
        return state

    async with PipelineRunner(settings) as runner:
        # Patch run_segment to be slow
        runner.run_segment = slow_segment
        # Patch icp_strategist.list_segments
        runner.agents.icp_strategist.list_segments = lambda: [
            "tutrain", "eqourse_content", "eqourse_ai_data"
        ]
        # Patch both telegram paths so no real network call is made.
        # These segments produce no validated_result, so the runner takes the
        # empty-digest branch — mock it to keep the timing assertion clean.
        runner.agents.telegram_sink.send_run_digest = AsyncMock(
            return_value={"message_id": 1, "leads_included": 0}
        )
        runner.agents.telegram_sink.send_empty_run_digest = AsyncMock(
            return_value=1
        )

        t0 = time.monotonic()
        results = await runner.run_all_segments(target_count=5)
        elapsed = time.monotonic() - t0

    # 3 segments × 0.1s each = 0.3s serial; concurrent should be well under.
    # Allow headroom for AgentRegistry init / event-loop scheduling on CI.
    assert elapsed < 0.3, f"Expected parallel execution < 0.3s, got {elapsed:.2f}s"
    assert len(results) == 3


@pytest.mark.asyncio
async def test_run_all_segments_failure_in_one_doesnt_block_others(tmp_path):
    """If one segment fails, the others should still complete."""
    import asyncio
    from orchestrator.runner import PipelineRunner

    settings = _make_settings(tmp_path)
    call_count = {"n": 0}

    async def maybe_fail_segment(segment: str, target_count: int = 30, thread_id=None):
        call_count["n"] += 1
        from orchestrator.state import make_initial_state
        state = make_initial_state(segment, f"{segment}_test", target_count)
        if segment == "tutrain":
            state["node_errors"] = {"graph_crash": "Apify quota exceeded"}
            state["final_status"] = "failed"
        else:
            state["final_status"] = "success"
        state["nodes_completed"] = ["dispatch"]
        return state

    async with PipelineRunner(settings) as runner:
        runner.run_segment = maybe_fail_segment
        runner.agents.icp_strategist.list_segments = lambda: [
            "tutrain", "eqourse_content", "eqourse_ai_data"
        ]
        runner.agents.telegram_sink.send_run_digest = AsyncMock(
            return_value={"message_id": 1, "leads_included": 0}
        )
        # No validated results → empty-digest path; mock it to avoid a real call.
        runner.agents.telegram_sink.send_empty_run_digest = AsyncMock(return_value=1)

        results = await runner.run_all_segments(target_count=5)

    assert call_count["n"] == 3  # all 3 were called
    assert results["tutrain"]["final_status"] == "failed"
    assert results["eqourse_content"]["final_status"] == "success"
    assert results["eqourse_ai_data"]["final_status"] == "success"


@pytest.mark.asyncio
async def test_consolidated_telegram_digest_sent_once(tmp_path):
    """Telegram send_run_digest should be called exactly once after all segments complete."""
    from orchestrator.runner import PipelineRunner

    settings = _make_settings(tmp_path)
    telegram_call_count = {"n": 0}

    async def fake_segment(segment: str, target_count: int = 30, thread_id=None):
        from orchestrator.state import make_initial_state
        state = make_initial_state(segment, f"{segment}_test", target_count)
        state["validated_result"] = _make_validated_result(
            _make_messaged_result(
                _make_enriched_result(
                    _make_enhanced_result(
                        _make_qualified_result(_make_hunt_result(segment))
                    )
                )
            )
        )
        state["final_status"] = "success"
        return state

    async def count_telegram(validated_by_segment, sheets_url=""):
        telegram_call_count["n"] += 1
        return {"message_id": 99, "leads_included": 1}

    async with PipelineRunner(settings) as runner:
        runner.run_segment = fake_segment
        runner.agents.icp_strategist.list_segments = lambda: [
            "tutrain", "eqourse_content", "eqourse_ai_data"
        ]
        runner.agents.telegram_sink.send_run_digest = count_telegram
        await runner.run_all_segments(target_count=5)

    assert telegram_call_count["n"] == 1, (
        f"Expected Telegram called once, got {telegram_call_count['n']}"
    )


@pytest.mark.asyncio
async def test_per_segment_dispatch_skips_telegram(tmp_path):
    """The dispatch node should never call telegram_sink.send_run_digest."""
    from agents._models import HuntResult
    from orchestrator.graph import build_pipeline_graph
    from orchestrator.state import make_initial_state

    agents, _ = _make_mock_agents(tmp_path, "tutrain")
    # Use empty hunt — dispatch runs but has no validated_result to write
    empty_hunt = HuntResult(
        segment="tutrain", run_id="run-no-tg",
        candidates=[], source_counts={}, merged_count=0, after_filter=0,
        after_dedupe=0, enriched_count=0, api_credits_used={}, errors=[],
        started_at=datetime.utcnow(), completed_at=datetime.utcnow(), duration_seconds=0.1,
    )
    agents.company_hunter.hunt = AsyncMock(return_value=empty_hunt)

    graph = build_pipeline_graph(agents, checkpointer=None)
    initial = make_initial_state("tutrain", "run-no-tg", target_count=5)
    await graph.ainvoke(dict(initial))

    # Telegram should NOT have been called from inside the graph
    agents.telegram_sink.send_run_digest.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_final_status_computed_correctly(tmp_path):
    """final_status should be success / partial_success / failed based on errors and results."""
    from orchestrator.runner import _compute_final_status
    from orchestrator.state import make_initial_state

    # Success: no errors
    state = make_initial_state("tutrain", "r1")
    state["node_errors"] = {}
    assert _compute_final_status(state) == "success"

    # Failed: errors, no validated_result
    state2 = make_initial_state("tutrain", "r2")
    state2["node_errors"] = {"hunt": "timeout"}
    assert _compute_final_status(state2) == "failed"

    # Partial success: errors, but validated_result has ready_to_send leads
    state3 = make_initial_state("tutrain", "r3")
    state3["node_errors"] = {"personalize": "ScrapeGraph down"}
    vr_mock = MagicMock()
    vr_mock.stats = {"ready_to_send": 2, "needs_review": 0, "rejected": 0}
    state3["validated_result"] = vr_mock
    assert _compute_final_status(state3) == "partial_success"
