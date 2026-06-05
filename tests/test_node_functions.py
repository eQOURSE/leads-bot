"""Phase 9 — Unit tests for individual node functions.

Each node is tested in isolation by:
  1. Providing a minimal PipelineState and mock AgentRegistry
  2. Verifying the correct agent method was called with expected args
  3. Verifying exceptions are caught and stored in node_errors (not re-raised)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.state import make_initial_state


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_state(segment: str = "tutrain") -> dict:
    s = dict(make_initial_state(segment, f"run_{segment}_test", target_count=5))
    return s


def _make_hunt_result(segment: str = "tutrain"):
    from agents._models import HuntResult
    from sources.models import CompanyCandidate
    from datetime import date

    return HuntResult(
        segment=segment, run_id=f"run_{segment}",
        candidates=[
            CompanyCandidate(
                domain="testco.com", name="TestCo", funding_stage="Seed",
                funding_date=date.today(), raw_source="rss", confidence=0.8,
            )
        ],
        source_counts={"rss": 1}, merged_count=1, after_filter=1, after_dedupe=1,
        enriched_count=0, api_credits_used={}, errors=[],
        started_at=datetime.utcnow(), completed_at=datetime.utcnow(), duration_seconds=1.0,
    )


def _make_qualified_result(segment: str = "tutrain"):
    from agents._models import QualifiedResult, QualifiedCandidate, QualifierSubScores
    from sources.models import CompanyCandidate
    from datetime import date

    c = CompanyCandidate(domain="testco.com", name="TestCo", raw_source="rss", confidence=0.8, funding_date=date.today())
    qc = QualifiedCandidate(
        candidate=c, total_score=85, pre_score=60,
        sub_scores=QualifierSubScores(
            funding_recency_score=40, reachability_score=10, geography_score=10,
            size_match_score=0, segment_fit_score=15, buying_signal_score=10,
        ),
        reasoning="ok", disqualifiers=[], tier="tier_1", domain_was_resolved=False,
    )
    return QualifiedResult(
        segment=segment, run_id=f"run_{segment}", qualified=[qc], dropped=[],
        stats={}, api_credits_used={},
        started_at=datetime.utcnow(), completed_at=datetime.utcnow(), duration_seconds=1.0,
    )


def _make_enhanced_result(segment: str = "tutrain"):
    from agents._models import (
        EnhancedQualifiedResult, QualifiedCandidateWithPeople, DecisionMaker
    )
    from tests.test_orchestrator import _make_qualified_result as _qr

    qr = _make_qualified_result(segment)
    dms = [DecisionMaker(full_name="Jane Doe", title="CEO", source="scrapegraph", seniority_score=90)]
    cwp = [QualifiedCandidateWithPeople(qualified=qc, decision_makers=dms, lookup_status="found", lookup_attempts={}) for qc in qr.qualified]
    return EnhancedQualifiedResult(
        segment=segment, run_id=qr.run_id,
        candidates_with_people=cwp, needs_manual_lookup=[],
        stats={}, api_credits_used={},
        started_at=datetime.utcnow(), completed_at=datetime.utcnow(), duration_seconds=1.0,
    )


def _make_enriched_result(segment: str = "tutrain"):
    from agents._models import EnrichedResult, EnrichedCandidate, EnrichedDecisionMaker, EmailResult
    from tests.test_orchestrator import _make_enhanced_result as _er

    enh = _make_enhanced_result(segment)
    enriched = [
        EnrichedCandidate(
            candidate_with_people=cwp,
            enriched_dms=[
                EnrichedDecisionMaker(
                    decision_maker=dm,
                    email_result=EmailResult(email="jane@testco.com", confidence=0.9, source="hunter_finder"),
                )
                for dm in cwp.decision_makers
            ],
            enrichment_status="full",
        )
        for cwp in enh.candidates_with_people
    ]
    return EnrichedResult(
        segment=segment, run_id=enh.run_id,
        enriched_candidates=enriched,
        stats={"emails_found": 1}, api_credits_used={},
        started_at=datetime.utcnow(), completed_at=datetime.utcnow(), duration_seconds=1.0,
    )


def _make_messaged_result(segment: str = "tutrain"):
    from agents._models import (
        MessagedResult, MessagedCandidate, MessagedDecisionMaker, GeneratedMessages
    )
    er = _make_enriched_result(segment)
    msgs = GeneratedMessages(
        email_subject_a="Test A", email_subject_b="Test B",
        email_body="Body " * 20, linkedin_dm="LinkedIn msg",
        reply_likelihood=7, quality_flags=[],
    )
    mc_list = [
        MessagedCandidate(
            enriched_candidate=ec,
            personalization=None,
            messaged_dms=[
                MessagedDecisionMaker(enriched_dm=edm, messages=msgs)
                for edm in ec.enriched_dms
            ],
        )
        for ec in er.enriched_candidates
    ]
    return MessagedResult(
        segment=segment, run_id=er.run_id,
        messaged_candidates=mc_list,
        stats={"messages_generated": 1, "total_dms": 1},
        api_credits_used={},
        started_at=datetime.utcnow(), completed_at=datetime.utcnow(), duration_seconds=1.0,
    )


# ---------------------------------------------------------------------------
# node_load_icp
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_load_icp_calls_strategist():
    from orchestrator.nodes import node_load_icp

    agents = MagicMock()
    agents.icp_strategist.load_strategy.return_value = MagicMock(segment_name="TUTRAIN")

    state = _base_state()
    result = await node_load_icp(state, agents)

    agents.icp_strategist.load_strategy.assert_called_once_with("tutrain")
    assert result["icp_strategy"] is not None
    assert "load_icp" in result["nodes_completed"]


@pytest.mark.asyncio
async def test_node_load_icp_captures_exception():
    from orchestrator.nodes import node_load_icp

    agents = MagicMock()
    agents.icp_strategist.load_strategy.side_effect = ValueError("Unknown segment")

    state = _base_state()
    result = await node_load_icp(state, agents)

    assert "load_icp" in result.get("node_errors", {})
    assert "load_icp" in result["nodes_completed"]


# ---------------------------------------------------------------------------
# node_hunt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_hunt_calls_company_hunter():
    from orchestrator.nodes import node_hunt

    hunt_result = _make_hunt_result()
    agents = MagicMock()
    agents.company_hunter.hunt = AsyncMock(return_value=hunt_result)

    state = _base_state()
    result = await node_hunt(state, agents)

    agents.company_hunter.hunt.assert_awaited_once_with("tutrain", target_count=5)
    assert result["hunt_result"] is hunt_result
    assert "hunt" in result["nodes_completed"]


@pytest.mark.asyncio
async def test_node_hunt_captures_exception():
    from orchestrator.nodes import node_hunt

    agents = MagicMock()
    agents.company_hunter.hunt = AsyncMock(side_effect=RuntimeError("RSS timeout"))

    state = _base_state()
    result = await node_hunt(state, agents)

    assert "hunt" in result.get("node_errors", {})
    assert "RSS timeout" in result["node_errors"]["hunt"]
    assert "hunt" in result["nodes_completed"]


# ---------------------------------------------------------------------------
# node_qualify
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_qualify_calls_qualifier_with_hunt_result():
    from orchestrator.nodes import node_qualify

    hunt_result = _make_hunt_result()
    qualified_result = _make_qualified_result()

    agents = MagicMock()
    agents.qualifier.qualify = AsyncMock(return_value=qualified_result)

    state = _base_state()
    state["hunt_result"] = hunt_result

    result = await node_qualify(state, agents)

    agents.qualifier.qualify.assert_awaited_once_with(hunt_result)
    assert result["qualified_result"] is qualified_result
    assert "qualify" in result["nodes_completed"]


@pytest.mark.asyncio
async def test_node_qualify_skips_when_no_hunt_result():
    from orchestrator.nodes import node_qualify

    agents = MagicMock()
    state = _base_state()  # no hunt_result

    result = await node_qualify(state, agents)

    agents.qualifier.qualify.assert_not_called()
    assert "qualify" in result.get("nodes_skipped", [])


@pytest.mark.asyncio
async def test_node_qualify_captures_exception():
    from orchestrator.nodes import node_qualify

    hunt_result = _make_hunt_result()
    agents = MagicMock()
    agents.qualifier.qualify = AsyncMock(side_effect=RuntimeError("Gemini rate limit"))

    state = _base_state()
    state["hunt_result"] = hunt_result

    result = await node_qualify(state, agents)

    assert "qualify" in result.get("node_errors", {})
    assert "qualify" in result["nodes_completed"]


# ---------------------------------------------------------------------------
# node_find_dms
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_find_dms_calls_dm_finder():
    from orchestrator.nodes import node_find_dms

    qualified_result = _make_qualified_result()
    enhanced_result = _make_enhanced_result()

    agents = MagicMock()
    agents.dm_finder.find_for_qualified = AsyncMock(return_value=enhanced_result)

    state = _base_state()
    state["qualified_result"] = qualified_result

    result = await node_find_dms(state, agents)

    agents.dm_finder.find_for_qualified.assert_awaited_once_with(qualified_result)
    assert result["enhanced_result"] is enhanced_result
    assert "find_dms" in result["nodes_completed"]


@pytest.mark.asyncio
async def test_node_find_dms_captures_exception():
    from orchestrator.nodes import node_find_dms

    agents = MagicMock()
    agents.dm_finder.find_for_qualified = AsyncMock(side_effect=RuntimeError("Apify down"))

    state = _base_state()
    state["qualified_result"] = _make_qualified_result()

    result = await node_find_dms(state, agents)

    assert "find_dms" in result.get("node_errors", {})
    assert "find_dms" in result["nodes_completed"]


# ---------------------------------------------------------------------------
# node_enrich
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_enrich_calls_contact_enricher():
    from orchestrator.nodes import node_enrich

    enhanced_result = _make_enhanced_result()
    enriched_result = _make_enriched_result()

    agents = MagicMock()
    agents.contact_enricher.enrich = AsyncMock(return_value=enriched_result)

    state = _base_state()
    state["enhanced_result"] = enhanced_result

    result = await node_enrich(state, agents)

    agents.contact_enricher.enrich.assert_awaited_once_with(enhanced_result)
    assert result["enriched_result"] is enriched_result
    assert "enrich" in result["nodes_completed"]


@pytest.mark.asyncio
async def test_node_enrich_captures_exception():
    from orchestrator.nodes import node_enrich

    agents = MagicMock()
    agents.contact_enricher.enrich = AsyncMock(side_effect=RuntimeError("Hunter limit"))

    state = _base_state()
    state["enhanced_result"] = _make_enhanced_result()

    result = await node_enrich(state, agents)

    assert "enrich" in result.get("node_errors", {})
    assert "enrich" in result["nodes_completed"]


# ---------------------------------------------------------------------------
# node_personalize
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_personalize_calls_personalizer():
    from orchestrator.nodes import node_personalize

    enriched_result = _make_enriched_result()
    agents = MagicMock()
    agents.personalizer.build_hooks_for_enriched_result = AsyncMock(return_value={"testco.com": MagicMock()})

    state = _base_state()
    state["enriched_result"] = enriched_result

    result = await node_personalize(state, agents)

    agents.personalizer.build_hooks_for_enriched_result.assert_awaited_once_with(enriched_result)
    assert result["personalization_map"] is not None
    assert "personalize" in result["nodes_completed"]


@pytest.mark.asyncio
async def test_node_personalize_captures_exception():
    from orchestrator.nodes import node_personalize

    agents = MagicMock()
    agents.personalizer.build_hooks_for_enriched_result = AsyncMock(
        side_effect=RuntimeError("ScrapeGraph quota exceeded")
    )

    state = _base_state()
    state["enriched_result"] = _make_enriched_result()

    result = await node_personalize(state, agents)

    assert "personalize" in result.get("node_errors", {})
    assert "personalize" in result["nodes_completed"]


# ---------------------------------------------------------------------------
# node_write_messages
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_write_messages_calls_message_writer():
    from orchestrator.nodes import node_write_messages

    enriched_result = _make_enriched_result()
    messaged_result = _make_messaged_result()

    agents = MagicMock()
    agents.message_writer.write_for_enriched = AsyncMock(return_value=messaged_result)

    state = _base_state()
    state["enriched_result"] = enriched_result
    state["personalization_map"] = {}

    result = await node_write_messages(state, agents)

    agents.message_writer.write_for_enriched.assert_awaited_once_with(enriched_result, {})
    assert result["messaged_result"] is messaged_result
    assert "write_messages" in result["nodes_completed"]


@pytest.mark.asyncio
async def test_node_write_messages_captures_exception():
    from orchestrator.nodes import node_write_messages

    agents = MagicMock()
    agents.message_writer.write_for_enriched = AsyncMock(side_effect=RuntimeError("Gemini overloaded"))

    state = _base_state()
    state["enriched_result"] = _make_enriched_result()

    result = await node_write_messages(state, agents)

    assert "write_messages" in result.get("node_errors", {})
    assert "write_messages" in result["nodes_completed"]


# ---------------------------------------------------------------------------
# node_validate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_validate_calls_validator():
    from orchestrator.nodes import node_validate

    messaged_result = _make_messaged_result()
    from tests.test_orchestrator import _make_validated_result
    validated_result = _make_validated_result(messaged_result)

    agents = MagicMock()
    agents.validator.validate = AsyncMock(return_value=validated_result)

    state = _base_state()
    state["messaged_result"] = messaged_result

    result = await node_validate(state, agents)

    agents.validator.validate.assert_awaited_once_with(messaged_result)
    assert result["validated_result"] is validated_result
    assert "validate" in result["nodes_completed"]


@pytest.mark.asyncio
async def test_node_validate_captures_exception():
    from orchestrator.nodes import node_validate

    agents = MagicMock()
    agents.validator.validate = AsyncMock(side_effect=RuntimeError("DB locked"))

    state = _base_state()
    state["messaged_result"] = _make_messaged_result()

    result = await node_validate(state, agents)

    assert "validate" in result.get("node_errors", {})
    assert "validate" in result["nodes_completed"]


# ---------------------------------------------------------------------------
# node_dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_node_dispatch_writes_sqlite_and_sheets_not_telegram():
    from orchestrator.nodes import node_dispatch
    from tests.test_orchestrator import _make_validated_result

    messaged_result = _make_messaged_result()
    validated_result = _make_validated_result(messaged_result)

    agents = MagicMock()
    agents.sqlite_writer.write_validated = AsyncMock(return_value={"inserted": 1, "skipped_existing": 0})
    agents.sheets_sink.write_leads = AsyncMock(return_value={"appended": 1})
    agents.sheets_sink.write_manual_lookup = AsyncMock(return_value=0)
    agents.sheets_sink.write_run_history = AsyncMock()
    agents.telegram_sink.send_run_digest = AsyncMock()

    state = _base_state()
    state["validated_result"] = validated_result
    state["run_id"] = "run_dispatch_test"

    result = await node_dispatch(state, agents)

    agents.sqlite_writer.write_validated.assert_awaited_once()
    agents.sheets_sink.write_leads.assert_awaited_once()
    # Telegram must NOT have been called
    agents.telegram_sink.send_run_digest.assert_not_called()
    assert "dispatch" in result["nodes_completed"]


@pytest.mark.asyncio
async def test_node_dispatch_captures_exception():
    from orchestrator.nodes import node_dispatch

    agents = MagicMock()
    agents.sqlite_writer.write_validated = AsyncMock(side_effect=RuntimeError("DB write failed"))
    agents.sheets_sink.write_leads = AsyncMock(return_value={"appended": 0})
    agents.sheets_sink.write_manual_lookup = AsyncMock(return_value=0)
    agents.sheets_sink.write_run_history = AsyncMock()
    agents.telegram_sink.send_run_digest = AsyncMock()

    state = _base_state()
    state["run_id"] = "run_dispatch_err"

    result = await node_dispatch(state, agents)
    # Dispatch catches internal errors gracefully; nodes_completed should include it
    assert "dispatch" in result["nodes_completed"]
