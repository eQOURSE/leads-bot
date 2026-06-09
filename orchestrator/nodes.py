"""Phase 9 — Graph node functions.

Each node is an async function that:
  1. Reads relevant fields from PipelineState
  2. Calls the appropriate agent method
  3. Returns a dict of fields to merge back into state

Error handling: any exception is caught, logged to node_errors, and the node
is still marked completed so the graph continues rather than crashing.

The dispatch node accepts a ``skip_telegram`` keyword (passed via the agents
registry flag) so that per-segment dispatch does NOT send Telegram — the
consolidated digest is sent once in run_all_segments instead.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Optional

from config.logging_config import setup_logging

if TYPE_CHECKING:
    from orchestrator.agents_registry import AgentRegistry
    from orchestrator.state import PipelineState

_log = setup_logging("orchestrator.nodes")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _append(existing: list, item: str) -> list:
    """Return a new list with item appended (state is immutable — always copy)."""
    return list(existing) + [item]


# ---------------------------------------------------------------------------
# Node 1 — load_icp
# ---------------------------------------------------------------------------

async def node_load_icp(state: "PipelineState", agents: "AgentRegistry") -> dict:
    name = "load_icp"
    t0 = time.monotonic()
    segment = state["segment"]
    try:
        icp = agents.icp_strategist.load_strategy(segment)
        _log.info("Node[%s][%s] completed in %.1fs", segment, name, time.monotonic() - t0)
        return {
            "icp_strategy": icp,
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    except Exception as exc:
        _log.error("Node[%s][%s] failed: %s", segment, name, exc, exc_info=True)
        return {
            "node_errors": {**state.get("node_errors", {}), name: str(exc)},
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }


# ---------------------------------------------------------------------------
# Node 2 — hunt
# ---------------------------------------------------------------------------

async def node_hunt(state: "PipelineState", agents: "AgentRegistry") -> dict:
    name = "hunt"
    t0 = time.monotonic()
    segment = state["segment"]
    target_count = state.get("target_count", 30)
    try:
        hunt_result = await agents.company_hunter.hunt(
            segment,
            target_count=target_count,
        )
        _log.info(
            "Node[%s][%s] found %d candidates in %.1fs",
            segment, name, len(hunt_result.candidates), time.monotonic() - t0,
        )
        # Phase 11 — capture hunter + gemini resilience metrics into state.
        # Coerce to plain serializable primitives (checkpointer uses msgpack).
        raw_hm = getattr(agents.company_hunter, "last_metrics", {}) or {}
        hunter_metrics = raw_hm if isinstance(raw_hm, dict) else {}
        gemini_metrics: dict = {}
        try:
            g = agents.company_hunter._gemini
            gemini_metrics = {
                "retry_count": int(getattr(g, "retry_count", 0) or 0),
                "fallback_count": int(getattr(g, "fallback_count", 0) or 0),
                "backoff_seconds": float(getattr(g, "backoff_seconds", 0.0) or 0.0),
            }
        except Exception:  # noqa: BLE001
            gemini_metrics = {}
        return {
            "hunt_result": hunt_result,
            "_hunter_metrics": hunter_metrics,
            "_gemini_metrics": gemini_metrics,
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    except Exception as exc:
        _log.error("Node[%s][%s] failed: %s", segment, name, exc, exc_info=True)
        return {
            "node_errors": {**state.get("node_errors", {}), name: str(exc)},
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }


# ---------------------------------------------------------------------------
# Node 3 — qualify
# ---------------------------------------------------------------------------

async def node_qualify(state: "PipelineState", agents: "AgentRegistry") -> dict:
    name = "qualify"
    t0 = time.monotonic()
    segment = state["segment"]
    hunt_result = state.get("hunt_result")
    if hunt_result is None:
        _log.warning("Node[%s][%s] skipped — no hunt_result", segment, name)
        return {
            "nodes_skipped": _append(state.get("nodes_skipped", []), name),
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    try:
        qualified_result = await agents.qualifier.qualify(hunt_result)
        _log.info(
            "Node[%s][%s] qualified %d in %.1fs",
            segment, name, len(qualified_result.qualified), time.monotonic() - t0,
        )
        return {
            "qualified_result": qualified_result,
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    except Exception as exc:
        _log.error("Node[%s][%s] failed: %s", segment, name, exc, exc_info=True)
        return {
            "node_errors": {**state.get("node_errors", {}), name: str(exc)},
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }


# ---------------------------------------------------------------------------
# Node 4 — find_dms
# ---------------------------------------------------------------------------

async def node_find_dms(state: "PipelineState", agents: "AgentRegistry") -> dict:
    name = "find_dms"
    t0 = time.monotonic()
    segment = state["segment"]
    qualified_result = state.get("qualified_result")
    if qualified_result is None:
        _log.warning("Node[%s][%s] skipped — no qualified_result", segment, name)
        return {
            "nodes_skipped": _append(state.get("nodes_skipped", []), name),
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    try:
        enhanced_result = await agents.dm_finder.find_for_qualified(qualified_result)
        total_dms = sum(
            len(c.decision_makers)
            for c in enhanced_result.candidates_with_people
        )
        _log.info(
            "Node[%s][%s] found %d DMs (manual=%d) in %.1fs",
            segment, name, total_dms,
            len(enhanced_result.needs_manual_lookup),
            time.monotonic() - t0,
        )
        return {
            "enhanced_result": enhanced_result,
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    except Exception as exc:
        _log.error("Node[%s][%s] failed: %s", segment, name, exc, exc_info=True)
        return {
            "node_errors": {**state.get("node_errors", {}), name: str(exc)},
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }


# ---------------------------------------------------------------------------
# Node 5 — enrich
# ---------------------------------------------------------------------------

async def node_enrich(state: "PipelineState", agents: "AgentRegistry") -> dict:
    name = "enrich"
    t0 = time.monotonic()
    segment = state["segment"]
    enhanced_result = state.get("enhanced_result")
    if enhanced_result is None:
        _log.warning("Node[%s][%s] skipped — no enhanced_result", segment, name)
        return {
            "nodes_skipped": _append(state.get("nodes_skipped", []), name),
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    try:
        enriched_result = await agents.contact_enricher.enrich(enhanced_result)
        _log.info(
            "Node[%s][%s] enriched %d candidates, %d emails found in %.1fs",
            segment, name,
            len(enriched_result.enriched_candidates),
            enriched_result.stats.get("emails_found", 0),
            time.monotonic() - t0,
        )
        return {
            "enriched_result": enriched_result,
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    except Exception as exc:
        _log.error("Node[%s][%s] failed: %s", segment, name, exc, exc_info=True)
        return {
            "node_errors": {**state.get("node_errors", {}), name: str(exc)},
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }


# ---------------------------------------------------------------------------
# Node 6 — personalize
# ---------------------------------------------------------------------------

async def node_personalize(state: "PipelineState", agents: "AgentRegistry") -> dict:
    name = "personalize"
    t0 = time.monotonic()
    segment = state["segment"]
    enriched_result = state.get("enriched_result")
    if enriched_result is None:
        _log.warning("Node[%s][%s] skipped — no enriched_result", segment, name)
        return {
            "nodes_skipped": _append(state.get("nodes_skipped", []), name),
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    try:
        personalization_map = await agents.personalizer.build_hooks_for_enriched_result(
            enriched_result
        )
        _log.info(
            "Node[%s][%s] built hooks for %d domains in %.1fs",
            segment, name, len(personalization_map), time.monotonic() - t0,
        )
        return {
            "personalization_map": personalization_map,
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    except Exception as exc:
        _log.error("Node[%s][%s] failed: %s", segment, name, exc, exc_info=True)
        return {
            "node_errors": {**state.get("node_errors", {}), name: str(exc)},
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }


# ---------------------------------------------------------------------------
# Node 7 — write_messages
# ---------------------------------------------------------------------------

async def node_write_messages(state: "PipelineState", agents: "AgentRegistry") -> dict:
    name = "write_messages"
    t0 = time.monotonic()
    segment = state["segment"]
    enriched_result = state.get("enriched_result")
    personalization_map = state.get("personalization_map") or {}
    if enriched_result is None:
        _log.warning("Node[%s][%s] skipped — no enriched_result", segment, name)
        return {
            "nodes_skipped": _append(state.get("nodes_skipped", []), name),
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    try:
        messaged_result = await agents.message_writer.write_for_enriched(
            enriched_result,
            personalization_map,
        )
        generated = messaged_result.stats.get("messages_generated", 0)
        _log.info(
            "Node[%s][%s] generated %d messages in %.1fs",
            segment, name, generated, time.monotonic() - t0,
        )
        return {
            "messaged_result": messaged_result,
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    except Exception as exc:
        _log.error("Node[%s][%s] failed: %s", segment, name, exc, exc_info=True)
        return {
            "node_errors": {**state.get("node_errors", {}), name: str(exc)},
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }


# ---------------------------------------------------------------------------
# Node 8 — validate
# ---------------------------------------------------------------------------

async def node_validate(state: "PipelineState", agents: "AgentRegistry") -> dict:
    name = "validate"
    t0 = time.monotonic()
    segment = state["segment"]
    messaged_result = state.get("messaged_result")
    if messaged_result is None:
        _log.warning("Node[%s][%s] skipped — no messaged_result", segment, name)
        return {
            "nodes_skipped": _append(state.get("nodes_skipped", []), name),
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    try:
        validated_result = await agents.validator.validate(messaged_result)
        ready = validated_result.stats.get("ready_to_send", 0)
        review = validated_result.stats.get("needs_review", 0)
        rejected = validated_result.stats.get("rejected", 0)
        _log.info(
            "Node[%s][%s] ready=%d review=%d rejected=%d in %.1fs",
            segment, name, ready, review, rejected, time.monotonic() - t0,
        )
        return {
            "validated_result": validated_result,
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    except Exception as exc:
        _log.error("Node[%s][%s] failed: %s", segment, name, exc, exc_info=True)
        return {
            "node_errors": {**state.get("node_errors", {}), name: str(exc)},
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }


# ---------------------------------------------------------------------------
# Node 9 — dispatch
#
# Per-segment dispatch: SQLite + Sheets ONLY. NO Telegram.
# Telegram is sent once consolidated after all segments complete.
# ---------------------------------------------------------------------------

async def node_dispatch(state: "PipelineState", agents: "AgentRegistry") -> dict:
    name = "dispatch"
    t0 = time.monotonic()
    segment = state["segment"]
    validated_result = state.get("validated_result")
    enhanced_result = state.get("enhanced_result")
    run_id = state.get("run_id", "unknown")

    needs_manual = []
    if enhanced_result is not None:
        needs_manual = getattr(enhanced_result, "needs_manual_lookup", [])

    try:
        # Write SQLite
        try:
            if validated_result is not None:
                sqlite_counts = await agents.sqlite_writer.write_validated(validated_result)
                _log.info(
                    "Node[%s][%s] sqlite: inserted=%d skipped=%d",
                    segment, name,
                    sqlite_counts.get("inserted", 0),
                    sqlite_counts.get("skipped_existing", 0),
                )
        except Exception as exc:
            _log.error("Node[%s][%s] sqlite write failed: %s", segment, name, exc)

        # Write Sheets (fault-tolerant)
        try:
            if validated_result is not None:
                sheets_result = await agents.sheets_sink.write_leads(validated_result)
                _log.info(
                    "Node[%s][%s] sheets: appended=%d",
                    segment, name, sheets_result.get("appended", 0),
                )
                if needs_manual:
                    await agents.sheets_sink.write_manual_lookup(needs_manual, run_id, segment)

                # Run history
                run_summary = _build_run_summary(validated_result, sheets_result, state)
                await agents.sheets_sink.write_run_history(run_summary)
        except Exception as exc:
            _log.error("Node[%s][%s] sheets write failed: %s", segment, name, exc)

        # Empty-run audit: if no validated_result, still write a run history row
        if validated_result is None:
            try:
                empty_summary = _build_empty_run_summary(state)
                await agents.sheets_sink.write_run_history(empty_summary)
                _log.info("Node[%s][%s] wrote empty run history row", segment, name)
            except Exception as exc:
                _log.warning("Node[%s][%s] empty run history failed: %s", segment, name, exc)

        _log.info(
            "Node[%s][%s] completed in %.1fs (Telegram skipped — sent consolidated)",
            segment, name, time.monotonic() - t0,
        )

        # Build a minimal SentResult to store in state
        from agents._models import SentResult
        sent_result = SentResult(
            segment=segment,
            run_id=run_id,
            sqlite_inserted=0,
            sqlite_skipped=0,
            sheets_appended=0,
            sheets_errors=[],
            telegram_message_id=None,
            telegram_error=None,
            duration_seconds=time.monotonic() - t0,
        )

        return {
            "sent_result": sent_result,
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }
    except Exception as exc:
        _log.error("Node[%s][%s] failed: %s", segment, name, exc, exc_info=True)
        return {
            "node_errors": {**state.get("node_errors", {}), name: str(exc)},
            "nodes_completed": _append(state.get("nodes_completed", []), name),
        }


# ---------------------------------------------------------------------------
# Helper: build run summary dict for Sheets
# ---------------------------------------------------------------------------

def compute_funnel_metrics(state: "PipelineState") -> dict:
    """Phase 11 — build the funnel_drop_off + source_contributions metrics dict.

    Computed entirely from the per-segment state result objects (isolated per
    segment, unlike the shared hunter instance), plus the hunter's last_metrics
    for source contributions and resolution rate.
    """
    hunt = state.get("hunt_result")
    qual = state.get("qualified_result")
    enh = state.get("enhanced_result")
    enr = state.get("enriched_result")
    msg = state.get("messaged_result")
    val = state.get("validated_result")

    hunted_raw = 0
    source_contributions: dict = {}
    resolution_rate = 0.0
    after_domain = 0
    if hunt is not None:
        source_contributions = dict(getattr(hunt, "source_counts", {}) or {})
        hunted_raw = sum(source_contributions.values()) or len(getattr(hunt, "candidates", []))
        after_domain = sum(
            1 for c in getattr(hunt, "candidates", [])
            if getattr(c, "domain", "") and not str(getattr(c, "domain", "")).endswith(".unknown")
        )

    after_dedupe = getattr(hunt, "after_dedupe", 0) if hunt else 0

    # Pre-score / gemini drop-offs from qualifier stats.
    after_prescore = 0
    after_gemini = 0
    if qual is not None:
        stats = getattr(qual, "stats", {}) or {}
        qualified_n = len(getattr(qual, "qualified", []))
        # pre_score_filtered = dropped before gemini; survivors = candidates - dropped
        dropped_pre = stats.get("pre_score_filtered", 0)
        hunted_for_qual = len(getattr(hunt, "candidates", [])) if hunt else 0
        after_prescore = max(hunted_for_qual - dropped_pre, qualified_n)
        after_gemini = qualified_n

    after_dm = 0
    if enh is not None:
        after_dm = sum(len(c.decision_makers) for c in enh.candidates_with_people)

    after_email = 0
    if enr is not None:
        after_email = (getattr(enr, "stats", {}) or {}).get("emails_found", 0)

    after_validation = 0
    ready = needs_review = 0
    if val is not None:
        vstats = getattr(val, "stats", {}) or {}
        ready = vstats.get("ready_to_send", 0)
        needs_review = vstats.get("needs_review", 0)
        after_validation = ready + needs_review

    # Pull resolution rate + apify spend from the hunter's last_metrics if present.
    apify_spend = 0.0
    hm = state.get("_hunter_metrics") or {}
    if hm:
        resolution_rate = hm.get("article_link_resolution_rate", 0.0)
        apify_spend = hm.get("apify_discovery_spend_estimate_usd", 0.0)
        if not source_contributions:
            source_contributions = hm.get("source_contributions", {})

    # Gemini resilience counters (from state if the runner stored them).
    gem = state.get("_gemini_metrics") or {}

    return {
        "funnel_drop_off": {
            "hunted_raw": hunted_raw,
            "after_domain_resolution": after_domain,
            "after_dedupe": after_dedupe,
            "after_prescore_40": after_prescore,
            "after_gemini_70": after_gemini,
            "after_dm_found": after_dm,
            "after_email_found": after_email,
            "after_validation": after_validation,
            "ready_to_send": ready,
            "needs_review": needs_review,
        },
        "source_contributions": source_contributions,
        "article_link_resolution_rate": resolution_rate,
        "gemini_retry_count": gem.get("retry_count", 0),
        "gemini_fallback_count": gem.get("fallback_count", 0),
        "apify_spend_estimate_usd": round(apify_spend, 2),
    }


def _build_run_summary(validated_result, sheets_result: dict, state: "PipelineState") -> dict:
    from sources._utils import utcnow

    # Collect api_credits from all result objects
    api_credits: dict = {}
    for key in ("hunt_result", "qualified_result", "enhanced_result",
                "enriched_result", "messaged_result", "validated_result"):
        result_obj = state.get(key)
        if result_obj is not None:
            credits = getattr(result_obj, "api_credits_used", {}) or {}
            for k, v in credits.items():
                api_credits[k] = api_credits.get(k, 0) + v

    hunt_result = state.get("hunt_result")
    qualified_result = state.get("qualified_result")
    enhanced_result = state.get("enhanced_result")
    enriched_result = state.get("enriched_result")
    messaged_result = state.get("messaged_result")

    return {
        "date": utcnow().strftime("%Y-%m-%d %H:%M"),
        "run_id": validated_result.run_id,
        "segment": validated_result.segment,
        "status": "completed",
        "duration_s": f"{validated_result.duration_seconds:.1f}",
        "candidates_hunted": len(hunt_result.candidates) if hunt_result else 0,
        "qualified": len(qualified_result.qualified) if qualified_result else 0,
        "dms_found": (
            sum(len(c.decision_makers) for c in enhanced_result.candidates_with_people)
            if enhanced_result else 0
        ),
        "emails_found": enriched_result.stats.get("emails_found", 0) if enriched_result else 0,
        "messages_generated": messaged_result.stats.get("messages_generated", 0) if messaged_result else 0,
        "ready_to_send": validated_result.stats.get("ready_to_send", 0),
        "needs_review": validated_result.stats.get("needs_review", 0),
        "rejected": validated_result.stats.get("rejected", 0),
        "api_credits": api_credits,
        "metrics": compute_funnel_metrics(state),
        "errors": "; ".join(
            f"{k}: {v}" for k, v in (state.get("node_errors") or {}).items()
        ),
    }


def _build_empty_run_summary(state: "PipelineState") -> dict:
    from sources._utils import utcnow
    return {
        "date": utcnow().strftime("%Y-%m-%d %H:%M"),
        "run_id": state.get("run_id", "unknown"),
        "segment": state.get("segment", "unknown"),
        "status": "empty_run",
        "duration_s": "0",
        "candidates_hunted": 0,
        "qualified": 0,
        "dms_found": 0,
        "emails_found": 0,
        "messages_generated": 0,
        "ready_to_send": 0,
        "needs_review": 0,
        "rejected": 0,
        "api_credits": {},
        "metrics": compute_funnel_metrics(state),
        "errors": "; ".join(
            f"{k}: {v}" for k, v in (state.get("node_errors") or {}).items()
        ),
    }


# ---------------------------------------------------------------------------
# Conditional routing functions
# ---------------------------------------------------------------------------

def should_continue_after_hunt(state) -> str:
    hunt_result = state.get("hunt_result")
    if hunt_result is None or len(hunt_result.candidates) == 0:
        _log.info(
            "Routing[%s]: hunt returned 0 candidates → skip to dispatch",
            state.get("segment"),
        )
        return "skip_to_dispatch"
    return "qualify"


def should_continue_after_qualify(state) -> str:
    qualified_result = state.get("qualified_result")
    if qualified_result is None or len(qualified_result.qualified) == 0:
        _log.info(
            "Routing[%s]: qualify returned 0 → skip to dispatch",
            state.get("segment"),
        )
        return "skip_to_dispatch"
    return "find_dms"


def should_continue_after_find_dms(state) -> str:
    enhanced_result = state.get("enhanced_result")
    if enhanced_result is None:
        _log.info(
            "Routing[%s]: no enhanced_result → skip to dispatch",
            state.get("segment"),
        )
        return "skip_to_dispatch"
    total_dms = sum(
        len(c.decision_makers) for c in enhanced_result.candidates_with_people
    )
    if total_dms == 0:
        _log.info(
            "Routing[%s]: 0 DMs found → skip to dispatch",
            state.get("segment"),
        )
        return "skip_to_dispatch"
    return "enrich"


def should_continue_after_enrich(state) -> str:
    enriched_result = state.get("enriched_result")
    if enriched_result is None:
        _log.info(
            "Routing[%s]: no enriched_result → skip to dispatch",
            state.get("segment"),
        )
        return "skip_to_dispatch"
    total_emails = sum(
        sum(1 for dm in c.enriched_dms if dm.email_result.email)
        for c in enriched_result.enriched_candidates
    )
    if total_emails == 0:
        _log.info(
            "Routing[%s]: 0 emails found → skip to dispatch",
            state.get("segment"),
        )
        return "skip_to_dispatch"
    return "personalize"
