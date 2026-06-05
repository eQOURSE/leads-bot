"""Phase 9 — Pipeline runner.

PipelineRunner is the single entry-point for executing the lead generation
pipeline. It wires together:
  - AgentRegistry (all agents/clients built once at startup)
  - AsyncSqliteSaver checkpointer (resume-from-crash support)
  - The compiled per-segment LangGraph graph
  - asyncio.gather for concurrent multi-segment execution
  - One consolidated Telegram digest after all segments complete

Usage:
    runner = PipelineRunner(settings)
    results = await runner.run_all_segments(target_count=30)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Optional

from config.logging_config import setup_logging
from config.settings import Settings
from orchestrator.agents_registry import AgentRegistry
from orchestrator.graph import build_pipeline_graph
from orchestrator.state import PipelineState, make_initial_state

_log = setup_logging("orchestrator.runner")

_CHECKPOINTS_DB = "./data/checkpoints.db"


class PipelineRunner:
    """Orchestrates end-to-end pipeline execution across all three segments."""

    def __init__(self, settings: Settings, dry_run: bool = False) -> None:
        self.settings = settings
        self.dry_run = dry_run
        self.agents = AgentRegistry(settings)
        self._checkpointer_ctx = None
        self._checkpointer = None
        self._compiled_graph = None

    # -----------------------------------------------------------------------
    # Lifecycle: checkpointer must be opened as an async context manager
    # -----------------------------------------------------------------------

    async def __aenter__(self) -> "PipelineRunner":
        await self._init_checkpointer()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._checkpointer_ctx is not None:
            try:
                await self._checkpointer_ctx.__aexit__(*exc)
            except Exception as e:
                _log.warning("Error closing checkpointer: %s", e)

    async def _init_checkpointer(self) -> None:
        """Try to initialise AsyncSqliteSaver. Fall back to MemorySaver on error."""
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            import os
            os.makedirs("./data", exist_ok=True)
            ctx = AsyncSqliteSaver.from_conn_string(_CHECKPOINTS_DB)
            self._checkpointer = await ctx.__aenter__()
            self._checkpointer_ctx = ctx
            _log.info("Checkpointer: AsyncSqliteSaver → %s", _CHECKPOINTS_DB)
        except Exception as exc:
            _log.warning(
                "AsyncSqliteSaver unavailable (%s), falling back to MemorySaver", exc
            )
            from langgraph.checkpoint.memory import MemorySaver
            self._checkpointer = MemorySaver()

        self._compiled_graph = build_pipeline_graph(self.agents, self._checkpointer)

    # -----------------------------------------------------------------------
    # Single-segment run
    # -----------------------------------------------------------------------

    async def run_segment(
        self,
        segment: str,
        target_count: int = 30,
        thread_id: Optional[str] = None,
    ) -> PipelineState:
        """Run one segment end-to-end. Returns the final PipelineState."""
        if self._compiled_graph is None:
            await self._init_checkpointer()

        run_id = thread_id or f"{segment}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        initial_state = make_initial_state(segment, run_id, target_count)

        # In dry-run mode, stop before dispatch
        if self.dry_run:
            _log.info("[DRY-RUN] Segment %s — dispatch skipped", segment)
            initial_state["nodes_skipped"] = ["dispatch"]
            initial_state["final_status"] = "success"
            initial_state["completed_at"] = datetime.utcnow().isoformat()
            return initial_state

        config = {"configurable": {"thread_id": run_id}}
        t0_abs = datetime.utcnow()

        try:
            result_dict = await self._compiled_graph.ainvoke(
                dict(initial_state), config=config
            )
            final_state: PipelineState = result_dict  # type: ignore[assignment]
            completed_at = datetime.utcnow()
            started_at = datetime.fromisoformat(
                final_state.get("started_at") or t0_abs.isoformat()
            )
            duration = (completed_at - started_at).total_seconds()
            final_state["completed_at"] = completed_at.isoformat()
            final_state["duration_seconds"] = duration
            final_state["final_status"] = _compute_final_status(final_state)
            _log.info(
                "Pipeline[%s] %s in %.1fs",
                segment, final_state["final_status"], duration,
            )
            return final_state

        except Exception as exc:
            _log.error("Pipeline[%s] crashed: %s", segment, exc, exc_info=True)
            initial_state["node_errors"] = {
                **initial_state.get("node_errors", {}),
                "graph_crash": str(exc),
            }
            initial_state["final_status"] = "failed"
            initial_state["completed_at"] = datetime.utcnow().isoformat()
            return initial_state

    # -----------------------------------------------------------------------
    # Resume from checkpoint
    # -----------------------------------------------------------------------

    async def resume_segment(
        self,
        thread_id: str,
        target_count: int = 30,
    ) -> PipelineState:
        """Resume a segment run from an existing checkpoint thread_id."""
        segment = thread_id.split("_")[0]  # best-effort extraction
        _log.info("Resuming segment %r from checkpoint thread_id=%s", segment, thread_id)
        return await self.run_segment(segment, target_count, thread_id=thread_id)

    # -----------------------------------------------------------------------
    # Multi-segment concurrent run
    # -----------------------------------------------------------------------

    async def run_all_segments(
        self,
        target_count: int = 30,
    ) -> dict[str, PipelineState]:
        """Run all 3 segments concurrently. Returns {segment: PipelineState}.

        After all segments complete, sends ONE consolidated Telegram digest.
        """
        segments = self.agents.icp_strategist.list_segments()
        _log.info("Starting concurrent run for segments: %s", segments)

        tasks = [self.run_segment(seg, target_count) for seg in segments]
        results_list = await asyncio.gather(*tasks, return_exceptions=False)
        # run_segment catches all exceptions internally; return_exceptions=False is safe
        results: dict[str, PipelineState] = dict(zip(segments, results_list))

        # ---- Log cost summary ----
        _log_cost_summary(results)

        # ---- Consolidated Telegram digest ----
        validated_by_segment = {
            seg: state["validated_result"]
            for seg, state in results.items()
            if state.get("validated_result") is not None
        }

        if self.dry_run:
            _log.info("[DRY-RUN] Telegram digest skipped")
        elif validated_by_segment:
            # Full digest — at least one segment produced validated leads
            try:
                sheets_url = _build_sheets_url(self.settings)
                _log.info(
                    "Sending consolidated Telegram digest for %d segment(s)…",
                    len(validated_by_segment),
                )
                tg_result = await self.agents.telegram_sink.send_run_digest(
                    validated_by_segment, sheets_url
                )
                if tg_result.get("error"):
                    _log.error("Telegram digest failed: %s", tg_result["error"])
                else:
                    _log.info(
                        "Telegram digest sent (message_id=%s, leads=%d)",
                        tg_result.get("message_id"),
                        tg_result.get("leads_included", 0),
                    )
            except Exception as exc:
                _log.error("Consolidated Telegram digest failed: %s", exc, exc_info=True)
        elif self.settings.TELEGRAM_SEND_EMPTY_DIGEST:
            # Empty-run digest — 0 leads, but operator wants the audit signal
            try:
                sheets_url = _build_sheets_url(self.settings)
                segment_stats = {
                    seg: {
                        "hunt_count": (
                            state["hunt_result"].merged_count
                            if state.get("hunt_result") is not None else 0
                        ),
                        "qualified_count": (
                            len(state["qualified_result"].qualified)
                            if state.get("qualified_result") is not None else 0
                        ),
                        "after_dedupe": (
                            state["hunt_result"].after_dedupe
                            if state.get("hunt_result") is not None else None
                        ),
                    }
                    for seg, state in results.items()
                }
                first_run_id = next(iter(results.values())).get("run_id", "unknown")
                _log.info("Sending empty-run Telegram digest (0 leads across all segments)…")
                msg_id = await self.agents.telegram_sink.send_empty_run_digest(
                    segment_stats, run_id=first_run_id, sheets_url=sheets_url
                )
                if msg_id is not None:
                    _log.info("Empty-run digest sent (message_id=%s)", msg_id)
                else:
                    _log.error("Empty-run digest failed to send")
            except Exception as exc:
                _log.error("Empty-run Telegram digest failed: %s", exc, exc_info=True)
        else:
            _log.info(
                "No validated results and TELEGRAM_SEND_EMPTY_DIGEST=false — digest skipped"
            )

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_final_status(state: PipelineState) -> str:
    """Determine final_status from the completed state."""
    node_errors = state.get("node_errors") or {}
    sent_result = state.get("sent_result")
    validated_result = state.get("validated_result")

    if not node_errors:
        return "success"
    # Has errors — check if we still produced some output
    if validated_result is not None:
        stats = getattr(validated_result, "stats", {}) or {}
        if stats.get("ready_to_send", 0) > 0 or stats.get("needs_review", 0) > 0:
            return "partial_success"
    if sent_result is not None:
        return "partial_success"
    return "failed"


def _build_sheets_url(settings: Settings) -> str:
    if settings.SHEET_ID:
        return f"https://docs.google.com/spreadsheets/d/{settings.SHEET_ID}/edit"
    return ""


def _log_cost_summary(results: dict[str, PipelineState]) -> None:
    """Log aggregated API credits per segment to the console/log."""
    for segment, state in results.items():
        credits_total: dict[str, int] = {}
        section_credits: dict[str, dict] = {}

        for node_key in (
            "hunt_result", "qualified_result", "enhanced_result",
            "enriched_result", "personalization_map", "messaged_result",
            "validated_result",
        ):
            result_obj = state.get(node_key)
            if result_obj is None:
                continue
            # personalization_map is a plain dict — no credits field
            if isinstance(result_obj, dict):
                continue
            credits = getattr(result_obj, "api_credits_used", {}) or {}
            if credits:
                section_credits[node_key.replace("_result", "")] = credits
                for k, v in credits.items():
                    credits_total[k] = credits_total.get(k, 0) + v

        total_calls = sum(credits_total.values())
        lines = [f"Pipeline[{segment}] Costs:"]
        for section, creds in section_credits.items():
            creds_str = "  ".join(f"{k}={v}" for k, v in creds.items())
            lines.append(f"  {section}: {creds_str}")
        lines.append(f"  TOTAL: ~{total_calls} API calls")
        _log.info("\n".join(lines))
