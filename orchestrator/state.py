"""Phase 9 — Pipeline state model.

Uses TypedDict (LangGraph 1.x native format) rather than Pydantic BaseModel
to avoid serialization issues with the SQLite checkpointer.
All complex result objects are stored as Optional[Any] — LangGraph handles
serialization via its built-in serde.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    # ---- Inputs (required at invocation) ----
    segment: str
    run_id: str
    target_count: int
    started_at: str           # ISO datetime string

    # ---- Per-node outputs (populated as graph progresses) ----
    icp_strategy: Optional[Any]           # IcpStrategy
    hunt_result: Optional[Any]            # HuntResult
    qualified_result: Optional[Any]       # QualifiedResult
    enhanced_result: Optional[Any]        # EnhancedQualifiedResult
    enriched_result: Optional[Any]        # EnrichedResult
    personalization_map: Optional[Any]    # dict[str, PersonalizationContext]
    messaged_result: Optional[Any]        # MessagedResult
    validated_result: Optional[Any]       # ValidatedResult
    sent_result: Optional[Any]            # SentResult

    # ---- Error / progress tracking ----
    node_errors: Dict[str, str]           # {"hunt": "RSS feed timeout", ...}
    nodes_completed: List[str]
    nodes_skipped: List[str]

    # ---- Phase 11 measurement (transient, populated by node_hunt) ----
    _hunter_metrics: Optional[Any]
    _gemini_metrics: Optional[Any]

    # ---- Final outputs ----
    completed_at: Optional[str]           # ISO datetime string
    final_status: str                     # "success" | "partial_success" | "failed"
    duration_seconds: Optional[float]


def make_initial_state(
    segment: str,
    run_id: str,
    target_count: int = 30,
) -> PipelineState:
    """Return a fully-initialised PipelineState with all optional fields set to None."""
    return PipelineState(
        segment=segment,
        run_id=run_id,
        target_count=target_count,
        started_at=datetime.utcnow().isoformat(),
        icp_strategy=None,
        hunt_result=None,
        qualified_result=None,
        enhanced_result=None,
        enriched_result=None,
        personalization_map=None,
        messaged_result=None,
        validated_result=None,
        sent_result=None,
        node_errors={},
        nodes_completed=[],
        nodes_skipped=[],
        completed_at=None,
        final_status="success",
        duration_seconds=None,
    )
