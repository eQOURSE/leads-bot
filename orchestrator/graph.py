"""Phase 9 — Graph builder.

Compiles the per-segment LangGraph StateGraph with:
  - 9 linear nodes (load_icp → hunt → qualify → find_dms → enrich →
    personalize → write_messages → validate → dispatch)
  - 4 conditional edges that short-circuit to dispatch on empty results
  - An optional checkpointer for resume-from-checkpoint support
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, END

from orchestrator.state import PipelineState
from orchestrator.nodes import (
    node_load_icp,
    node_hunt,
    node_qualify,
    node_find_dms,
    node_enrich,
    node_personalize,
    node_write_messages,
    node_validate,
    node_dispatch,
    should_continue_after_hunt,
    should_continue_after_qualify,
    should_continue_after_find_dms,
    should_continue_after_enrich,
)

if TYPE_CHECKING:
    from orchestrator.agents_registry import AgentRegistry


def build_pipeline_graph(agents: "AgentRegistry", checkpointer=None):
    """Build and compile the per-segment pipeline graph.

    Args:
        agents: Fully initialised AgentRegistry.
        checkpointer: An optional LangGraph checkpointer (e.g. AsyncSqliteSaver).
            When provided, the graph supports resume-from-checkpoint via thread_id.

    Returns:
        A compiled LangGraph CompiledStateGraph ready for ainvoke().
    """
    graph = StateGraph(PipelineState)

    # ---- Register nodes using proper async wrappers ----
    # LangGraph 1.x: lambdas returning coroutines don't work; use functools.partial
    # or proper async wrapper functions.

    import functools

    async def _load_icp(s):
        return await node_load_icp(s, agents)

    async def _hunt(s):
        return await node_hunt(s, agents)

    async def _qualify(s):
        return await node_qualify(s, agents)

    async def _find_dms(s):
        return await node_find_dms(s, agents)

    async def _enrich(s):
        return await node_enrich(s, agents)

    async def _personalize(s):
        return await node_personalize(s, agents)

    async def _write_messages(s):
        return await node_write_messages(s, agents)

    async def _validate(s):
        return await node_validate(s, agents)

    async def _dispatch(s):
        return await node_dispatch(s, agents)

    graph.add_node("load_icp",       _load_icp)
    graph.add_node("hunt",           _hunt)
    graph.add_node("qualify",        _qualify)
    graph.add_node("find_dms",       _find_dms)
    graph.add_node("enrich",         _enrich)
    graph.add_node("personalize",    _personalize)
    graph.add_node("write_messages", _write_messages)
    graph.add_node("validate",       _validate)
    graph.add_node("dispatch",       _dispatch)

    # ---- Entry point ----
    graph.set_entry_point("load_icp")

    # ---- Linear edges ----
    graph.add_edge("load_icp", "hunt")

    # ---- Conditional edges (short-circuit to dispatch on empty results) ----
    graph.add_conditional_edges(
        "hunt",
        should_continue_after_hunt,
        {"qualify": "qualify", "skip_to_dispatch": "dispatch"},
    )
    graph.add_conditional_edges(
        "qualify",
        should_continue_after_qualify,
        {"find_dms": "find_dms", "skip_to_dispatch": "dispatch"},
    )
    graph.add_conditional_edges(
        "find_dms",
        should_continue_after_find_dms,
        {"enrich": "enrich", "skip_to_dispatch": "dispatch"},
    )
    graph.add_conditional_edges(
        "enrich",
        should_continue_after_enrich,
        {"personalize": "personalize", "skip_to_dispatch": "dispatch"},
    )

    # ---- Tail of the happy path ----
    graph.add_edge("personalize",    "write_messages")
    graph.add_edge("write_messages", "validate")
    graph.add_edge("validate",       "dispatch")
    graph.add_edge("dispatch",       END)

    # ---- Compile ----
    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
