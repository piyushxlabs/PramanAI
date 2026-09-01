"""LangGraph StateGraph orchestration topology for ShasanAI GO-Retrieval & Citation Agent.

Implements the exact 9-node architecture specified in AGENT_ORCHESTRATION_BLUEPRINT.md Section 4:
1. query_interpretation
2. scope_screen
3. retrieval_invocation
4. confidence_supersession
5. human_verification_interrupt
6. grounded_synthesis
7. citation_integrity
8. refusal_redirect (terminal)
9. response_delivery (terminal)
"""

import logging
import time
from typing import Any, Callable, Coroutine, Literal
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.nodes.node1_query_interpretation import node1_query_interpretation
from src.agents.nodes.node2_scope_screening import node2_scope_screening
from src.agents.nodes.node3_retrieval_invocation import node3_retrieval_invocation
from src.agents.nodes.node4_supersession_confidence import node4_supersession_confidence
from src.agents.nodes.node5_human_verification import node5_human_verification
from src.agents.nodes.node6_grounded_synthesis import node6_grounded_synthesis
from src.agents.nodes.node7_citation_integrity import node7_citation_integrity
from src.agents.nodes.node8_refusal_redirect import node8_refusal_redirect
from src.agents.nodes.node9_response_delivery import node9_response_delivery
from src.state.schema import StateSchema

logger = logging.getLogger("shasanai.graph")

NODE_NAMES_ORDER: dict[str, str] = {
    "query_interpretation": "1/9",
    "scope_screen": "2/9",
    "retrieval_invocation": "3/9",
    "confidence_supersession": "4/9",
    "human_verification_interrupt": "5/9",
    "grounded_synthesis": "6/9",
    "citation_integrity": "7/9",
    "refusal_redirect": "8/9",
    "response_delivery": "9/9",
}


def wrap_node_with_logging(node_name: str, node_func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Wraps a graph node to print real-time start/finish status and latency to the console."""
    step_num = NODE_NAMES_ORDER.get(node_name, "?/9")

    async def _wrapped(state: StateSchema, *args: Any, **kwargs: Any) -> Any:
        session_id = state.get("session_id", "unknown")
        print(f"\n>>> [NODE {step_num}: {node_name}] Starting execution for session: {session_id}...", flush=True)
        logger.info(f">>> [NODE {step_num}: {node_name}] Starting execution for session: {session_id}")
        start_time = time.perf_counter()
        try:
            result = await node_func(state, *args, **kwargs)
            duration = time.perf_counter() - start_time
            print(f"<<< [NODE {step_num}: {node_name}] Completed in {duration:.2f}s", flush=True)
            logger.info(f"<<< [NODE {step_num}: {node_name}] Completed in {duration:.2f}s")
            return result
        except Exception as exc:
            duration = time.perf_counter() - start_time
            print(f"!!! [NODE {step_num}: {node_name}] Failed after {duration:.2f}s: {exc!s}", flush=True)
            logger.error(f"!!! [NODE {step_num}: {node_name}] Failed after {duration:.2f}s: {exc!s}")
            raise

    return _wrapped


# ---------------------------------------------------------------------------
# Conditional Routing Edges
# ---------------------------------------------------------------------------

def route_scope_screen(state: StateSchema) -> Literal["refusal_redirect", "retrieval_invocation"]:
    """Conditional Edge: Route to refusal if out-of-scope; else proceed to hybrid retrieval."""
    if state.get("graceful_refusal", False):
        return "refusal_redirect"
    return "retrieval_invocation"


def route_confidence_supersession(
    state: StateSchema,
) -> Literal["human_verification_interrupt", "grounded_synthesis"]:
    """Conditional Edge: Route to Human Verification Interrupt if low confidence, conflicts, or PII detected.
    
    Enforces rule: confidence_score < 0.85 OR has_supersession_conflict / len(conflict_flags) > 0.
    """
    confidence = state.get("confidence_score", 0.0)
    has_conflict = state.get("has_supersession_conflict", False) or len(state.get("conflict_flags", [])) > 0

    if confidence < 0.85 or has_conflict:
        return "human_verification_interrupt"  # Trigger Node 5
    return "grounded_synthesis"     # Direct to Node 6


def route_human_verification(
    state: StateSchema,
) -> Literal["grounded_synthesis", "refusal_redirect"]:
    """Conditional Edge: Route to synthesis if approved; else abort to refusal/redirect."""
    if state.get("graceful_refusal", False):
        return "refusal_redirect"

    human_verification = state.get("human_verification")
    if human_verification is not None:
        action = getattr(human_verification, "action", None) or (
            human_verification.get("action") if isinstance(human_verification, dict) else None
        )
        if action in ["approve", "approve_with_edit"]:
            return "grounded_synthesis"
    return "refusal_redirect"


def route_citation_integrity(
    state: StateSchema,
) -> Literal["response_delivery", "grounded_synthesis", "refusal_redirect"]:
    """Conditional Edge: Route to delivery if valid, retry synthesis if under retry cap, else refusal."""
    if state.get("graceful_refusal", False):
        return "refusal_redirect"

    error_logs = state.get("error_logs", [])
    config = state.get("config")
    max_retries = getattr(config, "max_citation_retries", 2) if config else 2

    # Check if latest record was a citation failure
    if error_logs and getattr(error_logs[-1], "error_type", None) == "CitationIntegrityFailure":
        failures = sum(
            1 for e in error_logs if getattr(e, "error_type", None) == "CitationIntegrityFailure"
        )
        if failures < max_retries:
            return "grounded_synthesis"
        return "refusal_redirect"

    return "response_delivery"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_scaffold_graph() -> StateGraph:
    """Constructs the 9-node LangGraph StateGraph topology for ShasanAI."""
    workflow = StateGraph(StateSchema)

    # Register wrapped nodes
    workflow.add_node("query_interpretation", wrap_node_with_logging("query_interpretation", node1_query_interpretation))
    workflow.add_node("scope_screen", wrap_node_with_logging("scope_screen", node2_scope_screening))
    workflow.add_node("retrieval_invocation", wrap_node_with_logging("retrieval_invocation", node3_retrieval_invocation))
    workflow.add_node("confidence_supersession", wrap_node_with_logging("confidence_supersession", node4_supersession_confidence))
    workflow.add_node("human_verification_interrupt", wrap_node_with_logging("human_verification_interrupt", node5_human_verification))
    workflow.add_node("grounded_synthesis", wrap_node_with_logging("grounded_synthesis", node6_grounded_synthesis))
    workflow.add_node("citation_integrity", wrap_node_with_logging("citation_integrity", node7_citation_integrity))
    workflow.add_node("refusal_redirect", wrap_node_with_logging("refusal_redirect", node8_refusal_redirect))
    workflow.add_node("response_delivery", wrap_node_with_logging("response_delivery", node9_response_delivery))

    # Edge definitions
    workflow.add_edge(START, "query_interpretation")
    workflow.add_edge("query_interpretation", "scope_screen")

    workflow.add_conditional_edges(
        "scope_screen",
        route_scope_screen,
        {
            "refusal_redirect": "refusal_redirect",
            "retrieval_invocation": "retrieval_invocation",
        },
    )

    workflow.add_edge("retrieval_invocation", "confidence_supersession")

    workflow.add_conditional_edges(
        "confidence_supersession",
        route_confidence_supersession,
        {
            "human_verification_interrupt": "human_verification_interrupt",
            "grounded_synthesis": "grounded_synthesis",
        },
    )

    workflow.add_conditional_edges(
        "human_verification_interrupt",
        route_human_verification,
        {
            "grounded_synthesis": "grounded_synthesis",
            "refusal_redirect": "refusal_redirect",
        },
    )

    workflow.add_edge("grounded_synthesis", "citation_integrity")

    workflow.add_conditional_edges(
        "citation_integrity",
        route_citation_integrity,
        {
            "response_delivery": "response_delivery",
            "grounded_synthesis": "grounded_synthesis",
            "refusal_redirect": "refusal_redirect",
        },
    )

    workflow.add_edge("refusal_redirect", END)
    workflow.add_edge("response_delivery", END)

    return workflow


def create_agent_graph(checkpointer: Any = None) -> CompiledStateGraph:
    """Compiles the StateGraph workflow with optional persistent checkpointer."""
    workflow = build_scaffold_graph()
    return workflow.compile(
        checkpointer=checkpointer,
    )
