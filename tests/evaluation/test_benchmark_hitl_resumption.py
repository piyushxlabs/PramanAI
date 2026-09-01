"""HITL Graph Resumption Benchmark Suite for ShasanAI.

Evaluates end-to-end human verification interruption, durable checkpointing, and graph resumption flows.
"""

import pytest
import uuid
from langgraph.types import Command
from src.agents.graph import create_agent_graph, route_confidence_supersession, route_human_verification
from src.state.checkpointing import get_checkpointer, ensure_windows_event_loop
from src.state.schema import (
    ApprovalState,
    ConflictRecord,
    OfficerContext,
    QueryFilters,
    RuntimeConfig,
    StateSchema,
)


def create_eval_base_state(
    query: str, thread_id: str, config: RuntimeConfig | None = None
) -> StateSchema:
    """Helper to create minimal valid state for testing evaluation scenarios."""
    return {
        "session_id": thread_id,
        "officer_context": OfficerContext(department="Forest", access_scope=["Forest"]),
        "query_text": query,
        "query_language": "en",
        "query_filters": QueryFilters(department="Forest"),
        "message_history": [],
        "retrieved_passages": [],
        "candidate_citations": [],
        "confidence_score": 0.40,
        "supersession_status": "CURRENT_ACTIVE",
        "conflict_flags": [
            ConflictRecord(go_numbers=["GO-1345/XII/2018", "GO-562/XXX/2014"], description="Version conflict")
        ],
        "human_verification": None,
        "answer_markdown": None,
        "citations": [],
        "graceful_refusal": False,
        "error_logs": [],
        "config": config or RuntimeConfig(),
    }


def test_benchmark_hitl_routing_gates():
    """Verify conditional edge routing rules for Node 4 and Node 5."""
    # Low confidence (< 0.60) triggers interrupt
    state_low_conf = create_eval_base_state("query", "gate_test_1")
    state_low_conf["confidence_score"] = 0.45
    state_low_conf["conflict_flags"] = []
    assert route_confidence_supersession(state_low_conf) == "human_verification_interrupt"

    # Non-empty conflict_flags triggers interrupt
    state_conflict = create_eval_base_state("query", "gate_test_2")
    state_conflict["confidence_score"] = 0.90
    state_conflict["conflict_flags"] = [
        ConflictRecord(go_numbers=["GO-1345/XII/2018", "GO-999/XII/2021"], description="Contradictory transfer timelines")
    ]
    assert route_confidence_supersession(state_conflict) == "human_verification_interrupt"

    # Approval routes to grounded_synthesis
    state_approved = create_eval_base_state("query", "gate_test_3")
    state_approved["human_verification"] = ApprovalState(
        action="approve", checkpoint_id="chk_1", reason="Officer approved"
    )
    assert route_human_verification(state_approved) == "grounded_synthesis"

    # Denial routes to refusal_redirect
    state_denied = create_eval_base_state("query", "gate_test_4")
    state_denied["human_verification"] = ApprovalState(
        action="deny", checkpoint_id="chk_2", reason="Officer denied"
    )
    assert route_human_verification(state_denied) == "refusal_redirect"


@pytest.mark.asyncio
async def test_benchmark_hitl_approval_and_denial_checkpoint_flows():
    """Verify StateGraph pause at Node 5 on low confidence and deterministic resumption with approve/deny commands."""
    ensure_windows_event_loop()
    async with get_checkpointer() as checkpointer:
        app = create_agent_graph(checkpointer=checkpointer)

        # Flow 1: Approval using fresh unique thread ID
        thread_id_app = f"eval_hitl_app_{uuid.uuid4().hex}"
        config_app = {"configurable": {"thread_id": thread_id_app, "checkpoint_ns": ""}}
        initial_state_app = create_eval_base_state(
            "2018 forest transfer policy GO",
            thread_id_app,
        )

        res_app = await app.ainvoke(initial_state_app, config=config_app)
        assert bool(res_app.get("__interrupt__")) is True

        resume_cmd_app = Command(resume={"action": "approve", "reason": "Officer verified policy manually"})
        final_state_app = await app.ainvoke(resume_cmd_app, config=config_app)
        assert final_state_app["human_verification"] is not None
        assert final_state_app["human_verification"].action == "approve"

        # Flow 2: Denial using fresh unique thread ID
        thread_id_den = f"eval_hitl_den_{uuid.uuid4().hex}"
        config_den = {"configurable": {"thread_id": thread_id_den, "checkpoint_ns": ""}}
        initial_state_den = create_eval_base_state(
            "2018 forest transfer policy GO",
            thread_id_den,
        )

        res_den = await app.ainvoke(initial_state_den, config=config_den)
        assert bool(res_den.get("__interrupt__")) is True

        resume_cmd_den = Command(resume={"action": "deny", "reason": "Officer declined"})
        final_state_den = await app.ainvoke(resume_cmd_den, config=config_den)
        assert final_state_den["human_verification"] is not None
        assert final_state_den["human_verification"].action == "deny"
        assert final_state_den.get("graceful_refusal") is True
