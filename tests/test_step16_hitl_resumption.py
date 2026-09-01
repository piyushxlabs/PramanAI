"""Step 16 Verification Test Suite: HITL Graph Resumption Endpoints & SSE Streaming.

Evaluates the 3 mandatory HITL resumption scenarios from Section 9.3:
1. Approval from Checkpoint
2. Approval-with-Resolution from Checkpoint
3. Denial from Checkpoint with Graceful Refusal
"""

import json
import pytest
from src.agents.graph import create_agent_graph
from src.state.checkpointing import ensure_windows_event_loop, get_checkpointer
from src.state.schema import (
    ConflictRecord,
    OfficerContext,
    QueryFilters,
    RuntimeConfig,
    StateSchema,
)
from src.ui.hitl_resumption import resume_hitl_stream


def create_step16_state(query: str, thread_id: str) -> StateSchema:
    """Creates a base state for testing Step 16 resumption scenarios."""
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
        "config": RuntimeConfig(confidence_threshold_low=0.99),
    }


@pytest.mark.asyncio
async def test_hitl_resumption_approval_scenario():
    """Scenario 1: Verify StateGraph pause at Node 5 and SSE resumption upon officer approval."""
    ensure_windows_event_loop()
    thread_id = "test_thread_hitl_001"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    async with get_checkpointer() as checkpointer:
        app = create_agent_graph(checkpointer=checkpointer)
        initial_state = create_step16_state("2018 forest transfer policy GO", thread_id)

        # 1. Run until pause at Node 5
        res = await app.ainvoke(initial_state, config=config)
        assert bool(res.get("__interrupt__")) is True

    # 2. Resume via resume_hitl_stream
    events: list[dict] = []
    async for raw_event in resume_hitl_stream(
        checkpoint_id=thread_id,
        action="approve",
        reason="Officer confirmed policy validity",
    ):
        for line in raw_event.strip().split("\n"):
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except Exception:
                    pass

    # Verify SSE events emitted
    event_types = [e.get("type") for e in events]
    assert "message-start" in event_types
    assert "data-graph-step" in event_types
    assert "text-delta" in event_types
    assert "finish" in event_types

    finish_event = next(e for e in events if e.get("type") == "finish")
    assert finish_event["finishReason"] == "success"

    text_deltas = [e["delta"] for e in events if e.get("type") == "text-delta"]
    assert len(text_deltas) > 0
    full_answer = "".join(text_deltas)
    assert len(full_answer) > 10


@pytest.mark.asyncio
async def test_hitl_resumption_approval_with_resolution_scenario():
    """Scenario 2: Verify StateGraph pause with conflict and resumption with resolved GO number."""
    ensure_windows_event_loop()
    thread_id = "test_thread_hitl_001"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    async with get_checkpointer() as checkpointer:
        app = create_agent_graph(checkpointer=checkpointer)
        initial_state = create_step16_state("2018 forest transfer policy GO", thread_id)

        # 1. Run until pause at Node 5
        res = await app.ainvoke(initial_state, config=config)
        assert bool(res.get("__interrupt__")) is True

    # 2. Resume with resolved GO
    events: list[dict] = []
    async for raw_event in resume_hitl_stream(
        checkpoint_id=thread_id,
        action="approve",
        resolved_go_number="GO-1345/XII/2018",
        reason="Officer selected active policy GO-1345",
    ):
        for line in raw_event.strip().split("\n"):
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except Exception:
                    pass

    event_types = [e.get("type") for e in events]
    assert "message-start" in event_types
    assert "data-graph-step" in event_types
    assert "text-delta" in event_types
    assert "finish" in event_types

    finish_event = next(e for e in events if e.get("type") == "finish")
    assert finish_event["finishReason"] == "success"

    text_deltas = [e["delta"] for e in events if e.get("type") == "text-delta"]
    full_answer = "".join(text_deltas)
    assert len(full_answer) > 10


@pytest.mark.asyncio
async def test_hitl_resumption_denial_scenario():
    """Scenario 3: Verify StateGraph pause and resumption with denial routes to Refusal/Redirect."""
    ensure_windows_event_loop()
    thread_id = "test_thread_hitl_001"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    async with get_checkpointer() as checkpointer:
        app = create_agent_graph(checkpointer=checkpointer)
        initial_state = create_step16_state("2018 forest transfer policy GO", thread_id)

        # 1. Run until pause at Node 5
        res = await app.ainvoke(initial_state, config=config)
        assert bool(res.get("__interrupt__")) is True

    # 2. Resume with denial
    events: list[dict] = []
    async for raw_event in resume_hitl_stream(
        checkpoint_id=thread_id,
        action="deny",
        reason="Officer rejected administrative query",
    ):
        for line in raw_event.strip().split("\n"):
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except Exception:
                    pass

    event_types = [e.get("type") for e in events]
    assert "message-start" in event_types
    assert "data-graph-step" in event_types
    assert "finish" in event_types

    finish_event = next(e for e in events if e.get("type") == "finish")
    assert finish_event["finishReason"] == "refused"

    text_deltas = [e["delta"] for e in events if e.get("type") == "text-delta"]
    full_answer = "".join(text_deltas)
    assert len(full_answer) > 0
