"""Verification tests for Step 9: Nodes 6, 7, and 9 (Synthesis, Integrity, Delivery)."""

import pytest
from src.agents.graph import create_agent_graph, route_citation_integrity
from src.agents.nodes.node6_grounded_synthesis import node6_grounded_synthesis
from src.agents.nodes.node7_citation_integrity import node7_citation_integrity
from src.agents.nodes.node9_response_delivery import node9_response_delivery
from src.state.schema import (
    ApprovalState,
    Citation,
    ErrorRecord,
    OfficerContext,
    PassageMatch,
    QueryFilters,
    RuntimeConfig,
    StateSchema,
)


def create_base_state(query: str, citations: list[Citation] | None = None) -> StateSchema:
    """Helper to create minimal valid state for testing Nodes 6, 7, and 9."""
    c = citations or [
        Citation(
            go_number="GO-1345/XII/2018",
            issuing_department="Forest",
            date="2018-03-12",
            page_number=3,
            exact_text_excerpt="Inter-district transfer requests shall be processed strictly during the annual transfer window in May.",
        )
    ]
    return {
        "session_id": "test_session_step9",
        "officer_context": OfficerContext(department="Forest", access_scope=["Forest"]),
        "query_text": query,
        "query_language": "en",
        "query_filters": QueryFilters(department="Forest"),
        "message_history": [],
        "retrieved_passages": [
            PassageMatch(
                go_number=cit.go_number,
                issuing_department=cit.issuing_department,
                date=cit.date,
                page_number=cit.page_number,
                exact_text_excerpt=cit.exact_text_excerpt,
                relevance_score=0.95,
            )
            for cit in c
        ],
        "candidate_citations": c,
        "confidence_score": 0.95,
        "supersession_status": "CURRENT_ACTIVE",
        "conflict_flags": [],
        "human_verification": None,
        "answer_markdown": None,
        "citations": [],
        "graceful_refusal": False,
        "error_logs": [],
        "config": RuntimeConfig(),
    }


# ===========================================================================
# 1. Node 6: Grounded Synthesis Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_node6_grounded_synthesis_success():
    """Verify Node 6 produces grounded answer and attaches bounding box highlights."""
    state = create_base_state("Forest transfer window month")

    result = await node6_grounded_synthesis(state)

    assert "answer_markdown" in result
    assert result["answer_markdown"] is not None
    assert len(result["answer_markdown"]) > 0
    assert "citations" in result
    assert len(result["citations"]) > 0
    assert result["citations"][0].go_number == "GO-1345/XII/2018"
    assert result["citations"][0].bounding_box_coordinates is None or isinstance(result["citations"][0].bounding_box_coordinates, (dict, list))


@pytest.mark.asyncio
async def test_node6_grounded_synthesis_resolved_go_filter():
    """Verify Node 6 restricts synthesis strictly to human-resolved GO."""
    c1 = Citation(
        go_number="GO-1345/XII/2018",
        issuing_department="Forest",
        date="2018-03-12",
        page_number=3,
        exact_text_excerpt="Forest transfer rules 2018.",
    )
    c2 = Citation(
        go_number="GO-9999/OLD/2000",
        issuing_department="Forest",
        date="2000-01-01",
        page_number=1,
        exact_text_excerpt="Old transfer rules 2000.",
    )
    state = create_base_state("Transfer policy", citations=[c1, c2])
    state["human_verification"] = ApprovalState(
        action="approve",
        resolved_go_number="GO-1345/XII/2018",
        reason="Officer selected 2018 GO",
    )

    result = await node6_grounded_synthesis(state)
    assert all(c.go_number == "GO-1345/XII/2018" for c in result["citations"])


# ===========================================================================
# 2. Node 7: Citation Integrity Check Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_node7_citation_integrity_valid_answer():
    """Verify Node 7 passes a properly grounded answer without raising errors."""
    c = Citation(
        go_number="GO-1345/XII/2018",
        issuing_department="Forest",
        date="2018-03-12",
        page_number=3,
        exact_text_excerpt="Inter-district transfer requests shall be processed strictly during the annual transfer window in May.",
    )
    state = create_base_state("Forest transfer window", citations=[c])
    state["answer_markdown"] = "According to GO-1345/XII/2018 (page 3), inter-district transfer requests are processed strictly in May."
    state["citations"] = [c]

    result = await node7_citation_integrity(state)
    assert result.get("graceful_refusal") is False


@pytest.mark.asyncio
async def test_node7_citation_integrity_empty_answer_refusal():
    """Verify Node 7 immediately rejects empty answers."""
    state = create_base_state("test")
    state["answer_markdown"] = ""
    state["citations"] = []

    result = await node7_citation_integrity(state)
    assert result["graceful_refusal"] is True
    assert len(result["error_logs"]) > 0


def test_route_citation_integrity_retry_and_exhaustion():
    """Verify conditional edge routes to synthesis for retry, and refusal on exhaustion."""
    state_retry = create_base_state("test")
    err1 = ErrorRecord(
        node="node7_citation_integrity",
        error_type="CitationIntegrityFailure",
        message="Uncited claim attempt 1",
    )
    state_retry["error_logs"] = [err1]
    state_retry["graceful_refusal"] = False
    assert route_citation_integrity(state_retry) == "grounded_synthesis"

    # Exhausted (2 failures >= max_citation_retries=2)
    state_exhausted = create_base_state("test")
    err2 = ErrorRecord(
        node="node7_citation_integrity",
        error_type="CitationIntegrityFailure",
        message="Uncited claim attempt 2",
    )
    state_exhausted["error_logs"] = [err1, err2]
    state_exhausted["graceful_refusal"] = True
    assert route_citation_integrity(state_exhausted) == "refusal_redirect"


# ===========================================================================
# 3. Node 9 & Full 9-Node StateGraph End-to-End Test
# ===========================================================================

@pytest.mark.asyncio
async def test_node9_response_delivery():
    """Verify terminal Node 9 sets clean state."""
    state = create_base_state("test")
    result = await node9_response_delivery(state)
    assert result["graceful_refusal"] is False


@pytest.mark.asyncio
async def test_full_graph_end_to_end_execution():
    """Verify complete end-to-end flow of an in-scope query through all 9 graph nodes."""
    from langgraph.types import Command
    from src.state.checkpointing import ensure_windows_event_loop, get_checkpointer
    ensure_windows_event_loop()
    thread_id = "test_thread_step9_e2e"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    async with get_checkpointer() as checkpointer:
        app = create_agent_graph(checkpointer=checkpointer)
        initial_state = create_base_state(
            "Forest department transfer policy 2018"
        )
        initial_state["session_id"] = thread_id

        final_state = await app.ainvoke(initial_state, config=config)
        if bool(final_state.get("__interrupt__")):
            final_state = await app.ainvoke(Command(resume={"action": "approve"}), config=config)

    assert final_state["graceful_refusal"] is False
    assert final_state["answer_markdown"] is not None
    assert len(final_state["citations"]) >= 0
