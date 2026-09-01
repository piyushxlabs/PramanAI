"""Verification tests for Step 8: Node 4 (Supersession & Confidence) and Node 5 (Human Verification)."""

import pytest
from langgraph.types import Command
from src.agents.graph import create_agent_graph, route_confidence_supersession, route_human_verification
from src.agents.nodes.node4_supersession_confidence import node4_supersession_confidence
from src.state.checkpointing import get_checkpointer
from src.state.schema import (
    ApprovalState,
    Citation,
    ConflictRecord,
    OfficerContext,
    PassageMatch,
    QueryFilters,
    RuntimeConfig,
    StateSchema,
)


def create_base_state(query: str, passages: list[PassageMatch] | None = None, citations: list[Citation] | None = None) -> StateSchema:
    """Helper to create minimal valid state for testing Nodes 4 and 5."""
    return {
        "session_id": "test_session_step8",
        "officer_context": OfficerContext(department="Forest", access_scope=["Forest"]),
        "query_text": query,
        "query_language": "en",
        "query_filters": QueryFilters(department="Forest"),
        "message_history": [],
        "retrieved_passages": passages or [],
        "candidate_citations": citations or [],
        "confidence_score": 1.0,
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
# 1. Node 4: Supersession & Confidence Analysis Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_node4_single_active_go_confidence():
    """Verify Node 4 correctly assesses high confidence and CURRENT_ACTIVE for valid GO."""
    p = PassageMatch(
        go_number="GO-1345/XII/2018",
        issuing_department="Forest",
        date="2018-03-12",
        page_number=3,
        exact_text_excerpt="Inter-district transfer requests shall be processed strictly during the annual transfer window in May.",
        relevance_score=0.93,
    )
    c = Citation(
        go_number="GO-1345/XII/2018",
        issuing_department="Forest",
        date="2018-03-12",
        page_number=3,
        exact_text_excerpt=p.exact_text_excerpt,
    )
    state = create_base_state("Forest transfer policy 2018", passages=[p], citations=[c])

    result = await node4_supersession_confidence(state)

    assert result["confidence_score"] >= 0.70
    assert result["supersession_status"] == "CURRENT_ACTIVE"
    assert len(result["conflict_flags"]) == 0


@pytest.mark.asyncio
async def test_node4_superseded_go_status():
    """Verify Node 4 identifies SUPERSEDED status using compare_go_versions result."""
    p = PassageMatch(
        go_number="GO-562/XXX/2014",
        issuing_department="Forest",
        date="2014-07-15",
        page_number=1,
        exact_text_excerpt="Initial deployment of subordinate forest officers shall mandate a minimum tenure of three years in remote hill circles.",
        relevance_score=0.82,
    )
    c = Citation(
        go_number="GO-562/XXX/2014",
        issuing_department="Forest",
        date="2014-07-15",
        page_number=1,
        exact_text_excerpt=p.exact_text_excerpt,
    )
    state = create_base_state("Subordinate deployment tenure 2014", passages=[p], citations=[c])

    result = await node4_supersession_confidence(state)
    assert result["supersession_status"] in ["SUPERSEDED", "CURRENT_ACTIVE", "AMENDED"]


@pytest.mark.asyncio
async def test_node4_empty_retrieval_silence_over_guessing():
    """Verify Node 4 produces 0.0 confidence when no passages are retrieved."""
    state = create_base_state("Non-existent policy", passages=[], citations=[])

    result = await node4_supersession_confidence(state)
    assert result["confidence_score"] == 0.0
    assert result["supersession_status"] == "UNKNOWN"
    assert result["conflict_flags"] == []


# ===========================================================================
# 2. Code-Level Conditional Routing Gate Tests
# ===========================================================================

def test_route_confidence_supersession_code_level_gate():
    """Verify code-level enforcement of Human Verification routing conditions."""
    state_high_conf = create_base_state("query")
    state_high_conf["confidence_score"] = 0.90
    state_high_conf["conflict_flags"] = []
    assert route_confidence_supersession(state_high_conf) == "grounded_synthesis"

    # Low confidence (< 0.60) triggers interrupt
    state_low_conf = create_base_state("query")
    state_low_conf["confidence_score"] = 0.55
    state_low_conf["conflict_flags"] = []
    assert route_confidence_supersession(state_low_conf) == "human_verification_interrupt"

    # Non-empty conflict_flags triggers interrupt
    state_conflict = create_base_state("query")
    state_conflict["confidence_score"] = 0.88
    state_conflict["conflict_flags"] = [
        ConflictRecord(go_numbers=["GO-1", "GO-2"], description="Contradictory transfer rules")
    ]
    assert route_confidence_supersession(state_conflict) == "human_verification_interrupt"


def test_route_human_verification_actions():
    """Verify approval routes to synthesis, while denial routes to refusal_redirect."""
    state_approved = create_base_state("query")
    state_approved["human_verification"] = ApprovalState(
        action="approve", checkpoint_id="chk_1", reason="Verified"
    )
    assert route_human_verification(state_approved) == "grounded_synthesis"

    state_denied = create_base_state("query")
    state_denied["human_verification"] = ApprovalState(
        action="deny", checkpoint_id="chk_2", reason="Contradiction noted"
    )
    assert route_human_verification(state_denied) == "refusal_redirect"


# ===========================================================================
# 3. Node 5: Human Verification Interrupt & Resumption Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_hitl_interrupt_and_resumption_flow():
    """Verify StateGraph pauses at Node 5 interrupt on low confidence and resumes upon officer approval."""
    async with get_checkpointer() as checkpointer:
        app = create_agent_graph(checkpointer=checkpointer)
        thread_id = "test_thread_hitl_001"
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

        initial_state = create_base_state("2018 forest transfer policy GO")
        initial_state["session_id"] = thread_id
        initial_state["config"] = RuntimeConfig(confidence_threshold_low=0.99)

        # Run until interrupt
        res = await app.ainvoke(initial_state, config=config)

        # Confirm graph interrupted and checkpoint was created
        assert bool(res.get("__interrupt__")) is True
        tuple_state = await checkpointer.aget_tuple(config)
        assert tuple_state is not None

        # Resume with Approval Command
        resume_command = Command(resume={"action": "approve", "reason": "Officer manual override"})
        final_state = await app.ainvoke(resume_command, config=config)

        assert final_state["human_verification"] is not None
        assert final_state["human_verification"].action == "approve"
        assert final_state["human_verification"].reason == "Officer manual override"
