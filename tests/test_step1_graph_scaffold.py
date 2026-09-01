"""Verification tests for Step 1: LangGraph StateGraph Scaffold & 17-field StateSchema."""

import pytest
from src.agents.graph import build_scaffold_graph, create_agent_graph
from src.state.schema import (
    ApprovalState,
    Citation,
    ConflictRecord,
    Message,
    OfficerContext,
    PassageMatch,
    QueryFilters,
    RuntimeConfig,
    StateSchema,
)
from src.state.reducers import (
    StateValidationError,
    append_only_reducer,
    immutable_reducer,
    merge_by_citation_key_reducer,
)


def test_state_schema_17_fields_presence():
    """Verify that StateSchema declares all 17 required state fields."""
    expected_fields = {
        "session_id",
        "officer_context",
        "query_text",
        "query_language",
        "query_filters",
        "message_history",
        "retrieved_passages",
        "candidate_citations",
        "confidence_score",
        "supersession_status",
        "conflict_flags",
        "human_verification",
        "answer_markdown",
        "citations",
        "graceful_refusal",
        "error_logs",
        "config",
    }
    assert set(StateSchema.__annotations__.keys()) == expected_fields
    assert len(expected_fields) == 17


def test_graph_has_exactly_9_nodes():
    """Verify that graph contains all 9 nodes specified in AGENT_ORCHESTRATION_BLUEPRINT.md."""
    workflow = build_scaffold_graph()
    graph = workflow.compile()
    graph_structure = graph.get_graph()

    expected_nodes = {
        "__start__",
        "__end__",
        "query_interpretation",
        "scope_screen",
        "retrieval_invocation",
        "confidence_supersession",
        "human_verification_interrupt",
        "grounded_synthesis",
        "citation_integrity",
        "refusal_redirect",
        "response_delivery",
    }
    actual_nodes = set(graph_structure.nodes.keys())
    assert actual_nodes == expected_nodes

    # Core non-framework nodes count
    core_nodes = actual_nodes - {"__start__", "__end__"}
    assert len(core_nodes) == 9


def test_immutable_reducer_behavior():
    """Verify immutable-after-init reducer enforces immutability."""
    # Initialization
    val = immutable_reducer(None, "session_123")
    assert val == "session_123"

    # Same value re-write is allowed
    val2 = immutable_reducer("session_123", "session_123")
    assert val2 == "session_123"

    # Mutation raises StateValidationError
    with pytest.raises(StateValidationError):
        immutable_reducer("session_123", "session_456")


def test_append_only_reducer_behavior():
    """Verify append-only reducer preserves history without dropping records."""
    msg1 = Message(role="user", content="hello")
    msg2 = Message(role="assistant", content="namaste")
    
    history = append_only_reducer([], [msg1])
    assert len(history) == 1
    assert history[0].content == "hello"

    history = append_only_reducer(history, [msg2])
    assert len(history) == 2
    assert history[1].content == "namaste"


def test_merge_by_citation_key_reducer():
    """Verify merge-by-key reducer prevents duplicate citations on (go_number, page_number)."""
    c1 = Citation(
        go_number="GO-100",
        issuing_department="Forest",
        date="2020-01-01",
        page_number=2,
        exact_text_excerpt="Excerpt 1",
    )
    c2 = Citation(
        go_number="GO-100",
        issuing_department="Forest",
        date="2020-01-01",
        page_number=2,
        exact_text_excerpt="Excerpt 1 updated",
    )
    c3 = Citation(
        go_number="GO-200",
        issuing_department="Finance",
        date="2021-05-10",
        page_number=1,
        exact_text_excerpt="Excerpt 2",
    )

    merged = merge_by_citation_key_reducer([c1], [c2, c3])
    assert len(merged) == 2
    assert merged[0].go_number == "GO-100"
    assert merged[0].exact_text_excerpt == "Excerpt 1 updated"
    assert merged[1].go_number == "GO-200"
