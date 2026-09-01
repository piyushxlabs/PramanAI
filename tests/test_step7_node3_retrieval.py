"""Verification tests for Step 7: Node 3 (Retrieval Invocation & Citation Ingestion)."""

import pytest
from src.agents.graph import create_agent_graph
from src.agents.nodes.node3_retrieval_invocation import node3_retrieval_invocation
from src.state.reducers import merge_by_citation_key_reducer, replace_on_new_turn_reducer
from src.state.schema import Citation, Message, OfficerContext, PassageMatch, QueryFilters, RuntimeConfig, StateSchema


def create_base_state(query: str, department: str = "Forest", history: list[Message] | None = None) -> StateSchema:
    """Helper to create minimal valid state for testing Node 3."""
    return {
        "session_id": "test_session_step7",
        "officer_context": OfficerContext(department=department, access_scope=[department]),
        "query_text": query,
        "query_language": "en",
        "query_filters": QueryFilters(department=department),
        "message_history": history or [],
        "retrieved_passages": [],
        "candidate_citations": [],
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
# 1. Node 3: Retrieval Invocation Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_node3_retrieval_success():
    """Verify Node 3 executes search_go_corpus and ingests passages and citations."""
    state = create_base_state("Forest transfer policy 2018", department="Forest")
    state["query_filters"] = QueryFilters(department="Forest", year_range=[2018, 2018])

    result = await node3_retrieval_invocation(state)

    assert "retrieved_passages" in result
    assert "candidate_citations" in result
    passages: list[PassageMatch] = result["retrieved_passages"]
    citations: list[Citation] = result["candidate_citations"]

    assert len(passages) > 0
    assert len(citations) == len(passages)
    assert len(citations[0].go_number) > 0
    assert citations[0].page_number >= 1
    assert citations[0].issuing_department is not None


@pytest.mark.asyncio
async def test_node3_candidate_citations_merge_by_key_deduplication():
    """Verify merge-by-key reducer behavior across multiple retrieval cycles."""
    c1 = Citation(
        go_number="GO-1345/XII/2018",
        issuing_department="Forest",
        date="2018-03-12",
        page_number=3,
        exact_text_excerpt="Excerpt turn 1",
    )
    c2 = Citation(
        go_number="GO-1345/XII/2018",
        issuing_department="Forest",
        date="2018-03-12",
        page_number=3,
        exact_text_excerpt="Excerpt turn 2 updated",
    )
    c3 = Citation(
        go_number="GO-562/XXX/2014",
        issuing_department="Forest",
        date="2014-07-15",
        page_number=1,
        exact_text_excerpt="New citation turn 2",
    )

    merged = merge_by_citation_key_reducer([c1], [c2, c3])
    assert len(merged) == 2
    # Verify update on matching key
    c1_updated = [c for c in merged if c.go_number == "GO-1345/XII/2018"][0]
    assert c1_updated.exact_text_excerpt == "Excerpt turn 2 updated"


@pytest.mark.asyncio
async def test_node3_retrieved_passages_replace_on_new_turn():
    """Verify replace-on-new-turn reducer replaces prior turn passages."""
    old_passage = PassageMatch(
        go_number="GO-OLD",
        issuing_department="Forest",
        date="2010-01-01",
        page_number=1,
        exact_text_excerpt="Old",
        relevance_score=0.5,
    )
    new_passage = PassageMatch(
        go_number="GO-NEW",
        issuing_department="Forest",
        date="2020-01-01",
        page_number=1,
        exact_text_excerpt="New",
        relevance_score=0.9,
    )

    replaced = replace_on_new_turn_reducer([old_passage], [new_passage])
    assert len(replaced) == 1
    assert replaced[0].go_number == "GO-NEW"


@pytest.mark.asyncio
async def test_node3_silence_over_guessing_empty_retrieval():
    """Verify silence-over-guessing policy: empty retrieval returns empty lists without hallucinating."""
    state = create_base_state("Non-existent policy 1960", department="Forest")
    state["query_filters"] = QueryFilters(department="Forest", year_range=[1960, 1965])

    result = await node3_retrieval_invocation(state)
    assert result["retrieved_passages"] == []
    assert result["candidate_citations"] == []


# ===========================================================================
# 2. Subgraph Execution Flow Tests (Node 1 -> Node 2 -> Node 3)
# ===========================================================================

@pytest.mark.asyncio
async def test_graph_execution_in_scope_retrieves_citations():
    """Verify end-to-end flow of in-scope query executing through Node 1 -> Node 2 -> Node 3."""
    app = create_agent_graph()
    initial_state = create_base_state(
        "2018 mein forest department ke transfer policy ke baare mein kya GO hai?",
        department="Forest",
    )

    final_state = await app.ainvoke(initial_state)

    assert final_state["graceful_refusal"] is False
    assert len(final_state["candidate_citations"]) > 0
    assert len(final_state["candidate_citations"][0].go_number) > 0
