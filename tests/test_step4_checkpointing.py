"""Verification tests for Step 4: PostgreSQL Checkpointing Backend (AsyncPostgresSaver)."""

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from src.agents.graph import create_agent_graph
from src.state.checkpointing import (
    ensure_windows_event_loop,
    get_checkpointer,
    setup_postgres_checkpoint_tables,
)
from src.state.schema import OfficerContext, QueryFilters, RuntimeConfig


@pytest.fixture(scope="session", autouse=True)
def setup_windows_loop():
    """Ensure Windows SelectorEventLoop is active for pytest-asyncio session."""
    ensure_windows_event_loop()


@pytest.mark.asyncio
async def test_setup_checkpoint_tables():
    """Verify PostgreSQL checkpointing tables initialization (.setup())."""
    await setup_postgres_checkpoint_tables()
    # Setup completes without error
    assert True


@pytest.mark.asyncio
async def test_checkpoint_write_read_roundtrip():
    """Verify writing and reading a checkpoint state round-trip with real PostgreSQL instance."""
    thread_id = "test_session_roundtrip_001"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    test_state = {
        "session_id": thread_id,
        "officer_context": OfficerContext(department="Forest", access_scope=["Forest"]),
        "query_text": "2018 transfer policy GO",
        "query_language": "hi",
        "query_filters": QueryFilters(department="Forest", year_range=[2018, 2018]),
        "message_history": [],
        "retrieved_passages": [],
        "candidate_citations": [],
        "confidence_score": 0.88,
        "supersession_status": "CURRENT_ACTIVE",
        "conflict_flags": [],
        "human_verification": None,
        "answer_markdown": None,
        "citations": [],
        "graceful_refusal": False,
        "error_logs": [],
        "config": RuntimeConfig(),
    }

    async with get_checkpointer() as checkpointer:
        # Write checkpoint using standard empty_checkpoint factory
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = test_state
        checkpoint["channel_versions"] = {k: 1 for k in test_state}

        saved_config = await checkpointer.aput(
            config, checkpoint, metadata={"source": "test"}, new_versions={k: 1 for k in test_state}
        )
        assert saved_config is not None

        # Read back checkpoint
        retrieved_tuple = await checkpointer.aget_tuple(config)
        assert retrieved_tuple is not None
        saved_state = retrieved_tuple.checkpoint["channel_values"]

        assert saved_state["session_id"] == thread_id
        assert saved_state["confidence_score"] == 0.88
        assert saved_state["supersession_status"] == "CURRENT_ACTIVE"
        assert saved_state["query_language"] == "hi"


@pytest.mark.asyncio
async def test_graph_compilation_with_postgres_saver():
    """Verify LangGraph StateGraph compiles and attaches to AsyncPostgresSaver."""
    async with get_checkpointer() as checkpointer:
        app = create_agent_graph(checkpointer=checkpointer)
        assert app.checkpointer is not None
        assert app.checkpointer == checkpointer
