"""Verification tests for Step 12: HITL Graph-Resumption Endpoints & Backend API Server (FastAPI).

Verifies FastAPI application endpoints: /health, /api/chat SSE streaming,
/api/hitl/resume graph checkpoint resumption, and feedback scoring endpoints.
"""

import uuid
import pytest
from httpx import ASGITransport, AsyncClient
from src.agents.graph import create_agent_graph
from src.server.app import app
from src.state.checkpointing import (
    ensure_windows_event_loop,
    get_checkpointer,
    setup_checkpoint_tables,
)
from src.state.schema import OfficerContext, StateSchema


@pytest.mark.asyncio
async def test_health_endpoints():
    """Verify /health and /api/health return 200 with service metadata."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp1 = await client.get("/health")
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["status"] == "healthy"
        assert data1["service"] == "shasanai-backend"
        assert data1["models"]["inference"] == "qwen2.5:7b"

        resp2 = await client.get("/api/health")
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_feedback_score_endpoint():
    """Verify /api/feedback/score accepts thumbs up/down and returns 200."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "session_id": "test_server_session_01",
            "feedback_value": True,
            "comment": "Accurate response with valid citations",
        }
        resp = await client.post("/api/feedback/score", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["name"] == "officer_feedback"
        assert data["data"]["value"] == 1.0


@pytest.mark.asyncio
async def test_feedback_citation_endpoint():
    """Verify /api/feedback/citation accepts citation accuracy flag and returns 200."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "session_id": "test_server_session_02",
            "go_number": "GO-1345/XII/2018",
            "page_number": 3,
            "is_accurate": False,
            "comment": "Provisions apply to 2019 onward",
        }
        resp = await client.post("/api/feedback/citation", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"]["name"] == "citation_accuracy"
        assert data["data"]["value"] == "incorrect"


@pytest.mark.asyncio
async def test_chat_stream_endpoint_sse():
    """Verify /api/chat returns text/event-stream and streams typed SSE events."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "session_id": f"test_stream_{uuid.uuid4().hex[:8]}",
            "query_text": "Uttarakhand Forest department transfer policy 2018",
            "officer_context": {
                "department": "Forest",
                "access_scope": ["Forest", "General"],
            },
        }
        resp = await client.post("/api/chat", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        body_text = resp.text
        assert "data: " in body_text
        assert "message-start" in body_text
        assert "finish" in body_text


@pytest.mark.asyncio
async def test_hitl_resumption_endpoint_flow():
    """Verify /api/hitl/resume resumes paused execution from PostgreSQL checkpoint."""
    ensure_windows_event_loop()
    await setup_checkpoint_tables()
    session_id = f"test_hitl_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": session_id, "checkpoint_ns": ""}}

    async with get_checkpointer() as saver:
        app_graph = create_agent_graph(checkpointer=saver)

        # 1. Query within Forest access scope that returns 0 matches in corpus,
        # yielding low confidence (0.05 < 0.60) and triggering Node 5 Human Verification Interrupt
        initial_state: StateSchema = {
            "session_id": session_id,
            "officer_context": OfficerContext(department="Forest", access_scope=["Forest"]),
            "query_text": "circular regarding unrecorded nursery subsidies in forest department 2021",
            "query_language": "en",
            "query_filters": None,
            "turn_history": [],
            "route": "standard_retrieval",
            "graceful_refusal": False,
            "candidate_citations": [],
            "retrieved_passages": [],
            "confidence_score": 0.0,
            "supersession_status": "UNKNOWN",
            "conflict_flags": [],
            "answer_markdown": None,
            "citations": [],
            "human_verification": None,
            "error_logs": [],
        }

        # Run graph until pause at Node 5
        res = await app_graph.ainvoke(initial_state, config=config)
        assert bool(res.get("__interrupt__")) is True

    # 2. Call /api/hitl/resume to approve and resume
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resume_payload = {
            "action": "approve",
            "checkpoint_id": session_id,
            "modified_inputs": None,
            "reason": "Officer manual override to proceed",
        }
        resp = await client.post("/api/hitl/resume", json=resume_payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        body = resp.text
        assert "data: " in body
        assert "finish" in body

    # Verify checkpoint is now completed (next is empty)
    async with get_checkpointer() as saver:
        app_graph = create_agent_graph(checkpointer=saver)
        final_snap = await app_graph.aget_state(config)
        assert len(final_snap.next) == 0
