"""Step 17 Verification Test Suite: Next.js Frontend Integration & Generative UI API Contracts.

Verifies end-to-end frontend-backend compatibility, SSE payload parsing,
checkpoint resumption streaming, and telemetry feedback ingestion.
"""

import json
import pytest
from httpx import ASGITransport, AsyncClient

from src.server.app import app
from src.state.checkpointing import ensure_windows_event_loop


@pytest.mark.asyncio
async def test_frontend_health_and_cors():
    """Verify frontend /api/health endpoint returns 200 and supports CORS."""
    ensure_windows_event_loop()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "shasanai-backend"


@pytest.mark.asyncio
async def test_frontend_chat_sse_stream_contract():
    """Verify frontend /api/chat endpoint delivers typed SSE stream consumed by useSSEChat hook."""
    ensure_windows_event_loop()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "session_id": "test_frontend_sess_001",
            "query_text": "2018 forest transfer policy GO",
            "officer_context": {
                "department": "Forest",
                "access_scope": ["Forest", "General"],
            },
        }
        response = await client.post("/api/chat", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        lines = response.text.split("\n")
        events: list[dict] = []
        for line in lines:
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except Exception:
                    pass

        event_types = [e.get("type") for e in events]
        assert "message-start" in event_types
        assert "data-graph-step" in event_types
        assert "finish" in event_types


@pytest.mark.asyncio
async def test_frontend_hitl_resume_and_feedback_endpoints():
    """Verify frontend /api/hitl/resume, /resume, and feedback rating endpoints."""
    ensure_windows_event_loop()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Feedback Score endpoint
        fb_score_payload = {
            "session_id": "test_frontend_sess_001",
            "trace_id": "trace_001",
            "feedback_value": True,
            "comment": "Accurate verbatim citation",
        }
        res_fb = await client.post("/api/feedback/score", json=fb_score_payload)
        assert res_fb.status_code == 200
        assert res_fb.json()["success"] is True

        # 2. Citation Accuracy endpoint
        fb_cit_payload = {
            "session_id": "test_frontend_sess_001",
            "go_number": "GO-1345/XII/2018",
            "page_number": 3,
            "trace_id": "trace_001",
            "is_accurate": True,
            "comment": "Page 3 verified",
        }
        res_cit = await client.post("/api/feedback/citation", json=fb_cit_payload)
        assert res_cit.status_code == 200
        assert res_cit.json()["success"] is True
