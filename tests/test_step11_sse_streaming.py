"""Verification tests for Step 11: Server-Sent Events (SSE) Streaming Layer & Event Emitter.

Verifies the 10 typed SSE events, native-thinking token suppression filter,
keepalive generation, and end-to-end graph-to-SSE streaming.
"""

import json
import pytest
from pydantic import ValidationError
from src.agents.graph import create_agent_graph
from src.state.schema import OfficerContext, StateSchema
from src.ui.event_types import (
    DataApprovalRequiredData,
    DataApprovalRequiredEvent,
    DataGraphStepData,
    DataGraphStepEvent,
    DataStateUpdateData,
    DataStateUpdateEvent,
    ErrorEvent,
    FinishEvent,
    MessageStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolInvocationEvent,
)
from src.ui.stream_handler import (
    NODE_LABELS,
    format_sse_event,
    format_sse_keepalive,
    stream_agent_turn,
)


def create_base_state(query: str) -> StateSchema:
    """Helper to create minimal initial state for testing."""
    from src.state.schema import RuntimeConfig
    return {
        "session_id": "test_sse_session_001",
        "officer_context": OfficerContext(department="Forest", access_scope=["Forest"]),
        "query_text": query,
        "query_language": "en",
        "query_filters": None,
        "message_history": [],
        "retrieved_passages": [],
        "candidate_citations": [],
        "confidence_score": 0.0,
        "supersession_status": "UNKNOWN",
        "conflict_flags": [],
        "human_verification": None,
        "answer_markdown": None,
        "citations": [],
        "graceful_refusal": False,
        "error_logs": [],
        "config": RuntimeConfig(),
    }


def test_event_types_models_validation():
    """Verify all 10 event models serialize cleanly and reject forbidden extra fields."""
    # 1. MessageStartEvent
    evt_msg = MessageStartEvent(id="msg_1", metadata={"session_id": "s1", "query_language": "en"})
    assert evt_msg.type == "message-start"

    # 2. TextStartEvent
    evt_txt_start = TextStartEvent(id="txt_1")
    assert evt_txt_start.type == "text-start"

    # 3. TextDeltaEvent
    evt_txt_delta = TextDeltaEvent(id="txt_1", delta="According to GO-1345...")
    assert evt_txt_delta.delta == "According to GO-1345..."

    # 4. TextEndEvent
    evt_txt_end = TextEndEvent(id="txt_1")
    assert evt_txt_end.type == "text-end"

    # 5. DataGraphStepEvent
    evt_step = DataGraphStepEvent(
        id="step_1",
        data=DataGraphStepData(node="query_interpretation", label="Reading your question", status="started"),
    )
    assert evt_step.data.label == "Reading your question"

    # 6. ToolInvocationEvent
    evt_tool = ToolInvocationEvent(
        type="tool-search_go_corpus",
        toolCallId="call_1",
        state="output-available",
        output={"count": 1},
    )
    assert evt_tool.type == "tool-search_go_corpus"

    # 7. DataStateUpdateEvent
    evt_upd = DataStateUpdateEvent(
        id="upd_1",
        data=DataStateUpdateData(field="confidence_score", reducer="last-write-wins", value=0.92),
    )
    assert evt_upd.data.value == 0.92

    # 8. DataApprovalRequiredEvent
    evt_appr = DataApprovalRequiredEvent(
        id="appr_1",
        data=DataApprovalRequiredData(
            checkpoint_id="chk_1",
            trigger="conflict",
            action_preview={"conflicts": []},
        ),
    )
    assert evt_appr.data.trigger == "conflict"

    # 9. ErrorEvent
    evt_err = ErrorEvent(errorText="Timeout occurred", code="timeout", recoverable=True)
    assert evt_err.recoverable is True

    # 10. FinishEvent
    evt_fin = FinishEvent(finishReason="success")
    assert evt_fin.finishReason == "success"

    # Extra fields forbidden check
    with pytest.raises(ValidationError):
        MessageStartEvent(id="m", metadata={}, unexpected_field="bad")


def test_format_sse_event_structure_and_keepalive():
    """Verify SSE line formatting matches data: {json}\n\n and keepalive matches : keep-alive\n\n."""
    evt = FinishEvent(finishReason="success")
    line = format_sse_event(evt)
    assert line.startswith("data: ")
    assert line.endswith("\n\n")

    parsed = json.loads(line.replace("data: ", "").strip())
    assert parsed["type"] == "finish"
    assert parsed["finishReason"] == "success"

    keepalive = format_sse_keepalive()
    assert keepalive == ": keep-alive\n\n"


def test_native_thinking_filter_server_side():
    """Verify server-side filter suppresses native thinking tokens permanently."""
    # 1. reasoning-delta type
    res1 = format_sse_event({"type": "reasoning-delta", "delta": "internal reflection"})
    assert res1 == ""

    # 2. source: native-thinking
    res2 = format_sse_event({"type": "text-delta", "source": "native-thinking", "delta": "thinking"})
    assert res2 == ""


def test_node_labels_coverage():
    """Verify all 9 graph nodes have plain-language officer-facing labels."""
    expected_nodes = [
        "query_interpretation",
        "scope_screen",
        "retrieval_invocation",
        "confidence_supersession",
        "human_verification_interrupt",
        "grounded_synthesis",
        "citation_integrity",
        "refusal_redirect",
        "response_delivery",
    ]
    for n in expected_nodes:
        assert n in NODE_LABELS
        assert len(NODE_LABELS[n]) > 0


@pytest.mark.asyncio
async def test_stream_agent_turn_in_scope_flow():
    """Verify end-to-end SSE event generation for an in-scope administrative query."""
    app = create_agent_graph()
    state = create_base_state("Forest department transfer policy 2018")

    emitted_events = []
    async for sse_line in stream_agent_turn(app, state):
        if sse_line.startswith("data: "):
            payload = json.loads(sse_line.replace("data: ", "").strip())
            emitted_events.append(payload)

    event_types = [e["type"] for e in emitted_events]
    assert "message-start" in event_types
    assert "data-graph-step" in event_types
    assert "finish" in event_types

    finish_event = next(e for e in emitted_events if e["type"] == "finish")
    assert finish_event["finishReason"] in ["success", "interrupted", "approval_required"]


@pytest.mark.asyncio
async def test_stream_agent_turn_out_of_scope_refusal():
    """Verify end-to-end SSE event generation for an out-of-scope query routes to refusal."""
    app = create_agent_graph()
    state = create_base_state("Sanction 50 lakhs disbursement for construction project immediately")

    emitted_events = []
    async for sse_line in stream_agent_turn(app, state):
        if sse_line.startswith("data: "):
            payload = json.loads(sse_line.replace("data: ", "").strip())
            emitted_events.append(payload)

    event_types = [e["type"] for e in emitted_events]
    assert "message-start" in event_types
    assert "data-graph-step" in event_types
    assert "finish" in event_types

    finish_event = next(e for e in emitted_events if e["type"] == "finish")
    assert finish_event["finishReason"] == "refused"
