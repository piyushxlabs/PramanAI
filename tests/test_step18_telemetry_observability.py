"""Step 18 Verification Test Suite: Telemetry, Observability & Langfuse Annotations.

Verifies OpenTelemetry GenAI semantic conventions, span exports,
feedback score annotations, citation tagging, and offline fallback resilience.
"""

import pytest
from opentelemetry.trace import StatusCode

from src.telemetry.feedback_annotations import (
    record_citation_accuracy,
    record_human_verification_outcome,
    record_officer_feedback,
)
from src.telemetry.tracing import (
    get_in_memory_exporter,
    get_tracer,
    setup_telemetry,
    trace_agent_turn,
    trace_node_span,
    trace_tool_span,
)


@pytest.mark.asyncio
async def test_opentelemetry_genai_spans_and_semantic_conventions():
    """Verify OpenTelemetry GenAI semantic conventions across agent, node, and tool spans."""
    setup_telemetry()
    exporter = get_in_memory_exporter()
    exporter.clear()

    session_id = "test_otel_sess_001"

    # 1. Root invoke_agent span
    async with trace_agent_turn(
        session_id=session_id,
        officer_department="Forest",
        officer_access_scope=["Forest", "General"],
        query_text="2018 forest transfer policy",
    ) as root_span:
        assert root_span is not None

        # 2. Nested cognitive node chat span
        async with trace_node_span(
            node_name="query_interpretation",
            session_id=session_id,
            model_name="qwen2.5:7b",
            prompt_text="Parse query filters",
        ) as node_span:
            assert node_span is not None

        # 3. Nested tool retrieval span
        async with trace_tool_span(
            tool_name="search_go_corpus",
            session_id=session_id,
            tool_input={"query_text": "forest transfer", "department": "Forest"},
        ) as tool_span:
            assert tool_span is not None

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    assert "invoke_agent" in span_names
    assert "node.query_interpretation" in span_names
    assert "tool.search_go_corpus" in span_names

    # Verify root attributes
    root_exported = next(s for s in spans if s.name == "invoke_agent")
    assert root_exported.attributes["gen_ai.system"] == "shasanai"
    assert root_exported.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root_exported.attributes["session.id"] == session_id
    assert root_exported.attributes["officer.department"] == "Forest"

    # Verify tool attributes
    tool_exported = next(s for s in spans if s.name == "tool.search_go_corpus")
    assert tool_exported.attributes["gen_ai.operation.name"] == "retrieval"
    assert tool_exported.attributes["shasanai.tool_name"] == "search_go_corpus"


@pytest.mark.asyncio
async def test_telemetry_span_error_recording():
    """Verify exceptions in spans are recorded as OpenTelemetry error events."""
    setup_telemetry()
    exporter = get_in_memory_exporter()
    exporter.clear()

    session_id = "test_otel_err_sess"

    with pytest.raises(ValueError, match="Simulated cognitive error"):
        async with trace_node_span(
            node_name="grounded_synthesis",
            session_id=session_id,
        ) as span:
            raise ValueError("Simulated cognitive error")

    spans = exporter.get_finished_spans()
    err_span = next(s for s in spans if s.name == "node.grounded_synthesis")
    assert err_span.status.status_code == StatusCode.ERROR
    assert "Simulated cognitive error" in err_span.status.description


def test_feedback_annotations_payload_and_fallback():
    """Verify Section 7a Langfuse score annotations execute safely and produce valid payloads."""
    session_id = "test_feedback_sess_001"

    # 1. Officer Feedback (BOOLEAN)
    res_fb_up = record_officer_feedback(session_id=session_id, feedback_value=True, comment="Accurate")
    assert res_fb_up["name"] == "officer_feedback"
    assert res_fb_up["value"] == 1.0
    assert res_fb_up["data_type"] == "BOOLEAN"

    res_fb_down = record_officer_feedback(session_id=session_id, feedback_value=False, comment="Outdated")
    assert res_fb_down["value"] == 0.0

    # 2. Citation Accuracy (CATEGORICAL)
    res_cit_flag = record_citation_accuracy(
        session_id=session_id,
        go_number="GO-1345/XII/2018",
        page_number=4,
        is_accurate=False,
        comment="Superseded in 2021",
    )
    assert res_cit_flag["name"] == "citation_accuracy"
    assert res_cit_flag["value"] == "incorrect"
    assert res_cit_flag["data_type"] == "CATEGORICAL"
    assert "GO: GO-1345/XII/2018" in res_cit_flag["comment"]

    # 3. Human Verification Outcomes (CATEGORICAL)
    res_hitl_app = record_human_verification_outcome(
        session_id=session_id,
        outcome="approved",
        reason="Officer confirmed valid",
    )
    assert res_hitl_app["name"] == "human_verification_outcome"
    assert res_hitl_app["value"] == "approved"

    res_hitl_res = record_human_verification_outcome(
        session_id=session_id,
        outcome="approved_with_resolution",
        resolved_go_number="GO-1345/XII/2018",
    )
    assert res_hitl_res["value"] == "approved_with_resolution"
    assert "Resolved GO: GO-1345/XII/2018" in res_hitl_res["comment"]

    res_hitl_den = record_human_verification_outcome(
        session_id=session_id,
        outcome="denied",
        reason="Administrative query denied",
    )
    assert res_hitl_den["value"] == "denied"
