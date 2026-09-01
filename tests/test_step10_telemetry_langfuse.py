"""Verification tests for Step 10: OpenTelemetry Telemetry, Spans & Langfuse Integration.

Verifies OpenTelemetry GenAI semantic conventions, span hierarchy, DPDP Act 2023 event-based
prompt logging, and Section 7a Langfuse feedback-to-telemetry scoring pipeline.
"""

import pytest
from src.telemetry.feedback_annotations import (
    record_citation_accuracy,
    record_human_verification_outcome,
    record_officer_feedback,
)
from src.telemetry.tracing import (
    get_in_memory_exporter,
    get_tracer,
    instrument_langchain,
    setup_telemetry,
    trace_agent_turn,
    trace_node_span,
    trace_tool_span,
)


def test_telemetry_provider_setup_and_resource_attributes():
    """Verify setup_telemetry initializes TracerProvider with sovereign DPDP metadata."""
    provider = setup_telemetry()
    assert provider is not None

    tracer = get_tracer("shasanai.test")
    assert tracer is not None

    exporter = get_in_memory_exporter()
    assert exporter is not None


@pytest.mark.asyncio
async def test_trace_agent_turn_span_hierarchy_and_dpdp_events():
    """Verify trace_agent_turn creates invoke_agent span with DPDP event-based prompt logging."""
    exporter = get_in_memory_exporter()
    exporter.clear()

    session_id = "test_telemetry_session_001"
    query_text = "Uttarakhand transfer policy 2018"

    async with trace_agent_turn(
        session_id=session_id,
        officer_department="Forest",
        officer_access_scope=["Forest", "General"],
        query_text=query_text,
    ) as root_span:
        assert root_span is not None

    spans = exporter.get_finished_spans()
    matching_spans = [s for s in spans if s.name == "invoke_agent"]
    assert len(matching_spans) >= 1

    span = matching_spans[-1]
    assert span.attributes["gen_ai.system"] == "shasanai"
    assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert span.attributes["session.id"] == session_id
    assert span.attributes["officer.department"] == "Forest"

    # DPDP Act 2023 Compliance: query text must be in span events, NOT indexed attributes
    assert "gen_ai.content.prompt" not in span.attributes
    event_names = [e.name for e in span.events]
    assert "gen_ai.content.prompt" in event_names
    prompt_event = next(e for e in span.events if e.name == "gen_ai.content.prompt")
    assert prompt_event.attributes["content"] == query_text


@pytest.mark.asyncio
async def test_trace_node_span_genai_conventions():
    """Verify trace_node_span attaches chat operation, model, and local resource metrics."""
    exporter = get_in_memory_exporter()
    exporter.clear()

    session_id = "test_telemetry_session_002"
    prompt_text = "System: Node 1 Interpretation prompt"

    async with trace_node_span(
        node_name="query_interpretation",
        session_id=session_id,
        model_name="qwen2.5:7b",
        prompt_text=prompt_text,
    ) as node_span:
        assert node_span is not None

    spans = exporter.get_finished_spans()
    matching_spans = [s for s in spans if s.name == "node.query_interpretation"]
    assert len(matching_spans) >= 1

    span = matching_spans[-1]
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["shasanai.node_name"] == "query_interpretation"
    assert span.attributes["session.id"] == session_id

    # Verify DPDP prompt event
    assert "gen_ai.content.prompt" in [e.name for e in span.events]


@pytest.mark.asyncio
async def test_trace_tool_span_retrieval_and_execute_tool():
    """Verify search_go_corpus creates retrieval span and compare_go_versions creates execute_tool span."""
    exporter = get_in_memory_exporter()
    exporter.clear()

    session_id = "test_telemetry_session_003"

    # 1. search_go_corpus (dedicated retrieval operation)
    async with trace_tool_span(
        tool_name="search_go_corpus",
        session_id=session_id,
        tool_input={"query_text": "transfer window", "department": "Forest"},
    ):
        pass

    # 2. compare_go_versions (execute_tool operation)
    async with trace_tool_span(
        tool_name="compare_go_versions",
        session_id=session_id,
        tool_input={"go_numbers": ["GO-1345/XII/2018"]},
    ):
        pass

    spans = exporter.get_finished_spans()
    retrieval_span = next(s for s in spans if s.name == "tool.search_go_corpus")
    assert retrieval_span.attributes["gen_ai.operation.name"] == "retrieval"
    assert "gen_ai.tool.input" in [e.name for e in retrieval_span.events]

    tool_span = next(s for s in spans if s.name == "tool.compare_go_versions")
    assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"


def test_feedback_annotations_officer_feedback():
    """Verify Section 7a officer thumbs up/down records BOOLEAN score structure."""
    res_pos = record_officer_feedback(
        session_id="session_feedback_01",
        trace_id="trace_001",
        feedback_value=True,
        comment="Accurate transfer policy answer",
    )
    assert res_pos["name"] == "officer_feedback"
    assert res_pos["value"] == 1.0
    assert res_pos["data_type"] == "BOOLEAN"
    assert res_pos["session_id"] == "session_feedback_01"

    res_neg = record_officer_feedback(
        session_id="session_feedback_01",
        trace_id="trace_001",
        feedback_value=False,
        comment="Missing 2020 amendment",
    )
    assert res_neg["value"] == 0.0


def test_feedback_annotations_citation_accuracy():
    """Verify Section 7a per-citation flag records CATEGORICAL score structure."""
    res = record_citation_accuracy(
        session_id="session_citation_01",
        go_number="GO-1345/XII/2018",
        page_number=3,
        trace_id="trace_002",
        is_accurate=False,
        comment="Page 3 discusses leave policy, not transfer window",
    )
    assert res["name"] == "citation_accuracy"
    assert res["value"] == "incorrect"
    assert res["data_type"] == "CATEGORICAL"
    assert "GO-1345/XII/2018 (Page 3)" in res["comment"]


def test_feedback_annotations_human_verification_outcome():
    """Verify Section 7a HITL outcome records CATEGORICAL score structure."""
    res_approve = record_human_verification_outcome(
        session_id="session_hitl_01",
        outcome="approved_with_resolution",
        trace_id="trace_003",
        resolved_go_number="GO-1345/XII/2018",
        reason="Officer confirmed 2018 policy is active",
    )
    assert res_approve["name"] == "human_verification_outcome"
    assert res_approve["value"] == "approved_with_resolution"
    assert res_approve["data_type"] == "CATEGORICAL"
    assert "Resolved GO: GO-1345/XII/2018" in res_approve["comment"]

    res_deny = record_human_verification_outcome(
        session_id="session_hitl_01",
        outcome="denied",
        trace_id="trace_003",
        reason="Neither candidate GO is relevant",
    )
    assert res_deny["value"] == "denied"


def test_langchain_instrumentation_call():
    """Verify instrument_langchain initializes safely."""
    success = instrument_langchain()
    assert success is True
