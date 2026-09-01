"""Telemetry and observability package for ShasanAI.

Exports OpenTelemetry GenAI semantic convention tracers, Langfuse scoring pipelines,
and LangChain instrumentation.
"""

from src.telemetry.feedback_annotations import (
    get_langfuse_client,
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

__all__ = [
    "get_in_memory_exporter",
    "get_langfuse_client",
    "get_tracer",
    "instrument_langchain",
    "record_citation_accuracy",
    "record_human_verification_outcome",
    "record_officer_feedback",
    "setup_telemetry",
    "trace_agent_turn",
    "trace_node_span",
    "trace_tool_span",
]
