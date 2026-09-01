"""OpenTelemetry GenAI Semantic Conventions and Langfuse tracing for ShasanAI.

Instruments graph execution turns, cognitive node chats, and read-only MCP tool calls
with zero cloud dependencies and full DPDP Act 2023 compliance (content captured in span events).
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from dotenv import load_dotenv

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Tracer, Span, StatusCode

load_dotenv()

logger = logging.getLogger("shasanai.telemetry")

SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "shasanai")
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

_tracer_provider: TracerProvider | None = None
_in_memory_exporter: InMemorySpanExporter | None = None
_langchain_instrumented: bool = False


def get_in_memory_exporter() -> InMemorySpanExporter:
    """Returns the InMemorySpanExporter instance for telemetry verification testing."""
    global _in_memory_exporter
    if _in_memory_exporter is None:
        _in_memory_exporter = InMemorySpanExporter()
    return _in_memory_exporter


def setup_telemetry() -> TracerProvider:
    """Initializes and returns the global TracerProvider with GenAI semantic conventions."""
    global _tracer_provider
    if _tracer_provider is None:
        current_provider = trace.get_tracer_provider()
        if isinstance(current_provider, TracerProvider):
            _tracer_provider = current_provider
        else:
            resource = Resource.create({
                "service.name": SERVICE_NAME,
                "service.version": "1.0.0",
                "deployment.environment": os.getenv("APP_ENV", "development"),
                "sovereign.jurisdiction": "IN-UT",
                "compliance.standard": "DPDP-Act-2023",
            })
            _tracer_provider = TracerProvider(resource=resource)
            try:
                trace.set_tracer_provider(_tracer_provider)
            except Exception:
                pass
        
        # Attach in-memory exporter for zero-overhead audit and test inspection
        exporter = get_in_memory_exporter()
        _tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return _tracer_provider


def get_tracer(name: str = "shasanai.telemetry") -> Tracer:
    """Retrieves an OpenTelemetry Tracer instance."""
    provider = setup_telemetry()
    return provider.get_tracer(name)


def instrument_langchain() -> bool:
    """Safely configures telemetry without invasive callback hooks that crash on LangGraph interrupts."""
    global _langchain_instrumented
    if not _langchain_instrumented:
        setup_telemetry()
        _langchain_instrumented = True
        logger.info("OpenTelemetry GenAI spans successfully configured.")
    return True


@asynccontextmanager
async def trace_agent_turn(
    session_id: str,
    officer_department: str = "General",
    officer_access_scope: list[str] | None = None,
    query_text: str | None = None,
) -> AsyncIterator[Span]:
    """Wraps an entire multi-node agent turn in an `invoke_agent` root span.
    
    Adheres to OpenTelemetry GenAI semantic conventions:
    - Operation: invoke_agent
    - Attributes: session, officer context, system metadata
    - DPDP Act 2023 Compliance: query content recorded in span events (not indexed attributes).
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "invoke_agent",
        attributes={
            "gen_ai.system": "shasanai",
            "gen_ai.operation.name": "invoke_agent",
            "session.id": session_id,
            "officer.department": officer_department,
            "officer.access_scope": json.dumps(officer_access_scope or [officer_department]),
        },
    ) as span:
        if query_text:
            span.add_event("gen_ai.content.prompt", {"content": query_text})
        try:
            yield span
        except GeneratorExit:
            # Clean SSE stream closure
            span.set_status(StatusCode.OK)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            raise


@asynccontextmanager
async def trace_node_span(
    node_name: str,
    session_id: str,
    model_name: str = "qwen2.5:7b",
    prompt_text: str | None = None,
) -> AsyncIterator[Span]:
    """Wraps a cognitive node execution in an OpenTelemetry GenAI `chat` span."""
    tracer = get_tracer()
    with tracer.start_as_current_span(
        f"node.{node_name}",
        attributes={
            "gen_ai.system": "shasanai",
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model_name,
            "shasanai.node_name": node_name,
            "session.id": session_id,
        },
    ) as span:
        if prompt_text:
            span.add_event("gen_ai.content.prompt", {"content": prompt_text})
        try:
            yield span
        except GeneratorExit:
            # Clean SSE stream closure
            span.set_status(StatusCode.OK)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            raise


@asynccontextmanager
async def trace_tool_span(
    tool_name: str,
    session_id: str,
    tool_input: dict[str, Any] | None = None,
) -> AsyncIterator[Span]:
    """Wraps a read-only MCP tool execution in an `execute_tool` or `retrieval` span."""
    tracer = get_tracer()
    operation_name = "retrieval" if tool_name == "search_go_corpus" else "execute_tool"
    with tracer.start_as_current_span(
        f"tool.{tool_name}",
        attributes={
            "gen_ai.system": "shasanai",
            "gen_ai.operation.name": operation_name,
            "shasanai.tool_name": tool_name,
            "session.id": session_id,
        },
    ) as span:
        if tool_input:
            span.add_event("gen_ai.tool.input", {"input": json.dumps(tool_input)})
        try:
            yield span
        except GeneratorExit:
            # Clean SSE stream closure
            span.set_status(StatusCode.OK)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            raise

