"""HITL Graph Resumption Handler for ShasanAI.

Manages durable human-in-the-loop resumption from PostgreSQL checkpoints upon officer approval,
approval-with-edit, or denial, streaming real-time SSE progress events.
"""

import logging
from typing import Any, AsyncIterator, Literal, Optional
from langgraph.types import Command

from src.agents.graph import create_agent_graph
from src.state.checkpointing import ensure_windows_event_loop, get_checkpointer
from src.telemetry.feedback_annotations import record_human_verification_outcome
from src.ui.stream_handler import stream_agent_turn

logger = logging.getLogger("shasanai.hitl_resumption")


async def resume_hitl_stream(
    checkpoint_id: str,
    action: Literal["approve", "deny"],
    resolved_go_number: Optional[str] = None,
    reason: Optional[str] = None,
) -> AsyncIterator[str]:
    """Resumes a paused LangGraph execution from a PostgreSQL checkpoint and yields SSE events.

    Wraps the entire resume pipeline in GeneratorExit-safe error handling so that SSE
    stream teardown never leaks into the OTel ContextVar stack.
    """
    ensure_windows_event_loop()

    outcome_val = (
        "approved_with_resolution"
        if action == "approve" and resolved_go_number
        else ("approved" if action == "approve" else "denied")
    )

    # Record Langfuse Human Verification Telemetry
    try:
        record_human_verification_outcome(
            session_id=checkpoint_id,
            outcome=outcome_val,  # type: ignore[arg-type]
            resolved_go_number=resolved_go_number,
            reason=reason,
        )
    except Exception as e:
        logger.warning(f"Failed to record telemetry for HITL resumption: {e}")

    # Build LangGraph resume command
    resume_payload: dict[str, Any] = {
        "action": action,
        "resolved_go_number": resolved_go_number,
        "reason": reason,
    }
    resume_command = Command(resume=resume_payload)

    try:
        async with get_checkpointer() as saver:
            app_graph = create_agent_graph(checkpointer=saver)
            config = {"configurable": {"thread_id": checkpoint_id, "checkpoint_ns": ""}, "recursion_limit": 15}
            async for event in stream_agent_turn(app_graph, resume_command, config=config):
                yield event
    except GeneratorExit:
        # SSE client disconnected cleanly during HITL resume — suppress silently.
        logger.debug("HITL resume stream closed by client (GeneratorExit).")
    except Exception as exc:
        logger.error(f"HITL resume stream error for checkpoint {checkpoint_id}: {exc!s}")
        # Emit a recoverable error event so the frontend shows feedback
        error_payload = (
            'event: error\n'
            f'data: {{"type":"error","errorText":"HITL resumption encountered an error. Please retry.","code":"hitl_resume_error","recoverable":true}}\n\n'
        )
        yield error_payload
        # Emit terminal finish event to ensure SSE client reader loop terminates
        yield 'event: finish\ndata: {"type":"finish","finishReason":"error"}\n\n'
