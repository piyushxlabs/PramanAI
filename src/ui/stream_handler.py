"""Server-Sent Events (SSE) streaming engine and event emitter for ShasanAI.

Translates LangGraph graph node execution, state updates, tool calls, and text tokens
into the typed SSE event vocabulary defined in INTERFACE_OBSERVABILITY_SYSTEM.md Section 2a.
Strictly filters native thinking tokens server-side per LLM-1's 'Chain-of-Thought Visibility: Never'.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator
from langgraph.types import Command

from src.state.schema import StateSchema
from src.ui.event_types import (
    BaseStreamEvent,
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

logger = logging.getLogger("shasanai.stream_handler")

# Fixed Plain-Language Node Labels per INTERFACE_OBSERVABILITY_SYSTEM.md Section 3b
NODE_LABELS: dict[str, str] = {
    "query_interpretation": "Reading your question",
    "scope_screen": "Checking this is something I can help with",
    "retrieval_invocation": "Searching official records",
    "confidence_supersession": "Checking for conflicts or updates",
    "human_verification_interrupt": "Needs your input",
    "grounded_synthesis": "Drafting a cited answer",
    "citation_integrity": "Double-checking every citation",
    "refusal_redirect": "Notice",
    "response_delivery": "Done",
}


NODE_TO_STEP_ID: dict[str, str] = {
    "query_interpretation": "reading_question",
    "scope_screen": "scope_check",
    "retrieval_invocation": "searching_records",
    "confidence_supersession": "supersession_check",
    "human_verification_interrupt": "needs_approval",
    "grounded_synthesis": "drafting_answer",
    "citation_integrity": "citation_verification",
    "refusal_redirect": "refusal",
    "response_delivery": "done",
}


def format_sse_event(event: BaseStreamEvent | dict[str, Any]) -> str:
    """Formats an event into a compliant SSE line (`data: {json}\n\n`).
    
    Hard Server-Side Filter:
    Permanently drops any event with source: native-thinking or reasoning-delta markers.
    """
    if isinstance(event, dict):
        event_dict = event
    else:
        event_dict = event.model_dump(exclude_none=True)

    # Enforce Section 3a: Native model thinking tokens are NEVER forwarded to client
    if event_dict.get("type") == "reasoning-delta" or event_dict.get("source") == "native-thinking":
        logger.debug("Filtered native thinking token from SSE stream.")
        return ""

    payload = json.dumps(event_dict, ensure_ascii=False)
    return f"data: {payload}\n\n"


def format_sse_keepalive() -> str:
    """Returns an SSE keepalive comment line to prevent reverse-proxy timeouts."""
    return ": keep-alive\n\n"


async def stream_agent_turn(
    app: Any,
    initial_state_or_command: StateSchema | Command,
    config: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """Asynchronously executes a graph turn and streams typed SSE events.
    
    Emits message-start, data-graph-step, tool-*, text-delta, data-state-update,
    data-approval-required, error, and finish events.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    session_id = "default_session"
    if isinstance(initial_state_or_command, dict):
        session_id = initial_state_or_command.get("session_id", "default_session")
    elif config and "configurable" in config:
        session_id = config["configurable"].get("thread_id", "default_session")

    # 1. Emit message-start and initial progress status for Node 1
    yield format_sse_event(
        MessageStartEvent(
            id=msg_id,
            metadata={"session_id": session_id, "query_language": None},
        )
    )
    await asyncio.sleep(0)
    if isinstance(initial_state_or_command, dict):
        yield format_sse_event(
            DataGraphStepEvent(
                id=f"step_{uuid.uuid4().hex[:8]}",
                data=DataGraphStepData(
                    node="query_interpretation",
                    step=NODE_TO_STEP_ID["query_interpretation"],
                    label=NODE_LABELS["query_interpretation"],
                    status="started",
                ),
            )
        )
        await asyncio.sleep(0)
    yield format_sse_keepalive()
    await asyncio.sleep(0)

    turn_finish_reason = "success"
    _turn_citations: list = []  # accumulates citations across the turn for re-emit after Node 7

    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def _run_astream():
        try:
            async for chk in app.astream(initial_state_or_command, config=config, stream_mode="updates"):
                await queue.put(("chunk", chk))
            await queue.put(("done", None))
        except Exception as e:
            await queue.put(("error", e))

    producer_task = asyncio.create_task(_run_astream())

    try:
        while True:
            try:
                msg_type, payload = await asyncio.wait_for(queue.get(), timeout=8.0)
            except asyncio.TimeoutError:
                yield format_sse_keepalive()
                await asyncio.sleep(0)
                continue

            if msg_type == "done":
                break
            if msg_type == "error":
                raise payload

            chunk = payload

            # Check for LangGraph interrupt
            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                interrupt_val = chunk["__interrupt__"][0].value if chunk["__interrupt__"] else {}
                checkpoint_id = config.get("configurable", {}).get("thread_id", session_id) if config else session_id
                trigger_reason = interrupt_val.get("trigger", "low_confidence") if isinstance(interrupt_val, dict) else "low_confidence"

                yield format_sse_event(
                    DataApprovalRequiredEvent(
                        id=f"step_{uuid.uuid4().hex[:8]}",
                        data=DataApprovalRequiredData(
                            checkpoint_id=checkpoint_id,
                            graph_node="human_verification_interrupt",
                            trigger=trigger_reason,
                            action_preview=interrupt_val if isinstance(interrupt_val, dict) else {},
                        ),
                    )
                )
                await asyncio.sleep(0)
                turn_finish_reason = "interrupted"
                break

            if not isinstance(chunk, dict):
                continue

            for node_name, state_update in chunk.items():
                if node_name.startswith("__"):
                    continue

                step_label = NODE_LABELS.get(node_name, node_name.replace("_", " ").title())
                step_id_key = NODE_TO_STEP_ID.get(node_name, node_name)
                step_id = f"step_{uuid.uuid4().hex[:8]}"

                if isinstance(state_update, dict):
                    # Node 1 updates: query_language, query_filters
                    if "query_language" in state_update:
                        yield format_sse_event(
                            DataStateUpdateEvent(
                                id=f"upd_{uuid.uuid4().hex[:8]}",
                                data=DataStateUpdateData(
                                    field="query_language",
                                    reducer="replace-on-new-turn",
                                    value=state_update["query_language"],
                                ),
                            )
                        )
                        await asyncio.sleep(0)

                    # Node 3 updates: search_go_corpus tool call & candidate_citations
                    if node_name == "retrieval_invocation":
                        tool_id = f"call_{uuid.uuid4().hex[:8]}"
                        yield format_sse_event(
                            ToolInvocationEvent(
                                type="tool-search_go_corpus",
                                toolCallId=tool_id,
                                state="output-available",
                                input={"query_text": state_update.get("query_text", "")},
                                output={"passages_count": len(state_update.get("retrieved_passages", []))},
                            )
                        )
                        await asyncio.sleep(0)
                        if "candidate_citations" in state_update:
                            yield format_sse_event(
                                DataStateUpdateEvent(
                                    id=f"upd_{uuid.uuid4().hex[:8]}",
                                    data=DataStateUpdateData(
                                        field="candidate_citations",
                                        reducer="merge-by-key",
                                        value=len(state_update["candidate_citations"]),
                                    ),
                                )
                            )
                            await asyncio.sleep(0)

                    # Node 4 updates: compare_go_versions & confidence_score
                    if node_name == "confidence_supersession":
                        tool_id = f"call_{uuid.uuid4().hex[:8]}"
                        yield format_sse_event(
                            ToolInvocationEvent(
                                type="tool-compare_go_versions",
                                toolCallId=tool_id,
                                state="output-available",
                                output={
                                    "supersession_status": state_update.get("supersession_status", "UNKNOWN"),
                                    "conflict_count": len(state_update.get("conflict_flags", [])),
                                },
                            )
                        )
                        await asyncio.sleep(0)
                        if "confidence_score" in state_update:
                            yield format_sse_event(
                                DataStateUpdateEvent(
                                    id=f"upd_{uuid.uuid4().hex[:8]}",
                                    data=DataStateUpdateData(
                                        field="confidence_score",
                                        reducer="last-write-wins",
                                        value=state_update["confidence_score"],
                                    ),
                                )
                            )
                            await asyncio.sleep(0)

                    # Node 9 (Terminal Response Delivery) & Node 8 (Terminal Refusal): Stream final verified text
                    if node_name in ["response_delivery", "refusal_redirect"] and "answer_markdown" in state_update and state_update["answer_markdown"]:
                        from src.agents.nodes.node7_citation_integrity import extract_clean_verifiable_text
                        answer_text = extract_clean_verifiable_text(state_update["answer_markdown"])
                        text_id = f"txt_{uuid.uuid4().hex[:8]}"

                        yield format_sse_event(TextStartEvent(id=text_id))
                        await asyncio.sleep(0)
                        # Stream text word-by-word for live token UX
                        words = answer_text.split(" ")
                        for idx, word in enumerate(words):
                            chunk_delta = word + (" " if idx < len(words) - 1 else "")
                            yield format_sse_event(TextDeltaEvent(id=text_id, delta=chunk_delta))
                            await asyncio.sleep(0.003)
                        yield format_sse_event(TextEndEvent(id=text_id))
                        await asyncio.sleep(0)

                    # Emit citations SSE event on Node 6 / Node 9
                    raw_citations = state_update.get("citations", [])
                    if raw_citations:
                        _turn_citations = raw_citations
                        serialized = [
                            c.model_dump() if hasattr(c, "model_dump") else c
                            for c in raw_citations
                        ]
                        yield f"event: citations\ndata: {json.dumps({'citations': serialized}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0)

                    # Node 7: re-emit citations after integrity check passes
                    if node_name == "citation_integrity" and not state_update.get("graceful_refusal") and _turn_citations:
                        serialized = [
                            c.model_dump() if hasattr(c, "model_dump") else c
                            for c in _turn_citations
                        ]
                        yield f"event: citations\ndata: {json.dumps({'citations': serialized}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0)

                    if state_update.get("graceful_refusal") is True:
                        turn_finish_reason = "refused"

                # Emit data-graph-step (completed)
                yield format_sse_event(
                    DataGraphStepEvent(
                        id=step_id,
                        data=DataGraphStepData(
                            node=node_name,
                            step=step_id_key,
                            label=step_label,
                            status="completed",
                        ),
                    )
                )
                await asyncio.sleep(0)
                yield format_sse_keepalive()
                await asyncio.sleep(0)

    except GeneratorExit:
        logger.debug("SSE client disconnected cleanly (GeneratorExit).")
        turn_finish_reason = "cancelled"
    except Exception as exc:
        logger.exception(f"Exception during SSE stream turn: {exc!s}")
        yield format_sse_event(
            ErrorEvent(
                errorText="An error occurred while processing your administrative query. Please try again.",
                code="stream_internal_error",
                recoverable=True,
            )
        )
        await asyncio.sleep(0)
        turn_finish_reason = "error"
    finally:
        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except (asyncio.CancelledError, Exception):
                pass

    # 3. Emit finish event
    yield format_sse_event(FinishEvent(finishReason=turn_finish_reason))
    await asyncio.sleep(0)

