"""Typed streaming event data contract for ShasanAI Server-Sent Events (SSE).

Defines strict Pydantic V2 models for all 10 event types specified in
INTERFACE_OBSERVABILITY_SYSTEM.md Section 2a.
"""

from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class BaseStreamEvent(BaseModel):
    """Base class for all SSE stream events with strict validation."""
    model_config = ConfigDict(strict=True, extra="forbid")


class MessageStartEvent(BaseStreamEvent):
    """Emitted once per turn when query is acknowledged."""
    type: Literal["message-start"] = "message-start"
    id: str = Field(..., description="Unique message ID.")
    metadata: dict[str, Any] = Field(..., description="Metadata containing session_id and query_language.")


class TextStartEvent(BaseStreamEvent):
    """Emitted when answer or refusal text stream begins."""
    type: Literal["text-start"] = "text-start"
    id: str = Field(..., description="Unique block ID.")


class TextDeltaEvent(BaseStreamEvent):
    """Emitted for incremental answer token deltas."""
    type: Literal["text-delta"] = "text-delta"
    id: str = Field(..., description="Unique block ID.")
    delta: str = Field(..., description="Incremental string chunk.")


class TextEndEvent(BaseStreamEvent):
    """Emitted when answer text stream completes."""
    type: Literal["text-end"] = "text-end"
    id: str = Field(..., description="Unique block ID.")


class DataGraphStepData(BaseModel):
    """Payload for graph-step transitions."""
    model_config = ConfigDict(strict=True, extra="forbid")
    node: str = Field(..., description="Internal graph node identifier.")
    step: Optional[str] = Field(default=None, description="Frontend UI step identifier.")
    label: str = Field(..., description="Officer-facing plain-language label.")
    status: Literal["started", "completed", "retrying"] = Field(..., description="Step execution status.")


class DataGraphStepEvent(BaseStreamEvent):
    """Emitted on entry and exit of each of the 9 graph nodes."""
    type: Literal["data-graph-step"] = "data-graph-step"
    id: str = Field(..., description="Unique step event ID.")
    data: DataGraphStepData = Field(..., description="Graph step details.")


class ToolInvocationEvent(BaseStreamEvent):
    """Emitted during tool calls (tool-search_go_corpus, tool-compare_go_versions, tool-get_source_highlight)."""
    type: str = Field(..., description="Tool event type (e.g. tool-search_go_corpus).")
    toolCallId: str = Field(..., description="Unique tool call correlation ID.")
    state: Literal["input-streaming", "input-available", "output-available", "output-error"] = Field(
        ..., description="Lifecycle state of the tool execution."
    )
    input: Optional[dict[str, Any]] = Field(default=None, description="Tool input parameters.")
    output: Optional[dict[str, Any]] = Field(default=None, description="Tool output results.")
    errorText: Optional[str] = Field(default=None, description="Tool error description if failed.")


class DataStateUpdateData(BaseModel):
    """Payload for typed state mutations."""
    model_config = ConfigDict(strict=True, extra="forbid")
    field: str = Field(..., description="StateSchema field name.")
    reducer: Literal["append-only", "merge-by-key", "last-write-wins", "replace-on-new-turn", "immutable-after-init"] = Field(
        ..., description="Reducer applied to the state write."
    )
    value: Any = Field(..., description="Value written into state.")


class DataStateUpdateEvent(BaseStreamEvent):
    """Emitted after typed state mutations in graph nodes."""
    type: Literal["data-state-update"] = "data-state-update"
    id: str = Field(..., description="Unique state update event ID.")
    data: DataStateUpdateData = Field(..., description="State update details.")


class DataApprovalRequiredData(BaseModel):
    """Payload for Human Verification Interrupt gate."""
    model_config = ConfigDict(strict=True, extra="forbid")
    checkpoint_id: str = Field(..., description="Checkpoint thread reference.")
    graph_node: str = Field(default="human_verification_interrupt", description="Interrupt node name.")
    trigger: Literal["low_confidence", "conflict", "personal_data"] = Field(
        ..., description="Trigger reason for human verification."
    )
    action_preview: dict[str, Any] = Field(..., description="Candidate GOs, confidence, or conflict details.")


class DataApprovalRequiredEvent(BaseStreamEvent):
    """Emitted when Node 5 pauses execution for officer verification."""
    type: Literal["data-approval-required"] = "data-approval-required"
    id: str = Field(..., description="Unique approval event ID.")
    data: DataApprovalRequiredData = Field(..., description="Approval card data.")


class ErrorEvent(BaseStreamEvent):
    """Emitted on unrecoverable tool or graph execution failure."""
    type: Literal["error"] = "error"
    errorText: str = Field(..., description="Officer-facing error explanation.")
    code: str = Field(..., description="Internal error code.")
    recoverable: bool = Field(default=False, description="Whether retry is permitted.")


class FinishEvent(BaseStreamEvent):
    """Emitted once at the true end of the agent turn."""
    type: Literal["finish"] = "finish"
    finishReason: Literal["success", "refused", "interrupted", "error"] = Field(
        ..., description="Turn termination reason."
    )


StreamEvent = Union[
    MessageStartEvent,
    TextStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    DataGraphStepEvent,
    ToolInvocationEvent,
    DataStateUpdateEvent,
    DataApprovalRequiredEvent,
    ErrorEvent,
    FinishEvent,
]
