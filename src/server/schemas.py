"""Pydantic V2 Request and Response schemas for ShasanAI FastAPI Backend Server.

Defines strict schemas for chat queries, HITL graph resumption, officer feedback,
and health inspection.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.state.schema import OfficerContext, QueryFilters


class StrictSchema(BaseModel):
    """Base model with strict validation and no extra fields."""
    model_config = ConfigDict(strict=True, extra="forbid")


class ChatQueryRequest(StrictSchema):
    """Request payload to initiate a new conversational query turn."""
    session_id: str = Field(..., min_length=1, description="Unique session/thread identifier.")
    query_text: str = Field(..., min_length=1, max_length=1000, description="Officer query text.")
    officer_context: Optional[OfficerContext] = Field(
        default=None,
        description="Authenticated officer metadata (department, access scope, role).",
    )
    query_filters: Optional[QueryFilters] = Field(
        default=None,
        description="Optional explicit officer-selected query filters (department, year range, category).",
    )


class ModifiedInputs(StrictSchema):
    """Optional modification payload on human verification approval."""
    resolved_go_number: Optional[str] = Field(
        default=None,
        description="Selected authoritative GO number when resolving a conflict.",
    )


class HITLResumptionRequest(StrictSchema):
    """Payload to resume a paused LangGraph execution at Node 5."""
    action: Literal["approve", "deny"] = Field(..., description="Officer decision.")
    checkpoint_id: str = Field(..., min_length=1, description="Checkpoint thread reference.")
    modified_inputs: Optional[ModifiedInputs] = Field(
        default=None,
        description="Selected GO number on approval if resolving conflict.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Explanatory reason when denying or overriding.",
    )


class OfficerFeedbackRequest(StrictSchema):
    """Officer thumbs up/down rating submission."""
    session_id: str = Field(..., min_length=1, description="Session identifier.")
    trace_id: Optional[str] = Field(default=None, description="OpenTelemetry trace correlation ID.")
    feedback_value: bool = Field(..., description="True for thumbs-up, False for thumbs-down.")
    comment: Optional[str] = Field(default=None, max_length=2000, description="Optional officer comments.")


class CitationAccuracyFeedbackRequest(StrictSchema):
    """Per-citation incorrect flag submission."""
    session_id: str = Field(..., min_length=1, description="Session identifier.")
    go_number: str = Field(..., min_length=1, description="Target Government Order number.")
    page_number: int = Field(..., ge=1, description="Target document page number.")
    trace_id: Optional[str] = Field(default=None, description="OpenTelemetry trace correlation ID.")
    is_accurate: bool = Field(default=False, description="Whether citation is accurate or flawed.")
    comment: Optional[str] = Field(default=None, max_length=2000, description="Officer critique/correction.")


class ApiResponse(StrictSchema):
    """Standard API response wrapper."""
    success: bool = True
    message: str = "Operation completed successfully."
    data: Optional[dict[str, Any]] = None


class HealthResponse(StrictSchema):
    """Health check response schema."""
    status: str = "healthy"
    service: str = "pramanai-backend"
    version: str = "2.0.0"
    models: dict[str, str] = Field(
        default_factory=lambda: {
            "inference": "gemini-3.5-flash",
            "fast_inference": "gemini-3.5-flash-lite",
            "armor": "gemma-2-2b-it",
            "vision": "gemini-3.5-flash",
        }
    )
    database: str = "connected"


# ---------------------------------------------------------------------------
# Authentication & User Management Schemas
# ---------------------------------------------------------------------------

class AuthLoginRequest(StrictSchema):
    """User authentication login payload."""
    email: str = Field(..., min_length=3, max_length=255, description="Officer official email address.")
    password: str = Field(..., min_length=1, max_length=255, description="Officer password.")


class UserProfile(StrictSchema):
    """Authenticated officer profile and role metadata."""
    id: int = Field(..., description="Unique user ID.")
    email: str = Field(..., description="Official government email address.")
    full_name: str = Field(..., description="Officer full name.")
    department: str = Field(..., description="Assigned government department.")
    designation: str = Field(..., description="Official administrative designation.")
    role: str = Field(default="OFFICER", description="Access role (OFFICER, ADMIN, AUDITOR).")
    created_at: Optional[str] = Field(default=None, description="Account creation timestamp.")


class AuthTokenResponse(StrictSchema):
    """Authentication successful response with JWT Bearer token."""
    access_token: str = Field(..., description="JWT Bearer access token.")
    token_type: str = Field(default="bearer", description="Token authorization type.")
    user: UserProfile = Field(..., description="Authenticated officer profile.")


# ---------------------------------------------------------------------------
# Persistent Chat History Schemas
# ---------------------------------------------------------------------------

class ChatSessionItem(StrictSchema):
    """Metadata summary of a persistent chat session."""
    session_id: str = Field(..., description="Unique session thread ID.")
    user_id: Optional[int] = Field(default=None, description="Owning user ID.")
    title: str = Field(..., description="Auto-generated or assigned session title.")
    department: str = Field(..., description="Department scope for this session.")
    created_at: str = Field(..., description="Session creation ISO timestamp.")
    updated_at: str = Field(..., description="Last query ISO timestamp.")


class ChatSessionListResponse(StrictSchema):
    """List of chat sessions for authenticated officer."""
    sessions: list[ChatSessionItem] = Field(default_factory=list, description="List of sessions.")
    total: int = Field(default=0, description="Total count of sessions.")


class ChatSessionDetailResponse(StrictSchema):
    """Full detail of a restored chat session including messages and verified citations."""
    session_id: str = Field(..., description="Unique session thread ID.")
    user_id: Optional[int] = Field(default=None, description="Owning user ID.")
    title: str = Field(..., description="Session title.")
    department: str = Field(..., description="Department scope.")
    created_at: str = Field(..., description="Creation ISO timestamp.")
    updated_at: str = Field(..., description="Last query ISO timestamp.")
    messages: list[dict[str, Any]] = Field(default_factory=list, description="Historical turn messages.")
    citations: list[dict[str, Any]] = Field(default_factory=list, description="Verified citations from last turn.")

