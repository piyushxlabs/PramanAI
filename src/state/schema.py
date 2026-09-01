"""Type-safe central state schemas and domain models for ShasanAI.

Strict Pydantic V2 models defining the 17-field StateSchema and all cognitive structured outputs.
"""

from typing import Annotated, Any, Literal, Optional, TypedDict
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.state.reducers import (
    AgentError,
    ApprovalTimeoutError,
    ScopeViolationError,
    StateValidationError,
    ToolExecutionError,
    append_only_reducer,
    immutable_reducer,
    last_write_wins_reducer,
    merge_by_citation_key_reducer,
    replace_on_new_turn_reducer,
)


class StrictBaseModel(BaseModel):
    """Base Pydantic model enforcing strict type checking."""
    model_config = ConfigDict(strict=True, extra="forbid")


# ---------------------------------------------------------------------------
# Core Domain Models
# ---------------------------------------------------------------------------

class OfficerContext(StrictBaseModel):
    """Authenticated officer profile and authorization scope."""
    department: str = Field(..., description="Primary department of the officer.")
    access_scope: list[str] = Field(default_factory=list, description="Authorized department access scopes.")


class QueryFilters(StrictBaseModel):
    """Extracted or defaulted query metadata filters."""
    department: Optional[str] = Field(None, description="Department filter if specified.")
    year_range: Optional[list[int]] = Field(None, description="[start_year, end_year] inclusive filter.")
    policy_category: Optional[str] = Field(None, description="Policy category filter if specified.")
    go_number: Optional[str] = Field(None, description="Explicit GO number filter if specified in query.")


class Message(StrictBaseModel):
    """A conversational turn in message history."""
    role: Literal["user", "assistant", "system"] = Field(..., description="Role of the message author.")
    content: str = Field(..., description="Message text content.")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of message creation.")


class PassageMatch(StrictBaseModel):
    """A single retrieved passage excerpt from the indexed corpus."""
    go_number: str = Field(..., description="Government Order identifier.")
    issuing_department: str = Field(..., description="Issuing department.")
    date: str = Field(..., description="Date of issuance (YYYY-MM-DD).")
    page_number: int = Field(..., ge=1, description="Page number of the excerpt.")
    exact_text_excerpt: str = Field(..., description="Verbatim text excerpt.")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Retrieval relevance score.")

    @field_validator("relevance_score", mode="before")
    @classmethod
    def sanitize_score(cls, v: Any) -> float:
        """Ensures relevance score is non-NaN, non-None, and clamped between 0.0 and 1.0."""
        import math
        if v is None:
            return 0.0
        try:
            val = float(v)
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0


class Citation(StrictBaseModel):
    """Evidentiary citation backing an answer claim."""
    go_number: str = Field(..., description="Government Order identifier.")
    issuing_department: str = Field(default="उत्तराखण्ड शासन", description="Issuing department.")
    date: str = Field(default="2018-01-01", description="Date of issuance (YYYY-MM-DD).")
    page_number: int = Field(default=1, ge=1, description="Page number of the source citation.")
    exact_text_excerpt: str = Field(default="", description="Verbatim source excerpt.")
    bounding_box_coordinates: Optional[dict[str, Any] | list[float]] = Field(
        None, description="Optional bounding-box coordinates for document viewer overlay."
    )


class ConflictRecord(StrictBaseModel):
    """Record of a detected conflict between multiple GOs."""
    go_numbers: list[str] = Field(..., min_length=2, description="GO numbers involved in the conflict.")
    description: str = Field(..., description="Description of the conflicting provisions.")


class ApprovalState(StrictBaseModel):
    """HITL resolution state for human verification."""
    action: Literal["approve", "deny"] = Field(..., description="Human officer decision.")
    checkpoint_id: Optional[str] = Field(None, description="ID of the checkpoint resumed.")
    resolved_go_number: Optional[str] = Field(None, description="Governing GO selected by officer in conflict case.")
    reason: Optional[str] = Field(None, description="Explanation provided on denial or override.")


class ErrorRecord(StrictBaseModel):
    """Audit log entry for failures, refusals, or errors."""
    node: str = Field(..., description="Graph node where error occurred.")
    error_type: str = Field(..., description="Category of error.")
    message: str = Field(..., description="Detailed error description.")
    timestamp: Optional[str] = Field(None, description="ISO timestamp.")


class RuntimeConfig(StrictBaseModel):
    """Session configuration and safety thresholds."""
    confidence_threshold_low: float = Field(0.6, ge=0.0, le=1.0)
    confidence_threshold_high: float = Field(0.85, ge=0.0, le=1.0)
    session_idle_timeout_seconds: int = Field(900, ge=1)
    rate_limit_queries_per_minute: int = Field(20, ge=1)
    max_citation_retries: int = Field(2, ge=1)
    max_tool_retries: int = Field(3, ge=1)


# ---------------------------------------------------------------------------
# Structured Output Schemas (Cognitive Nodes)
# ---------------------------------------------------------------------------

class QueryInterpretation(StrictBaseModel):
    """Node 1 Structured Output: Normalized query and detected language/filters."""
    query_text: str = Field(..., description="Normalized query text.")
    query_language: Literal["hi", "en", "hinglish"] = Field(..., description="Detected language.")
    query_filters: QueryFilters = Field(..., description="Extracted or defaulted filters.")


class ScopeScreenDecision(StrictBaseModel):
    """Node 2 Structured Output: In-scope/out-of-scope routing decision."""
    in_scope: bool = Field(..., description="True if query is a valid in-scope citation lookup.")
    category: Optional[Literal["financial_disbursement", "grievance", "order_drafting", "policy_opinion"]] = Field(
        None, description="Out-of-scope category if in_scope is False."
    )
    reason: Optional[str] = Field(None, description="Explanation if out-of-scope.")


class ConfidenceSupersessionAssessment(StrictBaseModel):
    """Node 4 Structured Output: Grounding confidence, supersession status, and safety flags."""
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Grounding confidence score.")
    supersession_status: Literal["CURRENT_ACTIVE", "AMENDED", "SUPERSEDED", "UNKNOWN"] = Field(
        ..., description="Supersession status of leading candidate GO."
    )
    conflict_flags: list[ConflictRecord] = Field(
        default_factory=list, description="Detected conflicts between candidate GOs."
    )
    personal_data_flag: bool = Field(..., description="True if retrieved text contains citizen personal data.")
    requires_deep_reasoning: bool = Field(
        ..., description="True if borderline and requires multi-pass deterministic re-invocation."
    )

    @field_validator("confidence_score", mode="before")
    @classmethod
    def normalize_confidence(cls, v: Any) -> float:
        """Normalizes integer percentages [2, 100] to float [0.02, 1.0]."""
        if isinstance(v, int) and 1 < v <= 100:
            return float(v) / 100.0
        return float(v)


class GroundedAnswer(StrictBaseModel):
    """Node 6 Structured Output: Synthesized grounded answer and citation set."""
    answer_markdown: str = Field(..., description="Markdown answer with verbatim citations.")
    citations: list[Citation] = Field(..., min_length=1, description="Final validated citations used in answer.")


class CitationIntegrityResult(StrictBaseModel):
    """Node 7 Structured Output: Citation integrity verification result."""
    all_claims_cited: bool = Field(..., description="True if every factual claim maps to a citation.")
    uncited_claims: list[str] = Field(default_factory=list, description="Verbatim sentences without citation.")


# ---------------------------------------------------------------------------
# Tool Schemas
# ---------------------------------------------------------------------------

class SearchGoCorpusInput(StrictBaseModel):
    """Input parameters for search_go_corpus tool."""
    query_text: str = Field(..., max_length=500, description="Normalized search query.")
    query_language: Literal["hi", "en", "hinglish"] = Field(..., description="Detected query language.")
    department_filter: Optional[str] = Field(None, description="Issuing department filter.")
    year_range_filter: Optional[list[int]] = Field(None, description="[start_year, end_year] filter.")
    policy_category_filter: Optional[str] = Field(None, description="Policy category filter.")
    max_results: int = Field(10, ge=1, le=25, description="Max candidate passages.")


class SearchGoCorpusOutput(StrictBaseModel):
    """Output result from search_go_corpus tool."""
    success: bool = Field(..., description="Whether search succeeded.")
    result: Optional[list[PassageMatch]] = Field(None, description="Candidate passages.")
    error: Optional[str] = Field(None, description="Error message if failed.")


class CompareGoVersionsInput(StrictBaseModel):
    """Input parameters for compare_go_versions tool."""
    go_numbers: list[str] = Field(..., min_length=1, max_length=20, description="Candidate GO numbers.")
    department: Optional[str] = Field(None, description="Issuing department.")


class SupersessionLink(StrictBaseModel):
    """Supersession linkage record for a single GO."""
    go_number: str = Field(..., description="GO identifier.")
    status: Literal["CURRENT_ACTIVE", "AMENDED", "SUPERSEDED", "UNKNOWN"] = Field(...)
    superseded_by: Optional[str] = Field(None, description="Superseding GO number if applicable.")
    amends: Optional[str] = Field(None, description="Amended GO number if applicable.")


class CompareGoVersionsOutput(StrictBaseModel):
    """Output result from compare_go_versions tool."""
    success: bool = Field(..., description="Whether version comparison succeeded.")
    result: Optional[list[SupersessionLink]] = Field(None, description="Supersession links.")
    error: Optional[str] = Field(None, description="Error message if failed.")


class BoundingBox(StrictBaseModel):
    """Bounding box coordinates for source highlighting."""
    x: float = Field(...)
    y: float = Field(...)
    width: float = Field(...)
    height: float = Field(...)


class GetSourceHighlightInput(StrictBaseModel):
    """Input parameters for get_source_highlight tool."""
    go_number: str = Field(..., description="GO number.")
    page_number: int = Field(..., ge=1, description="Page number in document.")


class GetSourceHighlightOutput(StrictBaseModel):
    """Output result from get_source_highlight tool."""
    success: bool = Field(..., description="Whether highlight lookup succeeded.")
    result: Optional[BoundingBox] = Field(None, description="Bounding-box coordinates.")
    error: Optional[str] = Field(None, description="Error message if failed.")


# ---------------------------------------------------------------------------
# Global 17-Field StateSchema (LangGraph TypedDict)
# ---------------------------------------------------------------------------

class StateSchema(TypedDict):
    """Single central 17-field StateSchema for LangGraph Pregel state machine."""
    session_id: Annotated[str, immutable_reducer]
    officer_context: Annotated[OfficerContext, immutable_reducer]
    query_text: Annotated[str, last_write_wins_reducer]
    query_language: Annotated[Literal["hi", "en", "hinglish"], last_write_wins_reducer]
    query_filters: Annotated[QueryFilters, last_write_wins_reducer]
    message_history: Annotated[list[Message], append_only_reducer]
    retrieved_passages: Annotated[list[PassageMatch], replace_on_new_turn_reducer]
    candidate_citations: Annotated[list[Citation], merge_by_citation_key_reducer]
    confidence_score: Annotated[float, last_write_wins_reducer]
    supersession_status: Annotated[Literal["CURRENT_ACTIVE", "AMENDED", "SUPERSEDED", "UNKNOWN"], last_write_wins_reducer]
    conflict_flags: Annotated[list[ConflictRecord], append_only_reducer]
    human_verification: Annotated[Optional[ApprovalState], last_write_wins_reducer]
    answer_markdown: Annotated[Optional[str], last_write_wins_reducer]
    citations: Annotated[list[Citation], last_write_wins_reducer]
    graceful_refusal: Annotated[bool, last_write_wins_reducer]
    error_logs: Annotated[list[ErrorRecord], append_only_reducer]
    config: Annotated[RuntimeConfig, immutable_reducer]
