import math
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PassageMatch(BaseModel):
    """A single retrieved passage excerpt from the indexed corpus."""
    model_config = ConfigDict(strict=True, extra="forbid")

    go_number: str = Field(..., description="Government Order identifier.")
    issuing_department: str = Field(..., description="Issuing department.")
    date: str = Field(..., description="Date of issuance (YYYY-MM-DD).")
    page_number: int = Field(..., ge=1, description="Page number of the excerpt.")
    exact_text_excerpt: str = Field(..., description="Verbatim text excerpt.")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Retrieval relevance score.")
    # Populated from VLM/layout extraction during ingestion; [x, y, width, height] in normalized page coordinates
    bounding_box_coordinates: Optional[list[float]] = Field(
        default=None, description="Bounding box [x, y, width, height] in page pixels for visual grounding."
    )

    @field_validator("relevance_score", mode="before")
    @classmethod
    def sanitize_score(cls, v: Any) -> float:
        """Ensures relevance score is non-NaN, non-None, and clamped between 0.0 and 1.0."""
        if v is None:
            return 0.0
        try:
            val = float(v)
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0


class SearchGoCorpusInput(BaseModel):
    """Input parameters for search_go_corpus hybrid retrieval."""
    model_config = ConfigDict(strict=True, extra="forbid")

    query_text: str = Field(..., max_length=500, description="Normalized search query, expanded with obvious synonyms.")
    query_language: Literal["hi", "en", "hinglish"] = Field(..., description="Detected language of the officer's query.")
    department_filter: Optional[str] = Field(None, description="Restrict results to this issuing department, if specified.")
    year_range_filter: Optional[list[int]] = Field(None, description="[start_year, end_year] inclusive, if specified.")
    policy_category_filter: Optional[str] = Field(None, description="Restrict results to this policy category, if specified.")
    go_number_filter: Optional[str] = Field(None, description="Explicit GO number to boost or filter, if specified.")
    max_results: int = Field(10, ge=1, le=25, description="Maximum number of candidate passages to return.")


class SearchGoCorpusOutput(BaseModel):
    """Result of a hybrid retrieval call."""
    model_config = ConfigDict(strict=True, extra="forbid")

    success: bool = Field(..., description="Whether the tool call succeeded.")
    result: Optional[list[PassageMatch]] = Field(None, description="Ranked candidate passages, possibly empty.")
    error: Optional[str] = Field(None, description="Error message if success is false.")


SEARCH_GO_CORPUS_JSON_SCHEMA: dict[str, Any] = {
    "name": "search_go_corpus",
    "description": "Hybrid dense+sparse retrieval over the indexed bilingual Government Order/circular corpus, scoped to the officer's department access.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "query_text": {"type": "string", "description": "Normalized search query, expanded with obvious synonyms."},
            "query_language": {"type": "string", "enum": ["hi", "en", "hinglish"], "description": "Detected language of the officer's query."},
            "department_filter": {"type": ["string", "null"], "description": "Restrict results to this issuing department, if specified."},
            "year_range_filter": {"type": ["array", "null"], "items": {"type": "integer"}, "description": "[start_year, end_year] inclusive, if specified."},
            "policy_category_filter": {"type": ["string", "null"], "description": "Restrict results to this policy category, if specified."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 25, "description": "Maximum number of candidate passages to return."},
        },
        "required": ["query_text", "query_language", "department_filter", "year_range_filter", "policy_category_filter", "max_results"],
        "additionalProperties": False,
    },
}
