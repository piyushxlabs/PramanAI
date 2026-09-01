"""Schema definitions for compare_go_versions tool in dual Pydantic V2 and JSON Schema formats."""

from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class SupersessionLink(BaseModel):
    """Supersession linkage record for a single GO."""
    model_config = ConfigDict(strict=True, extra="forbid")

    go_number: str = Field(..., description="GO identifier.")
    status: Literal["CURRENT_ACTIVE", "AMENDED", "SUPERSEDED", "UNKNOWN"] = Field(
        ..., description="Current validity status of this order."
    )
    superseded_by: Optional[str] = Field(None, description="Superseding GO number if status is SUPERSEDED.")
    amends: Optional[str] = Field(None, description="Amended GO number if status is AMENDED.")


class CompareGoVersionsInput(BaseModel):
    """Input parameters for compare_go_versions tool."""
    model_config = ConfigDict(strict=True, extra="forbid")

    go_numbers: list[str] = Field(
        ..., min_length=1, max_length=20, description="Candidate GO numbers to compare, from candidate_citations."
    )
    department: Optional[str] = Field(
        None, description="Issuing department, to disambiguate GO numbering schemes if needed."
    )


class CompareGoVersionsOutput(BaseModel):
    """Result of a version-comparison call."""
    model_config = ConfigDict(strict=True, extra="forbid")

    success: bool = Field(..., description="Whether the tool call succeeded.")
    result: Optional[list[SupersessionLink]] = Field(None, description="Supersession status per GO number.")
    error: Optional[str] = Field(None, description="Error message if success is false.")


COMPARE_GO_VERSIONS_JSON_SCHEMA: dict[str, Any] = {
    "name": "compare_go_versions",
    "description": "Checks amendment/supersession linkage across a set of candidate Government Order numbers.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "go_numbers": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Candidate GO numbers to compare, from candidate_citations.",
            },
            "department": {
                "type": ["string", "null"],
                "description": "Issuing department, to disambiguate GO numbering schemes if needed.",
            },
        },
        "required": ["go_numbers", "department"],
        "additionalProperties": False,
    },
}
