"""Schema definitions for get_source_highlight tool in dual Pydantic V2 and JSON Schema formats."""

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates for source highlighting in document viewer."""
    model_config = ConfigDict(strict=True, extra="forbid")

    x: float = Field(..., ge=0.0, description="X coordinate in pixels/normalized.")
    y: float = Field(..., ge=0.0, description="Y coordinate in pixels/normalized.")
    width: float = Field(..., gt=0.0, description="Box width.")
    height: float = Field(..., gt=0.0, description="Box height.")


class GetSourceHighlightInput(BaseModel):
    """Input parameters for get_source_highlight tool."""
    model_config = ConfigDict(strict=True, extra="forbid")

    go_number: str = Field(..., description="The GO number this citation belongs to.")
    page_number: int = Field(..., ge=1, description="The page number within that GO's scanned document.")


class GetSourceHighlightOutput(BaseModel):
    """Result of a bounding-box lookup."""
    model_config = ConfigDict(strict=True, extra="forbid")

    success: bool = Field(..., description="Whether the tool call succeeded.")
    result: Optional[BoundingBox] = Field(None, description="Bounding-box coordinates, if available for this scan.")
    error: Optional[str] = Field(None, description="Error message if success is false.")


GET_SOURCE_HIGHLIGHT_JSON_SCHEMA: dict[str, Any] = {
    "name": "get_source_highlight",
    "description": "Retrieves page-level bounding-box coordinates for a specific Government Order page, for source highlighting in the document viewer.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "go_number": {"type": "string", "description": "The GO number this citation belongs to."},
            "page_number": {"type": "integer", "minimum": 1, "description": "The page number within that GO's scanned document."},
        },
        "required": ["go_number", "page_number"],
        "additionalProperties": False,
    },
}
