"""get_source_highlight tool wrapper for Node 6 (Grounded Synthesis).

Enforces citation membership, page-number bounds, and graceful degradation fallback.
"""

from typing import Any
import logging
from src.state.reducers import StateValidationError
from src.state.schema import Citation
from src.tools.circuit_breaker import HIGHLIGHT_BREAKER, execute_with_retry
from src.tools.mcp_clients.mcp_client import get_mcp_client_manager
from src.tools.schemas.get_source_highlight import (
    BoundingBox,
    GetSourceHighlightInput,
    GetSourceHighlightOutput,
)

logger = logging.getLogger("shasanai.get_source_highlight")


async def get_source_highlight(
    params: GetSourceHighlightInput | dict[str, Any],
    provisional_citations: list[Citation] | None = None,
) -> GetSourceHighlightOutput:
    """Authorized wrapper for get_source_highlight MCP tool (Node 6 only).
    
    1. Validates input schema.
    2. Validates go_number belongs to provisional citations (if citations provided).
    3. Returns attached citation bounding_box_coordinates if present.
    4. Otherwise queries database via MCP client.
    5. On failure, gracefully degrades to success=True with result=None (page-number only).
    """
    if isinstance(params, dict):
        validated_input = GetSourceHighlightInput(**params)
    else:
        validated_input = params

    if provisional_citations is not None:
        known_gos = {c.go_number for c in provisional_citations}
        if validated_input.go_number not in known_gos:
            raise StateValidationError(
                f"GO number '{validated_input.go_number}' not in provisional citations: {known_gos}"
            )
        # If bounding box is already attached to provisional citation, return it directly
        for c in provisional_citations:
            if c.go_number == validated_input.go_number and c.page_number == validated_input.page_number and c.bounding_box_coordinates:
                bbox_data = c.bounding_box_coordinates
                if isinstance(bbox_data, dict):
                    return GetSourceHighlightOutput(
                        success=True,
                        result=BoundingBox(**bbox_data),
                        error=None,
                    )
                elif isinstance(bbox_data, list) and len(bbox_data) >= 4:
                    return GetSourceHighlightOutput(
                        success=True,
                        result=BoundingBox(
                            x=float(bbox_data[0]),
                            y=float(bbox_data[1]),
                            width=float(bbox_data[2]),
                            height=float(bbox_data[3]),
                        ),
                        error=None,
                    )

    sanitized_args = {
        "go_number": validated_input.go_number,
        "page_number": validated_input.page_number,
    }

    manager = get_mcp_client_manager()

    async def _call() -> GetSourceHighlightOutput:
        return await manager.call_get_source_highlight(sanitized_args)

    try:
        return await execute_with_retry(
            operation=_call,
            breaker=HIGHLIGHT_BREAKER,
            max_retries=3,
            operation_name="get_source_highlight",
        )
    except Exception as exc:
        # Graceful degradation per Section 3/9: never block synthesis on highlight lookup
        logger.warning(
            f"get_source_highlight failed for {validated_input.go_number} p.{validated_input.page_number}. Degrading gracefully: {exc!s}"
        )
        return GetSourceHighlightOutput(success=True, result=None, error=str(exc))
