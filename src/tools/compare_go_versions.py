"""compare_go_versions tool wrapper for Node 4 (Supersession & Confidence Analysis).

Enforces candidate_citations membership, input sanitization, exponential backoff, and circuit breaker.
"""

from typing import Any
from src.state.reducers import StateValidationError
from src.state.schema import Citation
from src.tools.circuit_breaker import SUPERSESSION_BREAKER, execute_with_retry
from src.tools.mcp_clients.mcp_client import get_mcp_client_manager
from src.tools.sanitization import sanitize_department
from src.tools.schemas.compare_go_versions import (
    CompareGoVersionsInput,
    CompareGoVersionsOutput,
)


async def compare_go_versions(
    params: CompareGoVersionsInput | dict[str, Any],
    candidate_citations: list[Citation] | None = None,
) -> CompareGoVersionsOutput:
    """Authorized wrapper for compare_go_versions MCP tool (Node 4 only).
    
    1. Validates input schema.
    2. Validates that requested GO numbers exist in candidate_citations (if candidate_citations is passed).
    3. Calls MCP server with exponential backoff and circuit breaker.
    """
    if isinstance(params, dict):
        validated_input = CompareGoVersionsInput(**params)
    else:
        validated_input = params

    if not validated_input.go_numbers:
        raise StateValidationError("compare_go_versions requires at least one GO number")

    # If candidate_citations provided, enforce membership check
    if candidate_citations is not None:
        known_gos = {c.go_number for c in candidate_citations}
        for go in validated_input.go_numbers:
            if go not in known_gos:
                raise StateValidationError(
                    f"GO number '{go}' in compare_go_versions is not in candidate_citations set: {known_gos}"
                )

    clean_dept = sanitize_department(validated_input.department)

    sanitized_args = {
        "go_numbers": validated_input.go_numbers,
        "department": clean_dept,
    }

    manager = get_mcp_client_manager()

    async def _call() -> CompareGoVersionsOutput:
        return await manager.call_compare_go_versions(sanitized_args)

    return await execute_with_retry(
        operation=_call,
        breaker=SUPERSESSION_BREAKER,
        max_retries=3,
        operation_name="compare_go_versions",
    )
