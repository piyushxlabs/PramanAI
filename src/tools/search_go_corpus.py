"""search_go_corpus tool wrapper for Node 3 (Retrieval Invocation).

Enforces input sanitization, server-side access_scope injection, exponential backoff, and circuit breaker.
"""

from typing import Any
from src.state.schema import OfficerContext
from src.tools.circuit_breaker import RETRIEVAL_BREAKER, execute_with_retry
from src.tools.mcp_clients.mcp_client import get_mcp_client_manager
from src.tools.sanitization import (
    inject_server_access_scope,
    sanitize_department,
    sanitize_policy_category,
    sanitize_query_text,
    sanitize_year_range,
)
from src.tools.schemas.search_go_corpus import SearchGoCorpusInput, SearchGoCorpusOutput


async def search_go_corpus(
    params: SearchGoCorpusInput | dict[str, Any],
    officer_context: OfficerContext,
) -> SearchGoCorpusOutput:
    """Authorized wrapper for search_go_corpus MCP tool (Node 3 only).
    
    1. Sanitizes all inputs (query text, department filter, year range, category).
    2. Enforces officer_context.access_scope server-side.
    3. Calls MCP server with exponential backoff and circuit breaker.
    """
    # 1. Input parsing and sanitization
    if isinstance(params, dict):
        validated_input = SearchGoCorpusInput(**params)
    else:
        validated_input = params

    clean_query = sanitize_query_text(validated_input.query_text)
    clean_dept = sanitize_department(validated_input.department_filter)
    clean_years = sanitize_year_range(validated_input.year_range_filter)
    clean_cat = sanitize_policy_category(validated_input.policy_category_filter)

    # 2. Server-side access scope verification
    authorized_scopes = inject_server_access_scope(officer_context, clean_dept)

    sanitized_args = {
        "query_text": clean_query,
        "query_language": validated_input.query_language,
        "department_filter": clean_dept,
        "year_range_filter": clean_years,
        "policy_category_filter": clean_cat,
        "go_number_filter": validated_input.go_number_filter,
        "max_results": validated_input.max_results,
        "access_scope": authorized_scopes,
    }

    # 3. Execution via MCP client with retry and circuit breaker
    manager = get_mcp_client_manager()

    async def _call() -> SearchGoCorpusOutput:
        return await manager.call_search_go_corpus(sanitized_args)

    output: SearchGoCorpusOutput = await execute_with_retry(
        operation=_call,
        breaker=RETRIEVAL_BREAKER,
        max_retries=3,
        operation_name="search_go_corpus",
    )

    if output and output.result:
        # Filter strictly by relevance threshold >= 0.50 (unless exact GO filter was queried)
        if not validated_input.go_number_filter:
            filtered_passages = [p for p in output.result if float(p.relevance_score) >= 0.50]
        else:
            filtered_passages = output.result
        filtered_passages.sort(key=lambda p: float(p.relevance_score), reverse=True)
        output.result = filtered_passages

    return output
