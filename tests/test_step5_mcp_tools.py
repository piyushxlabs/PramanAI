"""Verification tests for Step 5: Read-Only MCP Tool Client Layer, Dual Schemas & Sanitization."""

import pytest
from src.state.reducers import ScopeViolationError, StateValidationError, ToolExecutionError
from src.state.schema import Citation, OfficerContext
from src.tools.circuit_breaker import CircuitBreaker, execute_with_retry
from src.tools.compare_go_versions import compare_go_versions
from src.tools.get_source_highlight import get_source_highlight
from src.tools.mcp_clients.mcp_client import get_mcp_client_manager
from src.tools.sanitization import (
    inject_server_access_scope,
    sanitize_department,
    sanitize_policy_category,
    sanitize_query_text,
    sanitize_year_range,
)
from src.tools.schemas.compare_go_versions import (
    COMPARE_GO_VERSIONS_JSON_SCHEMA,
    CompareGoVersionsInput,
    CompareGoVersionsOutput,
)
from src.tools.schemas.get_source_highlight import (
    GET_SOURCE_HIGHLIGHT_JSON_SCHEMA,
    GetSourceHighlightInput,
    GetSourceHighlightOutput,
)
from src.tools.schemas.search_go_corpus import (
    SEARCH_GO_CORPUS_JSON_SCHEMA,
    SearchGoCorpusInput,
    SearchGoCorpusOutput,
)
from src.tools.search_go_corpus import search_go_corpus


# ===========================================================================
# 1. Input Sanitization & Parameter Governance Tests
# ===========================================================================

def test_query_text_sanitization_valid():
    """Verify clean query text is preserved."""
    clean = sanitize_query_text("2018 mein forest transfer policy ka circular")
    assert clean == "2018 mein forest transfer policy ka circular"


def test_query_text_sanitization_empty():
    """Verify empty or whitespace query is rejected."""
    with pytest.raises(StateValidationError):
        sanitize_query_text("   ")


def test_query_text_sanitization_length_overflow():
    """Verify query text > 500 characters is rejected."""
    long_query = "transfer " * 70  # 630 chars
    with pytest.raises(StateValidationError):
        sanitize_query_text(long_query)


def test_query_text_sanitization_sql_injection():
    """Verify SQL metacharacters and injection commands are rejected."""
    injections = [
        "transfer policy; DROP TABLE users;--",
        "GO 2018' UNION SELECT * FROM passwords--",
        "circular /* comment */",
        "transfer EXEC xp_cmdshell",
    ]
    for bad_query in injections:
        with pytest.raises(StateValidationError):
            sanitize_query_text(bad_query)


def test_query_text_sanitization_path_traversal():
    """Verify path traversal sequences are rejected."""
    traversals = [
        "show ../../../etc/passwd",
        "fetch file ..\\..\\boot.ini",
        "open /etc/shadow contents",
    ]
    for bad_query in traversals:
        with pytest.raises(StateValidationError):
            sanitize_query_text(bad_query)


def test_department_filter_allowlist():
    """Verify department filter against recognized allowlist."""
    assert sanitize_department("forest") == "Forest"
    assert sanitize_department("Revenue") == "Revenue"
    assert sanitize_department(None) is None

    # Unrecognized department falls back to None for broad search
    assert sanitize_department("UnknownUnauthorizedDept") is None

    # SQL Injection raises StateValidationError
    with pytest.raises(StateValidationError):
        sanitize_department("Forest; DROP TABLE departments;--")


def test_policy_category_allowlist():
    """Verify policy category filter against recognized allowlist."""
    assert sanitize_policy_category("Transfer Policy") == "Transfer Policy"
    assert sanitize_policy_category(None) is None

    # Unrecognized category falls back to None for broad search
    assert sanitize_policy_category("InvalidCategory") is None

    # SQL Injection raises StateValidationError
    with pytest.raises(StateValidationError):
        sanitize_policy_category("Transfer' OR 1=1--")


def test_year_range_sanitization():
    """Verify year range bounds validation."""
    assert sanitize_year_range([2015, 2020]) == [2015, 2020]
    assert sanitize_year_range(None) is None

    # start_year > end_year
    with pytest.raises(StateValidationError):
        sanitize_year_range([2025, 2018])

    # year < 1950
    with pytest.raises(StateValidationError):
        sanitize_year_range([1940, 2000])


def test_access_scope_enforcement():
    """Verify server-side access scope resolution and cross-department block."""
    officer = OfficerContext(department="Forest", access_scope=["Forest"])

    # Authorized department query
    scope = inject_server_access_scope(officer, "Forest")
    assert scope == ["Forest"]

    # Unauthorized department query
    with pytest.raises(ScopeViolationError):
        inject_server_access_scope(officer, "Finance")


# ===========================================================================
# 2. Tool Wrappers & Execution Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_search_go_corpus_tool_execution():
    """Verify search_go_corpus tool execution with sanitized parameters."""
    officer = OfficerContext(department="Personnel", access_scope=["Personnel", "General", "कार्मिक अनुभाग-4"])
    inp = SearchGoCorpusInput(
        query_text="शासनादेश संख्या 115",
        query_language="hi",
        go_number_filter="115",
        max_results=5,
    )
    result = await search_go_corpus(inp, officer_context=officer)
    assert isinstance(result, SearchGoCorpusOutput)
    assert result.success is True
    assert result.result is not None
    assert len(result.result) > 0
    assert "115" in result.result[0].go_number


@pytest.mark.asyncio
async def test_compare_go_versions_tool_execution():
    """Verify compare_go_versions returns valid supersession linkages."""
    cits = [
        Citation(go_number="GO-1345/XII/2018", issuing_department="Forest", date="2018-03-12", page_number=3, exact_text_excerpt="Excerpt 1"),
        Citation(go_number="GO-562/XXX/2014", issuing_department="Forest", date="2014-07-15", page_number=1, exact_text_excerpt="Excerpt 2"),
    ]
    inp = CompareGoVersionsInput(
        go_numbers=["GO-1345/XII/2018", "GO-562/XXX/2014"],
        department="Forest",
    )
    result = await compare_go_versions(inp, candidate_citations=cits)
    assert isinstance(result, CompareGoVersionsOutput)
    assert result.success is True
    assert result.result is not None
    assert len(result.result) == 2

    # Check supersession mapping
    status_map = {link.go_number: link.status for link in result.result}
    assert status_map["GO-1345/XII/2018"] == "CURRENT_ACTIVE"
    assert status_map["GO-562/XXX/2014"] == "SUPERSEDED"


@pytest.mark.asyncio
async def test_compare_go_versions_membership_rejection():
    """Verify compare_go_versions rejects GO numbers not present in candidate_citations."""
    cits = [Citation(go_number="GO-100", issuing_department="D", date="2020-01-01", page_number=1, exact_text_excerpt="E")]
    inp = CompareGoVersionsInput(go_numbers=["GO-UNAUTHORIZED-999"])
    with pytest.raises(StateValidationError):
        await compare_go_versions(inp, candidate_citations=cits)


@pytest.mark.asyncio
async def test_get_source_highlight_and_graceful_degradation():
    """Verify get_source_highlight coordinates and graceful degradation for missing scans."""
    cits = [
        Citation(
            go_number="GO-1345/XII/2018",
            issuing_department="Forest",
            date="2018-03-12",
            page_number=3,
            exact_text_excerpt="E",
            bounding_box_coordinates={"x": 0.05, "y": 0.10, "width": 0.90, "height": 0.30},
        )
    ]

    # 1. Existing highlight from attached citation coordinates
    inp1 = GetSourceHighlightInput(go_number="GO-1345/XII/2018", page_number=3)
    res1 = await get_source_highlight(inp1, provisional_citations=cits)
    assert res1.success is True
    assert res1.result is not None
    assert res1.result.x == 0.05
    assert res1.result.y == 0.10

    # 2. Unavailable highlight (graceful degradation -> result=None, success=True)
    inp2 = GetSourceHighlightInput(go_number="GO-1345/XII/2018", page_number=99)
    res2 = await get_source_highlight(inp2, provisional_citations=cits)
    assert res2.success is True
    assert res2.result is None


# ===========================================================================
# 3. Circuit Breaker & Retry Mechanism Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_circuit_breaker_trips_after_3_failures():
    """Verify circuit breaker transitions to OPEN after 3 consecutive failures."""
    breaker = CircuitBreaker(failure_threshold=3)
    assert breaker.is_available() is True

    call_count = 0

    async def failing_operation():
        nonlocal call_count
        call_count += 1
        raise ValueError("Simulated network timeout")

    # First attempt (3 retries internal to execute_with_retry)
    with pytest.raises(ToolExecutionError):
        await execute_with_retry(
            operation=failing_operation,
            breaker=breaker,
            max_retries=2,
            initial_backoff=0.01,
            operation_name="test_op",
        )

    assert breaker.failure_count == 1
    assert breaker.is_available() is True

    # Force 2 more failures
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_available() is False

    # Next call rejected immediately without executing operation
    with pytest.raises(ToolExecutionError, match="Circuit breaker is OPEN"):
        await execute_with_retry(
            operation=failing_operation,
            breaker=breaker,
            max_retries=1,
            operation_name="test_op",
        )


# ===========================================================================
# 4. MCP Tools Manifest & JSON Schemas Integrity Tests
# ===========================================================================

def test_mcp_tools_manifest():
    """Verify MCP tools manifest returns exactly 3 authorized read-only tools."""
    manager = get_mcp_client_manager()
    manifest = manager.get_tools_manifest()
    assert len(manifest) == 3

    tool_names = {t["name"] for t in manifest}
    assert tool_names == {"search_go_corpus", "compare_go_versions", "get_source_highlight"}

    # Strict mode checks
    assert SEARCH_GO_CORPUS_JSON_SCHEMA["strict"] is True
    assert COMPARE_GO_VERSIONS_JSON_SCHEMA["strict"] is True
    assert GET_SOURCE_HIGHLIGHT_JSON_SCHEMA["strict"] is True
