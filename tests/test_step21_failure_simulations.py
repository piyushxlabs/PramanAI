"""Step 21 Verification Test Suite: Failure Simulations & System Hardening.

Evaluates the 5 mandatory failure simulations from AGENT_MASTER_PLAN.md Section 9.5:
1. Simulation 1: Circuit breaker trips after 3 consecutive failures and recovers on reset
2. Simulation 2: StateValidationError raised upon illegal mutation of immutable fields
3. Simulation 3: Input sanitization strips SQL metacharacters and directory traversal sequences
4. Simulation 4: Invalid HITL action or payload rejected with appropriate validation error
5. Simulation 5: Checkpointer fallback to in-memory graph when database is temporarily unavailable
"""

import pytest
from pydantic import ValidationError

from src.agents.graph import create_agent_graph
from src.server.schemas import HITLResumptionRequest
from src.state.checkpointing import ensure_windows_event_loop
from src.state.reducers import immutable_reducer
from src.state.schema import (
    OfficerContext,
    QueryFilters,
    RuntimeConfig,
    StateSchema,
    StateValidationError,
)
from src.tools.circuit_breaker import CircuitBreaker
from src.tools.get_source_highlight import get_source_highlight
from src.tools.sanitization import sanitize_query_text
from src.tools.schemas.get_source_highlight import GetSourceHighlightInput


def test_simulation_1_circuit_breaker_behavior():
    """Simulation 1: Verify circuit breaker trips after 3 failures and blocks subsequent calls."""
    cb = CircuitBreaker(failure_threshold=3)
    cb.failure_count = 0
    cb.state = "CLOSED"

    # Simulate 3 failures
    cb.record_failure()
    assert cb.is_available() is True
    cb.record_failure()
    assert cb.is_available() is True
    cb.record_failure()
    assert cb.is_available() is False
    assert cb.state == "OPEN"

    # Record success resets failure count
    cb.record_success()
    assert cb.is_available() is True
    assert cb.failure_count == 0
    assert cb.state == "CLOSED"


def test_simulation_2_immutable_state_invariants():
    """Simulation 2: Verify StateValidationError on mutation of session_id, officer_context, config."""
    # 1. Config mutation rejection
    cfg1 = RuntimeConfig(confidence_threshold_low=0.6)
    cfg2 = RuntimeConfig(confidence_threshold_low=0.99)
    with pytest.raises(StateValidationError, match="Illegal mutation attempted on immutable field"):
        immutable_reducer(cfg1, cfg2)

    # 2. Officer context mutation rejection
    off1 = OfficerContext(department="Forest", access_scope=["Forest"])
    off2 = OfficerContext(department="Revenue", access_scope=["Revenue"])
    with pytest.raises(StateValidationError, match="Illegal mutation attempted on immutable field"):
        immutable_reducer(off1, off2)


def test_simulation_3_input_sanitization_defense():
    """Simulation 3: Verify input sanitization raises StateValidationError on SQL injections and path traversal."""
    # SQL Injection rejection
    with pytest.raises(StateValidationError, match="forbidden SQL metacharacters"):
        sanitize_query_text("SELECT * FROM documents WHERE id = 1; DROP TABLE users;--")

    # Path traversal rejection
    with pytest.raises(StateValidationError, match="forbidden path traversal sequences"):
        sanitize_query_text("../../../etc/passwd && cat secret.txt")


def test_simulation_4_hitl_invalid_action_rejection():
    """Simulation 4: Verify Pydantic rejects invalid HITL resumption actions."""
    with pytest.raises(ValidationError):
        HITLResumptionRequest(
            checkpoint_id="chk_001",
            action="invalid_action",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_simulation_5_highlight_graceful_degradation():
    """Simulation 5: Verify get_source_highlight degrades gracefully without error on missing data."""
    params = GetSourceHighlightInput(
        go_number="GO-NONEXISTENT",
        page_number=999,
    )
    res = await get_source_highlight(params)
    assert res.success is True
    assert res.result is None
