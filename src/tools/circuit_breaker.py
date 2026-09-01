"""Circuit breaker and exponential backoff retry manager for external MCP dependencies.

Implements AGENT_LOGIC_SPEC.md Section 9:
- Max retries: 3 attempts.
- Exponential backoff: 1s -> 2s -> 4s.
- Circuit breaker: trips to OPEN after 3 consecutive transient failures.
"""

import asyncio
import logging
from typing import Any, Callable, Coroutine, TypeVar
from src.state.reducers import ToolExecutionError

logger = logging.getLogger("shasanai.circuit_breaker")
T = TypeVar("T")


class CircuitBreaker:
    """Session-scoped circuit breaker tracking transient failures."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count: int = 0
        self.state: str = "CLOSED"  # CLOSED | OPEN

    def record_success(self) -> None:
        """Resets consecutive failure counter on success."""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self) -> None:
        """Increments failure counter and trips circuit if threshold reached."""
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker tripped to OPEN after {self.failure_count} consecutive failures"
            )

    def is_available(self) -> bool:
        """Returns True if circuit breaker is CLOSED and requests can proceed."""
        return self.state == "CLOSED"


# Default circuit breaker instances for the 3 tools
RETRIEVAL_BREAKER = CircuitBreaker(failure_threshold=3)
SUPERSESSION_BREAKER = CircuitBreaker(failure_threshold=3)
HIGHLIGHT_BREAKER = CircuitBreaker(failure_threshold=3)


async def execute_with_retry(
    operation: Callable[[], Coroutine[Any, Any, T]],
    breaker: CircuitBreaker,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    operation_name: str = "mcp_operation",
) -> T:
    """Executes an async operation with exponential backoff and circuit breaker protection.
    
    Args:
        operation: Async callable to execute.
        breaker: Associated CircuitBreaker instance.
        max_retries: Max retry attempts (default 3).
        initial_backoff: Initial sleep duration in seconds (1.0s).
        backoff_factor: Multiplier for backoff (2.0 -> 1s, 2s, 4s).
        operation_name: Logging name for operation.
        
    Returns:
        Result T of the operation.
        
    Raises:
        ToolExecutionError: On permanent failure or retry exhaustion.
    """
    if not breaker.is_available():
        raise ToolExecutionError(
            f"Circuit breaker is OPEN for {operation_name} due to repeated consecutive failures. Operation rejected."
        )

    last_error: Exception | None = None
    delay = initial_backoff

    for attempt in range(1, max_retries + 1):
        try:
            result = await operation()
            breaker.record_success()
            return result
        except Exception as exc:
            last_error = exc
            logger.warning(
                f"{operation_name} attempt {attempt}/{max_retries} failed: {exc!s}"
            )
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= backoff_factor

    # All retries exhausted
    breaker.record_failure()
    raise ToolExecutionError(
        f"{operation_name} failed after {max_retries} attempts. Last error: {last_error!s}"
    ) from last_error
