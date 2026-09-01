"""State reducers for ShasanAI 17-field StateSchema.

Reducers enforce exact state evolution semantics as specified in AGENT_ORCHESTRATION_BLUEPRINT.md:
- immutable-after-init: raises StateValidationError if mutated post-initialization.
- append-only: appends items without dropping historical records.
- merge-by-key: merges citations on (go_number, page_number) preventing duplicates.
- replace-on-new-turn: cleanly replaces working retrieval state on each turn.
- last-write-wins: updates field with latest incoming value.
"""

from typing import Any, Sequence, TypeVar
from pydantic import BaseModel

T = TypeVar("T")


class AgentError(Exception):
    """Base exception for all agent runtime errors."""
    pass


class StateValidationError(AgentError):
    """Raised when an illegal state mutation or validation failure occurs."""
    pass


class ToolExecutionError(AgentError):
    """Raised when a tool execution fails permanently or after retry exhaustion."""
    pass


class ApprovalTimeoutError(AgentError):
    """Raised when human verification approval times out."""
    pass


class ScopeViolationError(AgentError):
    """Raised when an operation attempts to bypass scope boundaries."""
    pass


def immutable_reducer(current: T | None, update: T | None) -> T:
    """Immutable-after-init reducer.
    
    Permits initialization when current is None or empty.
    Raises StateValidationError if a mutation to a different value is attempted.
    """
    if current is None or current == "":
        if update is None or update == "":
            raise StateValidationError("Cannot initialize immutable field with None or empty value")
        return update
    if update is None or update == "" or update == current:
        return current

    # Handle dict vs Pydantic model serialization equivalence
    c_val = current.model_dump() if hasattr(current, "model_dump") else current
    u_val = update.model_dump() if hasattr(update, "model_dump") else update
    if c_val == u_val:
        return current

    raise StateValidationError(
        f"Illegal mutation attempted on immutable field. Current: {current!r}, Attempted: {update!r}"
    )


def append_only_reducer(current: Sequence[T] | None, update: Sequence[T] | T | None) -> list[T]:
    """Append-only reducer for list accumulation (e.g. message_history, conflict_flags, error_logs)."""
    result: list[T] = list(current) if current is not None else []
    if update is None:
        return result
    if isinstance(update, list):
        result.extend(update)
    elif isinstance(update, (tuple, set)):
        result.extend(list(update))
    else:
        result.append(update)  # type: ignore[arg-type]
    return result


def _extract_citation_key(item: Any) -> tuple[str, int]:
    if isinstance(item, dict):
        return (str(item.get("go_number", "")), int(item.get("page_number", 0)))
    return (str(getattr(item, "go_number", "")), int(getattr(item, "page_number", 0)))


def merge_by_citation_key_reducer(
    current: Sequence[Any] | None, update: Sequence[Any] | Any | None
) -> list[Any]:
    """Merge-by-key reducer for candidate citations indexed on (go_number, page_number)."""
    merged: dict[tuple[str, int], Any] = {}
    if current is not None:
        for item in current:
            key = _extract_citation_key(item)
            merged[key] = item

    if update is not None:
        items_to_add = update if isinstance(update, (list, tuple)) else [update]
        for item in items_to_add:
            key = _extract_citation_key(item)
            merged[key] = item

    return list(merged.values())


def replace_on_new_turn_reducer(current: T | None, update: T | None) -> T:
    """Replace-on-new-turn reducer for turn-scoped working state (e.g. retrieved_passages)."""
    if update is not None:
        return update
    return current if current is not None else []  # type: ignore[return-value]


def last_write_wins_reducer(current: T | None, update: T | None) -> T:
    """Last-write-wins reducer."""
    if update is not None:
        return update
    return current  # type: ignore[return-value]
