import json
import re
from typing import Any
from src.state.schema import StateSchema
from src.agents.nodes.node7_citation_integrity import extract_clean_verifiable_text


def clean_ui_markdown(raw_payload: Any) -> str:
    """Extracts and formats clean Markdown for UI streaming, stripping JSON wrappers."""
    return extract_clean_verifiable_text(raw_payload)


async def node9_response_delivery(state: StateSchema) -> dict[str, Any]:
    """Delivers sanitized, single-instance markdown response to the client."""
    raw_answer = state.get("answer_markdown") or ""
    clean_text = extract_clean_verifiable_text(raw_answer)

    # Unconditionally clear refusal flags when valid response exists with confidence >= 0.85 or citations exist
    confidence = float(state.get("confidence_score") or 0.0)
    has_citations = len(state.get("citations", [])) > 0
    is_valid = (confidence >= 0.85 or has_citations) and len(clean_text) > 20

    return {
        "answer_markdown": clean_text,
        "graceful_refusal": False if is_valid else bool(state.get("graceful_refusal", False)),
        "out_of_scope_notice": False if is_valid else bool(state.get("out_of_scope_notice", False)),
        "confidence_score": confidence,
        "citations": state.get("citations", []),
    }
