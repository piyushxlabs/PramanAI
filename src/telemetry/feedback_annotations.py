"""Section 7a Feedback-to-Telemetry Annotation Pipeline for ShasanAI.

Captures officer feedback, citation accuracy flags, and human verification outcomes
as structured Langfuse scores attached to the relevant trace and session.
"""

import logging
import os
from typing import Any, Literal
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("shasanai.feedback_annotations")

LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

_langfuse_client: Any = None


def get_langfuse_client() -> Any:
    """Returns a connected Langfuse client singleton, or None if unconfigured/offline."""
    global _langfuse_client
    if _langfuse_client is None and LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        try:
            from langfuse import Langfuse
            _langfuse_client = Langfuse(
                public_key=LANGFUSE_PUBLIC_KEY,
                secret_key=LANGFUSE_SECRET_KEY,
                host=LANGFUSE_HOST,
            )
        except Exception as exc:
            logger.warning(f"Langfuse client initialization failed: {exc!s}")
            _langfuse_client = None
    return _langfuse_client


def record_officer_feedback(
    session_id: str,
    trace_id: str | None = None,
    feedback_value: bool = True,
    comment: str | None = None,
) -> dict[str, Any]:
    """Records officer thumbs up/down as a BOOLEAN score on the turn's trace."""
    score_payload = {
        "name": "officer_feedback",
        "value": 1.0 if feedback_value else 0.0,
        "data_type": "BOOLEAN",
        "session_id": session_id,
        "trace_id": trace_id,
        "comment": comment,
    }
    client = get_langfuse_client()
    if client:
        try:
            client.create_score(
                name="officer_feedback",
                value=1.0 if feedback_value else 0.0,
                data_type="BOOLEAN",
                session_id=session_id,
                trace_id=trace_id,
                comment=comment,
            )
            client.flush()
        except Exception as exc:
            logger.warning(f"Failed to export officer_feedback score to Langfuse: {exc!s}")
    return score_payload


def record_citation_accuracy(
    session_id: str,
    go_number: str,
    page_number: int,
    trace_id: str | None = None,
    is_accurate: bool = False,
    comment: str | None = None,
) -> dict[str, Any]:
    """Records per-citation flag as a CATEGORICAL score on the retrieval observation."""
    score_value = "correct" if is_accurate else "incorrect"
    details = f"GO: {go_number} (Page {page_number})"
    full_comment = f"{details} | {comment}" if comment else details

    score_payload = {
        "name": "citation_accuracy",
        "value": score_value,
        "data_type": "CATEGORICAL",
        "session_id": session_id,
        "trace_id": trace_id,
        "comment": full_comment,
    }
    client = get_langfuse_client()
    if client:
        try:
            client.create_score(
                name="citation_accuracy",
                value=score_value,
                data_type="CATEGORICAL",
                session_id=session_id,
                trace_id=trace_id,
                comment=full_comment,
            )
            client.flush()
        except Exception as exc:
            logger.warning(f"Failed to export citation_accuracy score to Langfuse: {exc!s}")
    return score_payload


def record_human_verification_outcome(
    session_id: str,
    outcome: Literal["approved", "approved_with_resolution", "denied"],
    trace_id: str | None = None,
    resolved_go_number: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Records human verification approval/denial outcome as a CATEGORICAL score."""
    comment_parts = []
    if resolved_go_number:
        comment_parts.append(f"Resolved GO: {resolved_go_number}")
    if reason:
        comment_parts.append(f"Reason: {reason}")
    comment_str = " | ".join(comment_parts) if comment_parts else None

    score_payload = {
        "name": "human_verification_outcome",
        "value": outcome,
        "data_type": "CATEGORICAL",
        "session_id": session_id,
        "trace_id": trace_id,
        "comment": comment_str,
    }
    client = get_langfuse_client()
    if client:
        try:
            client.create_score(
                name="human_verification_outcome",
                value=outcome,
                data_type="CATEGORICAL",
                session_id=session_id,
                trace_id=trace_id,
                comment=comment_str,
            )
            client.flush()
        except Exception as exc:
            logger.warning(f"Failed to export human_verification_outcome score to Langfuse: {exc!s}")
    return score_payload
