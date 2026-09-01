"""Node 5: Human Verification Interrupt Gate (LangGraph HITL).

Pauses graph execution when confidence is low (< 0.6), conflicting orders are detected, or citizen PII is present.
Enforces the Approve / Approve-with-Edit / Deny resumption contract per INTERFACE_OBSERVABILITY_SYSTEM.md Section 5.
"""

from datetime import datetime
from typing import Any
from langgraph.types import interrupt
from src.state.schema import ApprovalState, Citation, ConflictRecord, ErrorRecord, StateSchema


async def node5_human_verification(state: StateSchema) -> dict[str, Any]:
    """Executes Node 5 (Human Verification Interrupt Gate).
    
    1. Prepares verification card payload.
    2. Calls LangGraph interrupt() to pause and persist state to PostgreSQL checkpointer.
    3. Processes officer resumption payload (Approve / Approve-with-Edit / Deny).
    """
    confidence: float = state.get("confidence_score", 0.0)
    conflicts: list[ConflictRecord] = state.get("conflict_flags", [])
    citations: list[Citation] = state.get("candidate_citations", [])

    # Determine primary interrupt trigger
    if conflicts:
        reason = "Multiple conflicting government orders detected without clear supersession linkage."
    elif confidence < 0.85:
        reason = f"Grounding confidence score ({confidence:.2f}) is below the minimum threshold (0.85)."
    else:
        reason = "Human verification required prior to synthesis."

    interrupt_payload = {
        "interrupt_type": "human_verification_required",
        "reason": reason,
        "confidence_score": confidence,
        "conflict_flags": [c.model_dump() for c in conflicts],
        "candidate_citations": [c.model_dump() for c in citations],
        "allowed_actions": ["approve", "approve_with_edit", "deny"],
    }

    # Pause execution and await officer resumption command
    resumption_input: dict[str, Any] = interrupt(interrupt_payload)

    # Process resumption payload
    action = resumption_input.get("action", "deny")
    action_val = "approve" if action in ["approve", "approve_with_edit"] else "deny"
    decision_reason = resumption_input.get("reason", "Officer completed review")

    approval_state = ApprovalState(
        action=action_val,
        checkpoint_id=resumption_input.get("checkpoint_id"),
        resolved_go_number=resumption_input.get("resolved_go_number"),
        reason=decision_reason,
    )

    if action == "approve":
        return {
            "human_verification": approval_state,
            "graceful_refusal": False,
        }

    if action == "approve_with_edit":
        # If officer supplied specific edited citations to synthesize
        edited_cits_raw = resumption_input.get("modified_citations")
        if edited_cits_raw:
            edited_citations = [Citation(**c) if isinstance(c, dict) else c for c in edited_cits_raw]
            return {
                "human_verification": approval_state,
                "candidate_citations": edited_citations,
                "graceful_refusal": False,
            }
        return {
            "human_verification": approval_state,
            "graceful_refusal": False,
        }

    # action == "deny"
    err = ErrorRecord(
        node="node5_human_verification",
        error_type="HumanVerificationDenied",
        message=f"Officer denied retrieval grounding: {decision_reason}",
        timestamp=datetime.now().isoformat(),
    )
    return {
        "human_verification": approval_state,
        "graceful_refusal": True,
        "error_logs": [err],
    }
