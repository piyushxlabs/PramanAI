"""Node 8: Refusal / Redirect Terminal Node (Deterministic).

Generates standardized officer-facing refusal notices for out-of-scope requests or security violations.
Sets graceful_refusal: True and confidence_score: 0.0 with empty citations.
"""

from typing import Any
from src.state.schema import ErrorRecord, StateSchema

REFUSAL_TEMPLATES: dict[str, str] = {
    "financial_disbursement": (
        "**सूचना / Notice (Out of Scope — Financial Processing):**\n\n"
        "ShasanAI is a read-only Government Order (GO) retrieval and citation system. "
        "Financial disbursement calculations, treasury payment processing, and salary/pension computation "
        "cannot be executed through this system. Please route financial transactions through the Integrated Financial Management System (IFMS) / State Treasury portal."
    ),
    "grievance": (
        "**सूचना / Notice (Out of Scope — Citizen Grievance):**\n\n"
        "This system does not process public grievances or register citizen complaints. "
        "Please submit grievances through the official CM Helpline / Samadhan portal (samadhan.uk.gov.in) or your designated District Grievance Officer."
    ),
    "order_drafting": (
        "**सूचना / Notice (Out of Scope — Order Drafting):**\n\n"
        "ShasanAI retrieves and cites existing, officially notified Government Orders. "
        "Drafting new legally binding executive orders, circulars, or notifications is outside system capabilities and requires competent administrative authority."
    ),
    "policy_opinion": (
        "**सूचना / Notice (Out of Scope — Policy Opinion):**\n\n"
        "ShasanAI provides factual, verbatim citations from indexed Government Orders and circulars. "
        "Subjective policy assessments, legal interpretations, or political commentary are not provided."
    ),
    "PromptInjectionDetected": (
        "**सुरक्षा चेतावनी / Security Notice:**\n\n"
        "The submitted request contained unauthorized prompt-override or script sequences and was halted by security guardrails. "
        "Please submit standard administrative queries regarding indexed Government Orders."
    ),
    "HumanVerificationDenied": (
        "**सत्यापन अस्वीकृत / Verification Denied:**\n\n"
        "अधिकारी द्वारा सत्यापन अस्वीकृत कर दिया गया है। प्रक्रिया रोक दी गई है।\n\n"
        "(Human verification was denied by the reviewing officer. Process halted.)"
    ),
    "default": (
        "**सूचना / Notice (Unable to Process):**\n\n"
        "The requested query falls outside the operational scope of the Uttarakhand Government Order Retrieval System. "
        "ShasanAI strictly retrieves and cites published Government Orders, notifications, and administrative circulars."
    ),
}


async def node8_refusal_redirect(state: StateSchema) -> dict[str, Any]:
    """Executes Node 8 (Refusal/Redirect)."""
    error_logs: list[ErrorRecord] = state.get("error_logs", [])

    matched_template = REFUSAL_TEMPLATES["default"]

    for err in reversed(error_logs):
        if err.error_type == "HumanVerificationDenied":
            matched_template = REFUSAL_TEMPLATES["HumanVerificationDenied"]
            break
        if err.error_type == "PromptInjectionDetected":
            matched_template = REFUSAL_TEMPLATES["PromptInjectionDetected"]
            break
        for category in ["financial_disbursement", "grievance", "order_drafting", "policy_opinion"]:
            if category in err.message.lower() or category in err.error_type.lower():
                matched_template = REFUSAL_TEMPLATES[category]
                break

    return {
        "graceful_refusal": True,
        "confidence_score": 0.0,
        "answer_markdown": matched_template,
        "citations": [],
    }
