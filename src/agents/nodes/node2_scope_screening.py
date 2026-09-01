"""Node 2: Out-of-Scope Screening & Google Model Armor Security Gate.

Enforces exact security gate order per AGENT_LOGIC_SPEC.md and scope-screening-and-safety-gate-order.md:
1. Google Model Armor / Gemma 2 Guardrail -> Immediate refusal + security error log.
2. Deterministic Out-of-Scope Pre-Screening (financial_disbursement, grievance, order_drafting, policy_opinion) -> Refusal + ErrorRecord.
3. Structured Scope Screening on Gemini 3.5 Flash-Lite (temperature=0.0).
4. In-Scope -> Proceeds to Node 3 (Retrieval Invocation).
"""

import asyncio
from datetime import datetime
import logging
import re
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from src.security.model_armor import evaluate_security_armor
from src.state.schema import ErrorRecord, QueryFilters, ScopeScreenDecision, StateSchema
from src.utils.model_runtime import ainvoke_with_retry, get_structured_llm

logger = logging.getLogger("agents.node2_scope_screening")

NODE2_SYSTEM_PROMPT = """<identity_and_role>
You are the Scope Screening node of the GO-Retrieval & Citation Agent, running as Node 2 of the LangGraph StateGraph on qwen2.5:7b (local Ollama runtime).
Your purpose is to determine whether the normalized query falls within this agent's defined scope before any retrieval occurs.
</identity_and_role>

<primary_objective>
Determine whether the query concerns: (a) financial disbursement or treasury processing, (b) citizen grievance handling, (c) a request to draft a legally binding executive order, or (d) a request for subjective opinion on government policy or personnel. These are the agent's explicit out-of-scope categories. Think step-by-step: identify what the officer is actually asking for, then check it against each category before deciding.
</primary_objective>

<context_and_state_access>
You have read access to:
- query_text: str — normalized query from Node 1
- query_language: Literal["hi","en","hinglish"]
- query_filters: QueryFilters
You may write to the following fields, using their declared reducer behavior:
- graceful_refusal: bool — reducer: last-write-wins — true only if out-of-scope
- error_logs: list[ErrorRecord] — reducer: append-only — the specific out-of-scope reason, if refused
</context_and_state_access>

<available_tools_and_triggers>
None. — mode: Structured Output only.
</available_tools_and_triggers>

<hard_constraints_and_prohibitions>
You must NEVER:
- Attempt to partially answer an out-of-scope request "just to be helpful."
- Retrieve or reference any document content at this stage.
- Classify a legitimate citation-lookup request as out-of-scope.
</hard_constraints_and_prohibitions>

<few_shot_examples>
Example 1:
Input: query_text = "Uttarakhand mein contractual teachers ke regularization ka kaunsa GO hai?"
Thought: This asks for a citation to a specific government order on a personnel-regularization policy. It is a factual lookup, not a request for a subjective opinion, financial calculation, grievance processing, or order-drafting.
Tool Call: None.
Response: {"in_scope": true, "category": null, "reason": null}

Example 2:
Input: query_text = "Is officer ke pension arrears ka disbursement calculate karo aur treasury mein bhej do."
Thought: This explicitly asks for a financial-disbursement calculation and a treasury transmission action — both are explicit out-of-scope items.
Tool Call: None.
Response: {"in_scope": false, "category": "financial_disbursement", "reason": "Query requests calculation and treasury transmission of a pension disbursement, which is explicitly out of scope for this read-only citation agent."}

Example 3:
Input: query_text = "Mera gaon ka rasta kharab hai, complaint register karo aur action lo."
Thought: Citizen grievance ticketing and administrative enforcement request.
Response: {"in_scope": false, "category": "grievance", "reason": "Citizen grievance redressal and complaint registration are outside the scope of this GO-retrieval agent."}

Example 4:
Input: query_text = "Ek naya government order draft karo jisme transfer rules badle ja rahe hain."
Thought: Executive order drafting request.
Response: {"in_scope": false, "category": "order_drafting", "reason": "Drafting legally binding executive orders is strictly prohibited for this read-only citation system."}

Example 5:
Input: query_text = "Kya Uttarakhand ki nayi tourism policy achhi hai ya buri?"
Thought: Request for subjective opinion/editorial assessment on government policy.
Response: {"in_scope": false, "category": "policy_opinion", "reason": "Subjective policy evaluations and political commentary are strictly out of scope."}
</few_shot_examples>

<output_formatting_rules>
Respond only with a single ScopeScreenDecision structured-output JSON object matching the schema in Section 5. No prose.
</output_formatting_rules>
"""

# Prompt injection heuristics
PROMPT_INJECTION_HEURISTICS = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions|system\s+prompt|reveal\s+(internal\s+)?prompt|override\s+safety|developer\s+mode|jailbreak|drop\s+table|delete\s+from|bypass\s+guardrails)",
    re.IGNORECASE,
)

# Deterministic out-of-scope intent patterns
OUT_OF_SCOPE_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "financial_disbursement",
        re.compile(
            r"(disburse|disbursement|treasury|arreas|arrears|bhej\s+do|calculate\s+(?:pension|salary|gratuity|payout)|pension\s+calculation|salary\s+calculation|खाते\s*में\s*(?:डाल|भेज)|कोषागार\s*में\s*भेज|भुगतान\s*कर|disburse\s+karo)",
            re.IGNORECASE,
        ),
        "Query requests financial disbursement calculation or treasury processing, which is strictly out of scope.",
    ),
    (
        "grievance",
        re.compile(
            r"(register\s+(?:my\s+|a\s+|the\s+)?(?:complaint|grievance|shikayat)|complaint\s+(?:register|karo|darj|lodged?)|complaint|grievance|shikayat|शिकायत\s*(?:दर्ज|कर)|action\s*lo|कार्रवाई\s*करो|सड़क\s*खराब|रास्ता\s*खराब|pani\s+nahi\s+aa\s+raha)",
            re.IGNORECASE,
        ),
        "Citizen grievance ticketing and complaint registration are outside the scope of this GO retrieval agent.",
    ),
    (
        "order_drafting",
        re.compile(
            r"(draft\s+(?:a\s+)?(?:new\s+)?(?:executive\s+|administrative\s+|government\s+|state\s+)*(?:order|go|policy|circular|notification)|draft\s+(?:an?\s+)?(?:order|go|policy|circular|notification)|नया\s*आदेश\s*(?:बना|तैयार\s*कर|ड्राफ्ट)|शासनादेश\s*ड्राफ्ट|draft\s+karo)",
            re.IGNORECASE,
        ),
        "Drafting legally binding executive orders or notifications is outside system capabilities.",
    ),
    (
        "policy_opinion",
        re.compile(
            r"(kya\s+(?:.*)\s+(?:achhi|buri|theek|sahi)\s+hai|personal\s+opinion|your\s+opinion|opinion|critique|अच्छी\s*है\s*या\s*बुरी|राय\s*(?:दीजिये|दो)|समीक्षा\s*करो|is\s+(?:the\s+)?(?:new\s+)?(?:state\s+)?(?:policy|scheme|order)\s+(?:good|bad|beneficial|detrimental)|beneficial\s+or\s+detrimental|is\s+policy\s+(?:good|bad))",
            re.IGNORECASE,
        ),
        "Subjective policy assessments, opinions, and political commentary are strictly out of scope.",
    ),
]


async def node2_scope_screening(state: StateSchema) -> dict[str, Any]:
    """Executes Node 2 (Scope Screening & Security Gate).
    
    Order:
    1. Prompt Injection Pre-Screening -> Immediate refusal + security error log.
    2. Deterministic Out-of-Scope Pre-Screening -> Immediate refusal + category record.
    3. Structured LLM Scope Screening via qwen2.5:7b (temperature=0.0).
    4. In-Scope -> Proceeds to Node 3 (Retrieval Invocation).
    """
    query_text: str = state.get("query_text", "")
    query_lang: str = state.get("query_language", "en")
    query_filters: QueryFilters = state.get("query_filters") or QueryFilters()

    # Gate 1: Google Model Armor & Gemma 2 Security Shield
    is_safe, armor_reason = await evaluate_security_armor(query_text)
    if not is_safe:
        logger.warning("Node 2: Security shield triggered for query: '%s' (Reason: %s)", query_text[:80], armor_reason)
        error_rec = ErrorRecord(
            node="node2_scope_screening",
            error_type="PromptInjectionDetected",
            message=armor_reason or f"Prompt injection / security attack detected: '{query_text[:100]}'",
            timestamp=datetime.now().isoformat(),
        )
        return {
            "graceful_refusal": True,
            "error_logs": [error_rec],
        }

    # Gate 2: Deterministic Out-of-Scope Pre-Screening
    for category, pattern, reason_msg in OUT_OF_SCOPE_PATTERNS:
        if pattern.search(query_text):
            logger.info("Node 2: Query deterministically classified as out-of-scope (%s)", category)
            error_rec = ErrorRecord(
                node="node2_scope_screening",
                error_type="OutOfScopeQuery",
                message=f"Category: {category} - {reason_msg}",
                timestamp=datetime.now().isoformat(),
            )
            return {
                "graceful_refusal": True,
                "error_logs": [error_rec],
            }

    # Gate 3: LLM Structured Scope Decision
    user_prompt = f"""<query_text>
{query_text}
</query_text>
<query_language>
{query_lang}
</query_language>
<query_filters>
Department: {query_filters.department}
Year Range: {query_filters.year_range}
Category: {query_filters.policy_category}
</query_filters>
"""

    structured_llm = get_structured_llm(ScopeScreenDecision, temperature=0.0)
    messages = [
        SystemMessage(content=NODE2_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        decision: ScopeScreenDecision = await asyncio.wait_for(
            ainvoke_with_retry(
                structured_llm=structured_llm,
                messages=messages,
                max_retries=1,
                timeout_seconds=5.0,
            ),
            timeout=6.0,
        )

        if not decision.in_scope:
            reason_msg = decision.reason or f"Query classified as out-of-scope category: {decision.category}"
            error_rec = ErrorRecord(
                node="node2_scope_screening",
                error_type="OutOfScopeQuery",
                message=f"Category: {decision.category} - {reason_msg}",
                timestamp=datetime.now().isoformat(),
            )
            return {
                "graceful_refusal": True,
                "error_logs": [error_rec],
            }
    except Exception as exc:
        logger.warning("Node 2 LLM Scope Screening timed out/failed (%s); verifying deterministic safety", exc)
        # Fail-safe check
        for category, pattern, reason_msg in OUT_OF_SCOPE_PATTERNS:
            if pattern.search(query_text):
                error_rec = ErrorRecord(
                    node="node2_scope_screening",
                    error_type="OutOfScopeQuery",
                    message=f"Category: {category} - {reason_msg}",
                    timestamp=datetime.now().isoformat(),
                )
                return {
                    "graceful_refusal": True,
                    "error_logs": [error_rec],
                }

    # Gate 4: In-Scope -> Pass to Node 3
    return {
        "graceful_refusal": False,
    }
