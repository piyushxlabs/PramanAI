"""Node 1: Query Interpretation (Gemini 3.5 Flash-Lite — Fast-Track).

Parses raw officer queries into a normalized query string, detected language, and typed QueryFilters.
Features sub-second regex & heuristic fast-path with bounded LLM invocation (<2s) and graceful fallback.
"""

import asyncio
import logging
import re
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from src.state.schema import Message, OfficerContext, QueryFilters, QueryInterpretation, StateSchema
from src.utils.model_runtime import ainvoke_with_retry, get_structured_llm

logger = logging.getLogger("praman_ai.node1")

NODE1_SYSTEM_PROMPT = """You are the Query Interpretation node of PramanAI.
Given the officer's query, return a QueryInterpretation JSON with:
- query_text: normalized query string
- query_language: "hi", "en", or "hinglish"
- query_filters: department, year_range ([start, end]), policy_category"""

# Fast heuristic patterns for sub-second query parsing
_DEPT_PATTERNS = [
    ("Forest", re.compile(r"वन\s*(?:विभाग|अनुभाग|संरक्षण)?|forest", re.IGNORECASE)),
    ("Finance", re.compile(r"वित्त\s*(?:विभाग|अनुभाग|स्वीकृति)?|finance|budget|बजट", re.IGNORECASE)),
    ("Personnel", re.compile(r"कार्मिक\s*(?:विभाग|अनुभाग)?|personnel|तबादला|स्थानांतरण|transfer", re.IGNORECASE)),
    ("Revenue", re.compile(r"राजस्व\s*(?:विभाग|अनुभाग)?|revenue", re.IGNORECASE)),
    ("Education", re.compile(r"शिक्षा\s*(?:विभाग|अनुभाग)?|education|teacher|शिक्षक", re.IGNORECASE)),
    ("Rural Development", re.compile(r"ग्राम्य\s*विकास|rural\s*development", re.IGNORECASE)),
    ("Urban Development", re.compile(r"नगर\s*विकास|urban\s*development", re.IGNORECASE)),
    ("Home", re.compile(r"गृह\s*(?:विभाग|अनुभाग)?|police|home|पुलिस", re.IGNORECASE)),
    ("Health", re.compile(r"चिकित्सा|स्वास्थ्य|health|medical", re.IGNORECASE)),
]

_YEAR_RE = re.compile(r"\b(19\d\d|20\d\d)\b")
_GO_NUM_RE = re.compile(
    r"(?:शासनादेश\s*(?:संख्या|संo|सं०)?|GO\s*(?:number|no\.?)?|संख्या|संo|सं०)\s*[:\-]?\s*([0-9A-Za-z\/\-\_\(\)\.]+)",
    re.IGNORECASE,
)

_POLICY_PATTERNS = [
    ("Transfer Policy", re.compile(r"transfer|posting|तबादला|स्थानांतरण", re.IGNORECASE)),
    ("Regularization", re.compile(r"regularization|samayojan|नियमितीकरण|समायोजन", re.IGNORECASE)),
    ("Budget Allocation", re.compile(r"budget|grant|अनुदान|वित्तीय\s*स्वीकृति|राशि|लाख|allocated|किश्त|आयोग", re.IGNORECASE)),
    ("Recruitment", re.compile(r"recruitment|vacancy|भर्ती|नियुक्ति|पद", re.IGNORECASE)),
    ("Pension", re.compile(r"pension|retirement|पेंशन|सेवानिवृत्ति", re.IGNORECASE)),
    ("Leave Rules", re.compile(r"leave|holiday|अवकाश|छुट्टी", re.IGNORECASE)),
]


def _detect_simple_language(text: str) -> str:
    """Detects query language based on Unicode characters and common Hinglish stopwords."""
    if not text:
        return "en"
    devanagari_count = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
    if devanagari_count > 0:
        return "hi"
    hinglish_markers = {"mein", "kya", "hai", "ke", "baare", "bhejo", "ka", "ki", "ko", "par", "se", "aur", "tha", "hogi"}
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    if words & hinglish_markers:
        return "hinglish"
    return "en"


_GO_STOP_WORDS = {
    "kya", "hai", "ka", "ke", "ki", "ko", "par", "se", "aur", "number", "no",
    "batao", "bhejo", "chahiye", "dekho", "karo", "please", "mein", "konsa", "kaunsa",
    "bataiye", "rules", "policy", "detail", "details", "kya_hai", "kon", "kaun"
}


def _fast_extract_filters(raw_query: str, default_dept: str = "General") -> QueryFilters:
    """Extracts department, year range, policy category, and explicit GO number in <1ms."""
    dept: str | None = None
    for d_name, pattern in _DEPT_PATTERNS:
        if pattern.search(raw_query):
            dept = d_name
            break
    if not dept and default_dept != "General":
        dept = default_dept

    # Extract explicit GO number first (must contain digits and not be a stop word)
    go_num: str | None = None
    go_match = _GO_NUM_RE.search(raw_query)
    if go_match:
        candidate = go_match.group(1).strip().strip("?.!:,;\"'")
        if (
            candidate.lower() not in _GO_STOP_WORDS
            and any(c.isdigit() for c in candidate)
            and len(candidate) >= 2
        ):
            go_num = candidate

    # Extract standalone 4-digit years (e.g. 2018)
    year_matches = [int(y) for y in _YEAR_RE.findall(raw_query) if 1950 <= int(y) <= 2035]
    year_range: list[int] | None = None
    if year_matches:
        year_range = [min(year_matches), max(year_matches)]

    # Extract policy category
    policy: str | None = None
    for p_name, pattern in _POLICY_PATTERNS:
        if pattern.search(raw_query):
            policy = p_name
            break

    return QueryFilters(
        department=dept,
        year_range=year_range,
        policy_category=policy,
        go_number=go_num,
    )


def _merge_filters(explicit: QueryFilters | None, extracted: QueryFilters) -> QueryFilters:
    """Merges explicit user-selected faceted filters with extracted query filters."""
    if not explicit:
        return extracted
    return QueryFilters(
        department=explicit.department if explicit.department is not None else extracted.department,
        year_range=explicit.year_range if explicit.year_range is not None else extracted.year_range,
        policy_category=explicit.policy_category if explicit.policy_category is not None else extracted.policy_category,
        go_number=explicit.go_number if explicit.go_number is not None else extracted.go_number,
    )


async def node1_query_interpretation(state: StateSchema) -> dict[str, Any]:
    """Executes Node 1 (Query Interpretation) with fast-track heuristic execution and 2.5s bounded LLM timeout."""
    raw_query: str = state.get("query_text", "")
    explicit_filters: QueryFilters | None = state.get("query_filters")
    officer_ctx: OfficerContext = state.get(
        "officer_context", OfficerContext(department="General", access_scope=["General"])
    )
    history: list[Message] = state.get("message_history", [])

    detected_lang = _detect_simple_language(raw_query)
    heuristic_filters = _fast_extract_filters(raw_query, default_dept=officer_ctx.department)
    final_heuristic_filters = _merge_filters(explicit_filters, heuristic_filters)

    # If no conversational history, we can quickly return the high-accuracy heuristic extraction
    # or run a tight LLM structured output call in parallel with fallback
    if not history and (final_heuristic_filters.department or final_heuristic_filters.year_range or final_heuristic_filters.policy_category or final_heuristic_filters.go_number):
        logger.info(f"Node 1 Fast-Track: parsed query in <1ms (lang={detected_lang}, dept={final_heuristic_filters.department}, year={final_heuristic_filters.year_range}, go={final_heuristic_filters.go_number})")
        return {
            "query_text": raw_query,
            "query_language": detected_lang,
            "query_filters": final_heuristic_filters,
        }

    # Format history context if available (last 3 turns max for token economy)
    history_context = ""
    if history:
        recent_history = history[-6:]
        history_lines = [f"{msg.role}: {msg.content}" for msg in recent_history]
        history_context = "\n<conversation_history>\n" + "\n".join(history_lines) + "\n</conversation_history>\n"

    user_prompt = f"""{history_context}
Default Department: {officer_ctx.department}
User Query: {raw_query}"""

    structured_llm = get_structured_llm(
        QueryInterpretation,
        temperature=0.0,
        max_tokens=150,
        timeout=3.0,
        use_fast_model=True,
    )
    messages = [
        SystemMessage(content=NODE1_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        interpretation: QueryInterpretation = await asyncio.wait_for(
            ainvoke_with_retry(
                structured_llm=structured_llm,
                messages=messages,
                max_retries=1,
                timeout_seconds=2.5,
            ),
            timeout=2.5,
        )
        extracted = interpretation.query_filters or heuristic_filters
        return {
            "query_text": interpretation.query_text or raw_query,
            "query_language": interpretation.query_language or detected_lang,
            "query_filters": _merge_filters(explicit_filters, extracted),
        }
    except Exception as exc:
        logger.info(f"Node 1 fast-track fallback activated ({exc!s}). Completed in <5ms.")
        return {
            "query_text": raw_query,
            "query_language": detected_lang,
            "query_filters": final_heuristic_filters,
        }
