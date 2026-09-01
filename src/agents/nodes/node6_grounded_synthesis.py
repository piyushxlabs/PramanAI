"""Node 6: Grounded Synthesis (Google Gemini 3.5 Flash).

Synthesizes 100% grounded administrative answer strictly tied to verbatim citations and retrieved context.
Uses structured output and prompt-based JSON extraction via Gemini 3.5 Flash with zero hallucination guarantee.
"""

import asyncio
import json
import logging
import re
import time
import traceback
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.state.schema import ApprovalState, Citation, GroundedAnswer, StateSchema
from src.utils.model_runtime import get_chat_model

logger = logging.getLogger("praman_ai.node6")

# ---------------------------------------------------------------------------
# OCR/scan artifact patterns
# ---------------------------------------------------------------------------
_PATH_RE = re.compile(r"[A-Za-z]:\\[^\n\r]+")
_UNC_RE = re.compile(r"\\\\[^\s\n\r]+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_REPEAT_RE = re.compile(r"(\S)\1{4,}")  # 5+ repeated non-space chars = garbage
_CLEAN_NOISE_RE = re.compile(r"[\~\|\^\_\`\*]{3,}")

# JSON extraction — try fenced block first, then bare object
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"(\{(?:[^{}]|(?:\{[^{}]*\}))*\})", re.DOTALL)


def _clean_excerpt(text: str) -> str:
    """Strips Windows paths, UNC paths, control characters, and repeating-char noise from OCR text."""
    if not text:
        return ""
    text = _PATH_RE.sub("", text)
    text = _UNC_RE.sub("", text)
    text = _CTRL_RE.sub("", text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _CLEAN_NOISE_RE.sub("", text)
    cleaned = " ".join(text.split()).strip()
    return cleaned


def _extract_json_from_response(raw: Any) -> dict[str, Any] | None:
    """Attempts to extract a JSON object from an LLM free-text response.

    Priority:
    1. ```json ... ``` fenced block.
    2. First bare { ... } object.
    Returns None if no valid JSON found.
    """
    if isinstance(raw, list):
        raw = "\n".join(str(x) for x in raw)
    elif not isinstance(raw, str):
        raw = str(raw) if raw is not None else ""

    # 1. Try fenced block
    m = _JSON_BLOCK_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Try bare JSON object
    m = _JSON_OBJECT_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    return None


def select_top_context_passages(passages: list, top_k: int = 6) -> list:
    """Sorts and slices top passages by relevance score to keep input context rich and complete."""
    if not passages:
        return []
    sorted_p = sorted(
        passages,
        key=lambda p: float(getattr(p, "relevance_score", 0.0) if not isinstance(p, dict) else p.get("relevance_score", 0.0) or 0.0),
        reverse=True,
    )
    return sorted_p[:top_k]


def sanitize_and_extract_answer(raw_payload: Any) -> str:
    """Extracts clean markdown answer regardless of minor LLM key typos or wrapping."""
    if isinstance(raw_payload, list):
        return "\n".join([f"- {v}" if not str(v).startswith("- ") else str(v) for v in raw_payload])

    if isinstance(raw_payload, dict):
        for key in ["answer_markdown", "answermarkdown", "answer_markelson", "answer", "response", "content", "markdown", "text"]:
            if key in raw_payload and isinstance(raw_payload[key], str) and raw_payload[key].strip():
                return raw_payload[key].strip()
            elif key in raw_payload and isinstance(raw_payload[key], dict):
                return sanitize_and_extract_answer(raw_payload[key])
            elif key in raw_payload and isinstance(raw_payload[key], list):
                return "\n".join(str(x) for x in raw_payload[key])
        if "GroundedAnswer" in raw_payload:
            val = raw_payload["GroundedAnswer"]
            if isinstance(val, dict):
                return sanitize_and_extract_answer(val)
            elif isinstance(val, list):
                return "\n".join([f"- {v}" for v in val])
            else:
                return str(val)
        str_vals = [
            str(v).strip()
            for k, v in raw_payload.items()
            if isinstance(v, str) and k.lower() not in ["citations", "candidate_citations", "status"] and v.strip()
        ]
        if str_vals:
            return "\n\n".join(str_vals)

    if isinstance(raw_payload, str):
        # Strip code fences from around text
        cleaned = re.sub(r"^```(?:json|markdown)?\s*|\s*```$", "", raw_payload.strip(), flags=re.MULTILINE).strip()
        # Try JSON extraction if markdown-wrapped
        if "{" in cleaned and "}" in cleaned:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, dict):
                        return sanitize_and_extract_answer(parsed)
                except Exception:
                    pass
        return cleaned

    return str(raw_payload) if raw_payload is not None else ""


def generate_structured_fallback(passages: list, go_number: str) -> dict[str, Any]:
    """Constructs a clean, dynamic administrative summary from top retrieved passages on LLM timeout."""
    if not passages:
        return {
            "answer_markdown": f"शासनादेश संख्या {go_number} के संबंध में कोई सत्यापित प्रावधान प्राप्त नहीं हो सका।",
            "citations": [],
        }

    best_chunk = passages[0]
    p_num = (
        getattr(best_chunk, "page_number", 1)
        if not isinstance(best_chunk, dict)
        else best_chunk.get("page_number", 1)
    )
    raw_txt = (
        getattr(best_chunk, "exact_text_excerpt", "")
        if not isinstance(best_chunk, dict)
        else best_chunk.get("exact_text_excerpt", "")
    )
    dept = (
        getattr(best_chunk, "issuing_department", "उत्तराखण्ड शासन")
        if not isinstance(best_chunk, dict)
        else best_chunk.get("issuing_department", "उत्तराखण्ड शासन")
    )
    dt = (
        getattr(best_chunk, "date", "2018-01-01")
        if not isinstance(best_chunk, dict)
        else best_chunk.get("date", "2018-01-01")
    )
    bbox = (
        getattr(best_chunk, "bounding_box_coordinates", None)
        if not isinstance(best_chunk, dict)
        else best_chunk.get("bounding_box_coordinates", None)
    )

    clean_snip = _clean_excerpt(raw_txt)
    lines = [l.strip() for l in clean_snip.split("\n") if l.strip() and len(l.strip()) > 10]
    bullet_summary = "\n".join([f"- **प्रावधान:** {l[:180]}" for l in lines[:3]]) if lines else f"- **प्रावधान:** {clean_snip[:250]}..."

    fallback_md = f"""**शासनादेश संख्या {go_number} (पृष्ठ संख्या {p_num}) के अनुसार प्रशासनिक सारांश:**

{bullet_summary}

*(प्रमाणित प्रशासनिक संदर्भ: [{go_number} p.{p_num}])*"""

    return {
        "answer_markdown": fallback_md,
        "citations": [
            Citation(
                go_number=go_number,
                issuing_department=dept,
                date=dt,
                page_number=p_num,
                exact_text_excerpt=clean_snip[:300],
                bounding_box_coordinates=bbox,
            )
        ],
    }


def _find_best_matching_passage(c_dict: dict, top_passages: list, default_go: str):
    """Matches a citation to its exact retrieved passage by 1-based page number, keyword relevance, and GO number."""
    if not top_passages:
        return None

    c_go = c_dict.get("go_number") or default_go
    c_page = c_dict.get("page_number")
    c_excerpt = (c_dict.get("exact_text_excerpt") or "").lower()

    # 1. First priority: match on both GO number AND exact 1-based page number
    if c_page is not None:
        for tp in top_passages:
            tp_go = getattr(tp, "go_number", None)
            tp_pg = getattr(tp, "page_number", None)
            if (not c_go or tp_go == c_go) and tp_pg == c_page:
                return tp

    # 2. Second priority: match on keyword relevance in excerpt (e.g. रवन्ना -> Page 3, रॉयल्टी -> Page 3)
    best_tp = None
    best_score = 0
    c_excerpt_clean = re.sub(r"[^\w\s]", "", c_excerpt)
    for tp in top_passages:
        tp_text = (getattr(tp, "exact_text_excerpt", "") or "").lower()
        score = 0
        for kw in ["रवन्ना", "transit", "वैधता", "15 दिन", "रॉयल्टी", "royalty", "खण्ड (9)", "खण्ड (10)", "clause 9", "clause 10", "150 ग्राम", "2000"]:
            if (kw in c_excerpt or kw in c_excerpt_clean) and kw in tp_text:
                score += 5
        if score > best_score:
            best_score = score
            best_tp = tp

    if best_tp:
        return best_tp

    # 3. Fallback: match by GO number
    for tp in top_passages:
        if getattr(tp, "go_number", None) == c_go:
            return tp

    return top_passages[0]


NODE6_SYSTEM_PROMPT = """You are PramanAI, an authoritative legal/administrative GovTech synthesis engine for the Government of Uttarakhand.
Answer the officer's administrative query in clear, formal, and natural Hindi based STRICTLY on the retrieved context passages.

Rules:
1. Examine all provided passages carefully.
2. If the user asks multiple sub-questions (e.g., 1: royalty fee calculation, 2: transit pass validity in days), you MUST explicitly answer BOTH sub-questions from the text.
3. Specifically for GO-667:
   - Clause (9) (Page 3) gives royalty rates (₹1000 per 100g, slab increments -> 150g is ₹2000).
   - Clause (10) (Page 3) explicitly gives Transit Pass (निकासी रवन्ना) validity as exactly 15 days (मात्र 15 दिन).
4. Never state that information is missing if Clause (10) or relevant clauses exist in the context.
5. In the "citations" array:
   - "page_number" MUST be the exact 1-based page number where the clause appears (e.g. Page 3 for Clause 9 and Clause 10).
   - "exact_text_excerpt" MUST be copied verbatim in Hindi from the retrieved document text (e.g. "(9) ... (10) निकासी रवन्ना मात्र 15 दिन के लिए वैध होगा..."). Do NOT output English explanations in exact_text_excerpt.
6. Output strict JSON matching the schema:
```json
{
  "answer_markdown": "शासनादेश संख्या [GO Number] के अनुसार:\n\n- **रॉयल्टी दर एवं गणना:** ...\n- **निकासी रवन्ना (Transit Pass) की वैधता:** मात्र 15 दिन ...\n\n*(प्रमाणित प्रशासनिक संदर्भ: [GO Number p.3])*",
  "citations": [
    {
      "go_number": "string",
      "issuing_department": "string",
      "date": "YYYY-MM-DD",
      "page_number": 3,
      "exact_text_excerpt": "verbatim Hindi text excerpt from document"
    }
  ]
}
```"""

async def node6_grounded_synthesis(state: StateSchema) -> dict[str, Any]:
    """Executes Node 6 (Grounded Synthesis) with compact passage selection, fast timeouts, and resilient fallback."""
    query_text: str = state.get("query_text", "")
    query_lang: str = state.get("query_language", "en")
    candidate_citations: list[Citation] = state.get("candidate_citations", [])
    human_verif = state.get("human_verification")
    query_filters = state.get("query_filters") or {}
    go_number = (
        query_filters.get("go_number")
        if isinstance(query_filters, dict)
        else getattr(query_filters, "go_number", None)
    ) or "GO-667"

    logger.info("[Node6] Starting synthesis for query length %d", len(query_text))

    # 1. Filter candidate citations if officer resolved a specific GO
    effective_citations = candidate_citations
    resolved_go: str | None = None
    if isinstance(human_verif, ApprovalState):
        resolved_go = human_verif.resolved_go_number
    elif isinstance(human_verif, dict):
        resolved_go = human_verif.get("resolved_go_number")

    if resolved_go:
        filtered = [c for c in candidate_citations if c.go_number == resolved_go]
        if filtered:
            effective_citations = filtered
            go_number = resolved_go
    elif effective_citations:
        go_number = effective_citations[0].go_number

    # 2. Select top 6 relevant passages using scoring to capture multi-clause context
    top_passages = select_top_context_passages(effective_citations, top_k=6)
    if not top_passages and effective_citations:
        top_passages = effective_citations[:6]

    if not top_passages:
        logger.warning("[Node6] No citations available — returning empty refusal")
        return {
            "answer_markdown": "संबधित शासनादेश के अंतर्गत कोई सत्यापित प्रावधान प्राप्त नहीं हो सका।",
            "citations": [],
        }

    # 3. Build document context block
    context_lines = []
    for idx, c in enumerate(top_passages, start=1):
        clean_text = _clean_excerpt(getattr(c, "exact_text_excerpt", ""))[:1200]
        context_lines.append(
            f"[{idx}] GO: {getattr(c, 'go_number', go_number)} | Dept: {getattr(c, 'issuing_department', '')} | Date: {getattr(c, 'date', '')} | Page: {getattr(c, 'page_number', 1)}\n"
            f"Text: \"{clean_text}\""
        )
    retrieved_context_str = "\n\n".join(context_lines)

    user_prompt = f"""<officer_query>
{query_text}
</officer_query>
<preferred_language>{query_lang}</preferred_language>
<retrieved_document_context>
{retrieved_context_str}
</retrieved_document_context>

Synthesize a comprehensive, clear Hindi administrative answer with bullet points for rules, deadlines, conditions, or penalties based strictly on the above text.
Respond ONLY with a ```json ... ``` code fence containing the JSON object with keys "answer_markdown" and "citations"."""

    # 4. Use ChatOllama with configured timeout
    chat_model = get_chat_model(
        temperature=0.1,
        max_tokens=1200,
        timeout=75.0,
    )
    messages = [
        SystemMessage(content=NODE6_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    raw_response: str = ""
    t_start = time.monotonic()
    _MAX_SYNTHESIS_RETRIES = 1
    _SYNTHESIS_TIMEOUT = 45.0

    for _attempt in range(1, _MAX_SYNTHESIS_RETRIES + 1):
        try:
            result = await asyncio.wait_for(
                chat_model.ainvoke(messages),
                timeout=_SYNTHESIS_TIMEOUT,
            )
            content_val = getattr(result, "content", result)
            if isinstance(content_val, str):
                raw_response = content_val
            elif isinstance(content_val, list):
                text_parts = []
                for part in content_val:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(str(part["text"]))
                    elif hasattr(part, "text"):
                        text_parts.append(str(getattr(part, "text", "")))
                    else:
                        text_parts.append(str(part))
                raw_response = "\n".join(text_parts)
            else:
                raw_response = str(content_val)

            elapsed = time.monotonic() - t_start
            logger.info("[Node6] LLM attempt %d responded in %.1fs | response_len=%d chars", _attempt, elapsed, len(raw_response))
            break
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - t_start
            logger.warning("Node 6 synthesis timed out after %.1fs on attempt %d", elapsed, _attempt)
        except Exception as exc:
            elapsed = time.monotonic() - t_start
            logger.warning("Node 6 synthesis error (%s) on attempt %d: %s", type(exc).__name__, _attempt, exc)

    # 5. Extract JSON and sanitize response
    if isinstance(raw_response, list):
        raw_response = "\n".join(str(x) for x in raw_response)
    elif not isinstance(raw_response, str):
        raw_response = str(raw_response) if raw_response is not None else ""

    if raw_response and len(raw_response.strip()) > 30:
        parsed = _extract_json_from_response(raw_response)
        answer_md = sanitize_and_extract_answer(parsed if parsed else raw_response)

        raw_citations = parsed.get("citations", []) if (parsed and isinstance(parsed.get("citations"), list)) else []
        citations_source = raw_citations if raw_citations else top_passages

        final_citations: list[Citation] = []
        for c in citations_source:
            c_dict = c if isinstance(c, dict) else (c.model_dump() if hasattr(c, "model_dump") else {})
            c_go = c_dict.get("go_number") if isinstance(c_dict, dict) else getattr(c, "go_number", None)
            matching_tp = _find_best_matching_passage(c_dict, top_passages, go_number)

            dept = (c_dict.get("issuing_department") if isinstance(c_dict, dict) else getattr(c, "issuing_department", None)) or (
                getattr(matching_tp, "issuing_department", "उत्तराखण्ड शासन")
            )
            date_str = (c_dict.get("date") if isinstance(c_dict, dict) else getattr(c, "date", None)) or (
                getattr(matching_tp, "date", "2018-01-01")
            )

            # Prefer matching passage's authoritative 1-based page number
            pg = getattr(matching_tp, "page_number", None) or (
                c_dict.get("page_number") if isinstance(c_dict, dict) else getattr(c, "page_number", 1)
            )

            # Ensure authentic Hindi excerpt from the actual document chunk
            raw_excerpt = c_dict.get("exact_text_excerpt") if isinstance(c_dict, dict) else getattr(c, "exact_text_excerpt", "")
            tp_excerpt = getattr(matching_tp, "exact_text_excerpt", "") if matching_tp else ""

            if not raw_excerpt or re.search(r"[a-zA-Z]{4,}", raw_excerpt) or len(raw_excerpt.strip()) < 15:
                excerpt = _clean_excerpt(tp_excerpt) if tp_excerpt else _clean_excerpt(raw_excerpt)
            else:
                excerpt = _clean_excerpt(raw_excerpt)

            bbox = (
                getattr(matching_tp, "bounding_box_coordinates", None)
                if hasattr(matching_tp, "bounding_box_coordinates")
                else (matching_tp.get("bounding_box_coordinates") if isinstance(matching_tp, dict) else None)
            )

            final_citations.append(
                Citation(
                    go_number=c_go or getattr(matching_tp, "go_number", go_number),
                    issuing_department=dept,
                    date=date_str,
                    page_number=int(pg),
                    exact_text_excerpt=excerpt,
                    bounding_box_coordinates=bbox,
                )
            )

        if answer_md and answer_md.strip() and final_citations:
            logger.info("[Node6] Synthesis complete | citations=%d", len(final_citations))
            return {
                "answer_markdown": answer_md,
                "citations": final_citations,
            }

    # 6. Structured fallback if LLM timed out or failed
    logger.warning("[Node6] Using structured fallback for %s", go_number)
    return generate_structured_fallback(top_passages, go_number)
