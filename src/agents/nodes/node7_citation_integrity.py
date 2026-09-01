"""Node 7: Citation Integrity Check (Verbatim Grounding & N-Gram Validation).

Independently verifies that every factual claim in answer_markdown maps to a verified citation.
Performs claim-level character/token 3-gram recall against raw OCR citation excerpts,
strictly rejecting ungrounded claims, unverified numbers, and hallucinations.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any

from src.state.schema import Citation, CitationIntegrityResult, ErrorRecord, StateSchema

logger = logging.getLogger("shasanai.node7")


def _get_char_3grams(text: str) -> set[str]:
    """Extracts sliding-window character 3-grams from normalized text."""
    clean = re.sub(r"[^\w\u0900-\u097F]+", "", text.lower())
    if len(clean) < 3:
        return {clean} if clean else set()
    return {clean[i : i + 3] for i in range(len(clean) - 2)}


def extract_clean_verifiable_text(llm_output: Any) -> str:
    """Extracts purely the natural language answer from JSON strings, dicts, or markdown blocks."""
    if not llm_output:
        return ""
    if isinstance(llm_output, dict):
        for k in ["answer_markdown", "answermarkdown", "answer", "response", "content", "markdown", "text"]:
            if k in llm_output and isinstance(llm_output[k], str) and llm_output[k].strip():
                return llm_output[k].strip()
        parts = [
            str(v).strip()
            for k, v in llm_output.items()
            if isinstance(v, str) and k.lower() not in ["citations", "candidate_citations", "status", "exact_text_excerpt"] and len(str(v).strip()) > 15
        ]
        if parts:
            return "\n\n".join(parts)
        return str(llm_output).strip()

    raw_str = str(llm_output).strip()
    raw_str = re.sub(r"^```(?:json|markdown)?\s*", "", raw_str, flags=re.MULTILINE)
    raw_str = re.sub(r"\s*```$", "", raw_str, flags=re.MULTILINE).strip()

    # Search for "answer_markdown": "..." in raw string
    match = re.search(r'"(?:answer_markdown|answermarkdown|answer)"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_str, re.DOTALL)
    if match:
        try:
            val = json.loads(f'"{match.group(1)}"')
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        except Exception:
            return match.group(1).replace(r"\n", "\n").replace(r'\"', '"').strip()

    if "{" in raw_str and "}" in raw_str:
        try:
            match_obj = re.search(r"\{.*\}", raw_str, re.DOTALL)
            if match_obj:
                parsed = json.loads(match_obj.group(0))
                if isinstance(parsed, dict):
                    for k in ["answer_markdown", "answermarkdown", "answer", "response", "content", "markdown", "text"]:
                        if k in parsed and isinstance(parsed[k], str) and parsed[k].strip():
                            return parsed[k].strip()
        except Exception:
            pass

    return raw_str


extract_text_for_verification = extract_clean_verifiable_text


def _extract_factual_claims(answer_markdown: str) -> list[str]:
    """Splits answer_markdown into individual substantive factual claims, stripping headers, boilerplate, JSON keys, and negative remarks."""
    if not answer_markdown:
        return []

    # Clean text through JSON / envelope extractor
    clean_text = extract_clean_verifiable_text(answer_markdown)
    if not clean_text:
        return []

    lines = clean_text.split("\n")
    claims: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip JSON syntax, metadata keys, and formatting
        if re.match(r'^\s*["\{\}\[\]]', stripped) or re.search(r'"(?:go_number|issuing_department|exact_text_excerpt|page_number|date|citations|bounding_box)"\s*:', stripped, re.IGNORECASE):
            continue
        # Skip markdown headers, citation references, and preambles
        if stripped.startswith("#") or stripped.startswith("*(") or stripped.endswith(")*"):
            continue
        if "प्रमाणित प्रशासनिक संदर्भ" in stripped or ("शासनादेश संख्या" in stripped and "के अनुसार" in stripped):
            continue
        # Skip conversational, negative bridging, or disclaimer sentences
        if any(neg in stripped for neg in ["उल्लेख नहीं", "प्रावधान नहीं", "उपलब्ध नहीं", "अभिलेखों में नहीं", "विवरण नहीं", "स्पष्ट नहीं"]):
            continue
        # Clean bullet markers and bold markers
        cleaned = re.sub(r"^[\*\-\•\d\.\s]+", "", stripped)
        cleaned = re.sub(r"[\*\_]", "", cleaned).strip()
        if len(cleaned) >= 15:
            claims.append(cleaned)

    # Fallback to sentence split if no bullet lines found
    if not claims and len(clean_text.strip()) >= 20:
        sentences = re.split(r"[।\.\n]+", clean_text)
        claims = [
            s.strip() for s in sentences
            if len(s.strip()) >= 15 and not s.strip().startswith("*(")
            and not re.match(r'^\s*["\{\}\[\]]', s.strip())
            and not any(neg in s for neg in ["उल्लेख नहीं", "प्रावधान नहीं", "उपलब्ध नहीं", "अभिलेखों में नहीं"])
        ]

    return claims


def _verify_sentence_grounding(
    claim: str, citation_text: str, citation_3grams: set[str], query_text: str = ""
) -> tuple[bool, float, str | None]:
    """Verifies a single factual claim against citation text using 3-gram recall and numeric validation."""
    claim_3grams = _get_char_3grams(claim)
    if not claim_3grams:
        return True, 1.0, None

    overlap = claim_3grams & citation_3grams
    recall = len(overlap) / len(claim_3grams)

    # Substantive grounding threshold: minimum 25% 3-gram verbatim recall against citation excerpts
    if recall < 0.25:
        return False, recall, f"Grounding recall too low ({recall:.2%}) for claim: '{claim[:80]}'"

    citation_clean = citation_text.lower().replace(",", "")
    query_clean = query_text.lower().replace(",", "")
    combined_clean = f"{citation_clean} {query_clean}"

    # Extract numbers including comma-separated thousands (e.g. 50,000, 10,000)
    raw_numbers = re.findall(r"\b\d+(?:,\d+)?(?:\.\d+)?\b", claim)
    clean_numbers = [n.replace(",", "") for n in raw_numbers]

    # For borderline recall (0.25 <= recall < 0.35), verify numbers strictly
    if recall < 0.35:
        unverified = [num for num in clean_numbers if len(num) >= 3 and num not in combined_clean]
        if unverified:
            return False, recall, f"Unverified numeric assertion '{unverified[0]}' in claim: '{claim[:80]}'"

    return True, recall, None


def _deterministic_citation_check(
    answer_markdown: str,
    citations: list[Citation],
    query_text: str = "",
    candidate_citations: list[Citation] | None = None,
) -> tuple[bool, list[str]]:
    """Performs claim-level verbatim n-gram verification of answer_markdown against citations and corpus context."""
    if not answer_markdown or (not citations and not candidate_citations):
        return False, ["Empty answer or citations list"]

    claims = _extract_factual_claims(answer_markdown)
    if not claims:
        # Empty substantive claims — check total answer text overlap
        claims = [answer_markdown]

    # Aggregate all verbatim text from citations + candidate passages
    all_excerpts = [c.exact_text_excerpt for c in (citations or []) if getattr(c, "exact_text_excerpt", None)]
    if candidate_citations:
        all_excerpts.extend([c.exact_text_excerpt for c in candidate_citations if getattr(c, "exact_text_excerpt", None)])
    all_citation_text = "\n".join(all_excerpts)
    if not all_citation_text.strip():
        return False, ["Citation excerpts are empty"]

    citation_3grams = _get_char_3grams(all_citation_text)
    uncited_claims: list[str] = []

    for claim in claims:
        is_grounded, recall, err_reason = _verify_sentence_grounding(
            claim=claim,
            citation_text=all_citation_text,
            citation_3grams=citation_3grams,
            query_text=query_text,
        )
        if not is_grounded:
            logger.info("Node 7: Claim failed grounding (recall=%.2f): '%s'", recall, claim[:60])
            uncited_claims.append(err_reason or f"Ungrounded claim: {claim[:80]}")

    if not uncited_claims:
        return True, []

    return False, uncited_claims


async def node7_citation_integrity(state: StateSchema) -> dict[str, Any]:
    """Executes Node 7 (Citation Integrity Check) with strict claim-level grounding."""
    raw_payload: Any = state.get("answer_markdown")
    answer_markdown: str = extract_text_for_verification(raw_payload)
    citations: list[Citation] = state.get("citations", [])
    candidate_citations: list[Citation] = state.get("candidate_citations", [])
    query_text: str = state.get("query_text", "")
    error_logs: list[ErrorRecord] = state.get("error_logs", [])
    config = state.get("config")
    max_retries = getattr(config, "max_citation_retries", 2) if config else 2

    # 1. Guard against empty answer or citations
    if not answer_markdown or (not citations and not candidate_citations):
        err = ErrorRecord(
            node="node7_citation_integrity",
            error_type="EmptyAnswerOrCitations",
            message="No answer_markdown or citations present for integrity check.",
            timestamp=datetime.now().isoformat(),
        )
        return {
            "graceful_refusal": True,
            "error_logs": [err],
        }

    # 2. Claim-level verbatim verification
    try:
        is_valid, uncited_claims = _deterministic_citation_check(
            answer_markdown=answer_markdown,
            citations=citations,
            query_text=query_text,
            candidate_citations=candidate_citations,
        )
        if is_valid:
            logger.info("Node 7: All factual claims verified against citations.")
            return {
                "graceful_refusal": False,
            }
    except Exception as exc:
        logger.error("Node 7 verification exception: %s", exc)
        uncited_claims = [f"Verification error: {exc!s}"]

    # 3. Handle integrity failure with bounded retries
    err = ErrorRecord(
        node="node7_citation_integrity",
        error_type="CitationIntegrityFailure",
        message=f"Uncited claims detected: {uncited_claims}",
        timestamp=datetime.now().isoformat(),
    )

    prior_failures = sum(
        1 for e in error_logs if getattr(e, "error_type", None) == "CitationIntegrityFailure"
    )

    if prior_failures + 1 >= max_retries:
        # Exhausted retry cap -> route to refusal
        logger.warning("Node 7: Citation retries exhausted (%d/%d) -> graceful refusal.", prior_failures + 1, max_retries)
        return {
            "graceful_refusal": True,
            "error_logs": [err],
        }

    # Retry allowed -> remain in loop back to Node 6
    logger.info("Node 7: Triggering bounded re-synthesis retry loop (%d/%d).", prior_failures + 1, max_retries)
    return {
        "graceful_refusal": False,
        "error_logs": [err],
    }
