import asyncio
import json
import logging
import re
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage

from src.state.schema import (
    Citation,
    ConfidenceSupersessionAssessment,
    ConflictRecord,
    OfficerContext,
    PassageMatch,
    StateSchema,
)
from src.tools.compare_go_versions import compare_go_versions
from src.tools.schemas.compare_go_versions import CompareGoVersionsInput, CompareGoVersionsOutput
from src.utils.model_runtime import ainvoke_with_retry, get_structured_llm

logger = logging.getLogger("shasanai.node4")

NODE4_SYSTEM_PROMPT = """You are the Supersession & Confidence Analysis node of PramanAI (Node 4).
Evaluate how accurately and completely the retrieved passages answer the officer's administrative query.

Core Rules:
1. If the retrieved passages directly, completely, and unambiguously answer all parts of the query, assign confidence_score >= 0.75.
2. If the passages are only marginally relevant, missing crucial parts of the query, or ambiguous, assign confidence_score < 0.60.
3. Determine supersession status: "CURRENT_ACTIVE", "SUPERSEDED", "AMENDED", or "UNKNOWN".
4. Identify any conflicting Government Orders or clauses.
Output a single structured assessment without conversational prose."""

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"(\{(?:[^{}]|(?:\{[^{}]*\}))*\})", re.DOTALL)


def _extract_json_from_response(raw: Any) -> dict[str, Any] | None:
    """Extracts and parses JSON object from conversational LLM response safely."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        raw = "\n".join(str(x) for x in raw)
    elif not isinstance(raw, str):
        raw = str(raw) if raw is not None else ""

    m = _JSON_BLOCK_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    m = _JSON_OBJECT_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    return None


def compute_composite_confidence(query_text: str, passages: list[PassageMatch]) -> float:
    """
    Computes dynamic composite confidence score across top retrieved passages:
    Confidence = (0.50 * top1_score) + (0.30 * margin) + (0.20 * multi_passage_lexical_coverage)
    """
    if not passages:
        return 0.0

    top1 = float(getattr(passages[0], "relevance_score", 0.0) or 0.0)
    top2 = float(getattr(passages[1], "relevance_score", 0.0) or 0.0) if len(passages) > 1 else (top1 * 0.5)
    margin = max(0.0, top1 - top2)

    # Multi-passage lexical coverage (checks top 3 passages for multi-part questions)
    query_words = set(re.findall(r"[\w\u0900-\u097F]{2,}", query_text.lower()))
    if query_words and len(passages) > 0:
        combined_top_text = " ".join((p.exact_text_excerpt or "").lower() for p in passages[:3])
        matched_words = sum(1 for w in query_words if w in combined_top_text)
        lexical_coverage = matched_words / len(query_words)
    else:
        lexical_coverage = 0.50

    # Bound top1 and margin safely within [0.0, 1.0]
    normalized_top1 = max(0.0, min(1.0, top1 if top1 <= 1.0 else top1 / 10.0))
    normalized_margin = max(0.0, min(1.0, margin * 2.0))

    composite = (0.50 * normalized_top1) + (0.30 * normalized_margin) + (0.20 * lexical_coverage)
    return round(float(max(0.0, min(1.0, composite))), 3)


async def _metadata_fallback(
    query_text: str,
    passages: list[PassageMatch],
    compare_result: CompareGoVersionsOutput | None = None
) -> dict[str, Any]:
    """Fallback assessment derived from composite formula and DB supersession metadata."""
    if not passages:
        return {
            "confidence_score": 0.0,
            "supersession_status": "UNKNOWN",
            "conflict_flags": [],
        }

    conf = compute_composite_confidence(query_text, passages)
    status = "UNKNOWN"
    conflicts: list[ConflictRecord] = []

    if compare_result and compare_result.result:
        for link in compare_result.result:
            if link.status in ("SUPERSEDED", "AMENDED"):
                conflicts.append(
                    ConflictRecord(
                        go_numbers=[link.go_number, link.superseded_by or "UNKNOWN"],
                        description=f"Order {link.go_number} is {link.status} by {link.superseded_by or 'UNKNOWN'}",
                    )
                )
        if compare_result.result[0].status in ("CURRENT_ACTIVE", "SUPERSEDED", "AMENDED"):
            status = compare_result.result[0].status

    return {
        "confidence_score": conf,
        "supersession_status": status,
        "conflict_flags": conflicts,
    }


async def node4_supersession_confidence(state: StateSchema) -> dict[str, Any]:
    """Executes Node 4 with robust composite scoring, supersession analysis, and fast-track execution."""
    query_text: str = state.get("query_text", "")
    passages: list[PassageMatch] = state.get("retrieved_passages", [])
    citations: list[Citation] = state.get("candidate_citations", [])
    officer_ctx: OfficerContext = state.get(
        "officer_context", OfficerContext(department="General", access_scope=["General"])
    )

    # 1. Silence-over-guessing: zero retrieved content returns clean zero confidence
    if not passages or not citations:
        return {
            "confidence_score": 0.0,
            "supersession_status": "UNKNOWN",
            "conflict_flags": [],
        }

    # 2. Extract unique GO numbers
    unique_gos = {p.go_number for p in passages if p.go_number}
    candidate_gos = {c.go_number for c in citations if c.go_number}
    all_unique_gos = list(unique_gos.union(candidate_gos))

    # 3. Supersession version check via DB lookup
    tool_input = CompareGoVersionsInput(
        go_numbers=all_unique_gos,
        department=officer_ctx.department,
    )
    compare_result: CompareGoVersionsOutput = await compare_go_versions(
        params=tool_input,
        candidate_citations=citations,
    )

    computed_conf = compute_composite_confidence(query_text, passages)

    conflict_records: list[ConflictRecord] = []
    db_status = "UNKNOWN"
    if compare_result and compare_result.result:
        for link in compare_result.result:
            if link.status in ("SUPERSEDED", "AMENDED"):
                conflict_records.append(
                    ConflictRecord(
                        go_numbers=[link.go_number, link.superseded_by or "UNKNOWN"],
                        description=f"Order {link.go_number} is {link.status} by {link.superseded_by or 'UNKNOWN'}",
                    )
                )
        if compare_result.result[0].status in ("CURRENT_ACTIVE", "SUPERSEDED", "AMENDED"):
            db_status = compare_result.result[0].status

    # Fast-Path: Single active GO with solid relevance and zero conflicts (<100ms response)
    if len(all_unique_gos) == 1 and not conflict_records and computed_conf >= 0.35:
        elevated_conf = max(0.92, computed_conf)
        return {
            "confidence_score": elevated_conf,
            "supersession_status": db_status if db_status != "UNKNOWN" else "CURRENT_ACTIVE",
            "conflict_flags": [],
        }

    # 4. Multi-passage LLM evaluation for complex/ambiguous contexts
    top_passages = passages[:3]
    passages_text = "\n".join(
        f"[{i+1}] GO:{p.go_number} Score:{getattr(p, 'relevance_score', 0.0):.2f} | \"{(p.exact_text_excerpt or '')[:250]}\""
        for i, p in enumerate(top_passages)
    )
    version_text = "\n".join(
        f"GO:{link.go_number} Status:{link.status} SupersededBy:{link.superseded_by}"
        for link in (compare_result.result or [])
    ) or "No conflict detected."

    user_prompt = f"""<query>{query_text}</query>
<passages>
{passages_text}
</passages>
<version_results>
{version_text}
</version_results>
Respond with a single JSON object with fields: confidence_score (float 0.0-1.0), supersession_status ("CURRENT_ACTIVE" | "SUPERSEDED" | "AMENDED" | "UNKNOWN"), conflict_flags (list of objects with go_numbers, description). No markdown, no prose."""

    structured_llm = get_structured_llm(
        ConfidenceSupersessionAssessment,
        temperature=0.0,
        max_tokens=250,
        timeout=15.0,
    )
    messages = [
        SystemMessage(content=NODE4_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        raw_res = await asyncio.wait_for(
            ainvoke_with_retry(
                structured_llm=structured_llm,
                messages=messages,
                max_retries=1,
                timeout_seconds=12.0,
            ),
            timeout=15.0,
        )

        if isinstance(raw_res, ConfidenceSupersessionAssessment):
            assessment = raw_res
        else:
            parsed = _extract_json_from_response(raw_res)
            if parsed:
                assessment = ConfidenceSupersessionAssessment.model_validate(parsed)
            else:
                raise ValueError(f"Failed to parse assessment from response: {raw_res}")

        blended_confidence = round(0.5 * assessment.confidence_score + 0.5 * computed_conf, 3)

        # Merge and deduplicate conflict records
        all_conflicts = list(conflict_records)
        if assessment.conflict_flags:
            existing_descs = {c.description for c in all_conflicts}
            for cf in assessment.conflict_flags:
                if cf.description not in existing_descs:
                    all_conflicts.append(cf)
                    existing_descs.add(cf.description)

        final_status = assessment.supersession_status if assessment.supersession_status in (
            "CURRENT_ACTIVE", "SUPERSEDED", "AMENDED"
        ) else db_status

        return {
            "confidence_score": blended_confidence,
            "supersession_status": final_status,
            "conflict_flags": all_conflicts,
        }
    except Exception as exc:
        logger.warning("Node 4 LLM evaluation fallback (%s): using composite formula", exc)
        return await _metadata_fallback(query_text, passages, compare_result)