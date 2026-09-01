import asyncio
import logging
import re
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.state.schema import Citation, OfficerContext, PassageMatch, QueryFilters, StateSchema
from src.tools.search_go_corpus import search_go_corpus
from src.tools.schemas.search_go_corpus import SearchGoCorpusInput, SearchGoCorpusOutput
from src.utils.model_runtime import ainvoke_with_retry, get_structured_llm

logger = logging.getLogger("shasanai.node3")

NODE3_SYSTEM_PROMPT = """<identity_and_role>
You are the Retrieval Invocation node of the PramanAI GovTech Agent Fleet.
Your purpose is to formulate an optimal search query expansion for hybrid dense+sparse retrieval against indexed government records.
</identity_and_role>

<primary_objective>
Expand the user's query with administrative, legal, and operational synonyms while strictly preserving:
1. ALL numeric figures, metrics, and day limits (e.g., '150', '100', '15 दिन', '₹1000', '₹2000').
2. ALL original Hindi/Devanagari keywords (e.g., 'यारसा गम्बू', 'कीड़ा-जड़ी', 'रॉयल्टी', 'निकासी रवन्ना', 'विदोहन', 'शासनादेश').
3. ALL Government Order codes, section tags, and clause identifiers (e.g., 'GO-667', 'प्रस्तर 9', 'प्रस्तर 10', 'Transit Pass').
</primary_objective>

<constraints>
- Return a single cohesive search string containing both the original keywords and essential administrative synonyms.
- Never strip or alter numeric values or units.
</constraints>

<few_shot_examples>
Example 1:
Input: query_text = "यदि किसी संग्रहकर्ता ने 150 ग्राम यारसा गम्बू एकत्रित की है, तो उसे कितनी रॉयल्टी देनी होगी और निकासी रवन्ने की वैधता कितने दिन की होती है?"
Output: {"expanded_search_terms": "यारसा गम्बू 150 ग्राम कीड़ा-जड़ी रॉयल्टी दर स्लैब निकासी रवन्ना वैधता मात्र 15 दिन प्रस्तर 9 प्रस्तर 10 विदोहन शुल्क GO-667 Yarsa Gambu royalty transit pass"}

Example 2:
Input: query_text = "Forest department transfer policy 2018 provisions"
Output: {"expanded_search_terms": "Forest department transfer posting policy 2018 samayojan cadre rules स्थानांतरण नीति वन विभाग"}
</few_shot_examples>
"""


class RetrievalExpansion(BaseModel):
    """Expanded search query schema for hybrid dense+sparse retrieval."""
    expanded_search_terms: str = Field(
        ...,
        description="Search query expanded with administrative synonyms while preserving all original numbers and keywords."
    )


async def node3_retrieval_invocation(state: StateSchema) -> dict[str, Any]:
    """
    Executes Node 3 (Retrieval Invocation) with deterministic keyword preservation,
    robust hybrid retrieval, and high-precision neural cross-encoder reranking.
    """
    query_text: str = state.get("query_text", "").strip(' \t\n\r"\'')
    query_lang: str = state.get("query_language", "hi" if re.search(r"[\u0900-\u097F]", query_text) else "en")
    query_filters: QueryFilters = state.get("query_filters", QueryFilters())
    officer_ctx: OfficerContext = state.get(
        "officer_context", OfficerContext(department="General", access_scope=["General"])
    )

    # 1. LLM Query Expansion
    user_prompt = f"""<query_text>
{query_text}
</query_text>
<query_language>
{query_lang}
</query_language>
<query_filters>
Department: {query_filters.department}
Year Range: {query_filters.year_range}
Policy Category: {query_filters.policy_category}
GO Number: {query_filters.go_number}
</query_filters>
"""

    structured_llm = get_structured_llm(RetrievalExpansion, temperature=0.0, max_tokens=200, timeout=15.0)
    messages = [
        SystemMessage(content=NODE3_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        expansion: RetrievalExpansion = await asyncio.wait_for(
            ainvoke_with_retry(
                structured_llm=structured_llm,
                messages=messages,
                max_retries=1,
                timeout_seconds=12.0,
            ),
            timeout=15.0,
        )
        search_terms = expansion.expanded_search_terms.strip() if expansion.expanded_search_terms else query_text
    except Exception as exc:
        logger.warning("Node 3: Query expansion fallback triggered (%s)", exc)
        search_terms = query_text

    # 2. Hardened Keyword & Numeric Retention Guard
    # Extracts: Devanagari words (>=2 chars), exact numbers/metrics, and alphanumeric identifiers
    vital_tokens = re.findall(r"[\u0900-\u097F]{2,}|\b\d+\b|[A-Za-z0-9\-\(\)/]{3,}", query_text)
    for token in vital_tokens:
        if token not in search_terms:
            search_terms = f"{search_terms} {token}".strip()

    # 3. Hybrid Retrieval via search_go_corpus Tool
    tool_input = SearchGoCorpusInput(
        query_text=search_terms,
        query_language=query_lang,  # type: ignore[arg-type]
        department_filter=query_filters.department,
        year_range_filter=query_filters.year_range,
        policy_category_filter=query_filters.policy_category,
        go_number_filter=query_filters.go_number,
        max_results=20,
    )

    logger.info("Node 3: Executing hybrid search for query: '%s'", search_terms[:90])
    passages: list[PassageMatch] = []
    try:
        tool_output: SearchGoCorpusOutput = await asyncio.wait_for(
            search_go_corpus(
                params=tool_input,
                officer_context=officer_ctx,
            ),
            timeout=10.0,
        )
        passages = tool_output.result or []
        logger.info("Node 3: Retrieval successfully returned %d candidate passages", len(passages))
    except Exception as exc:
        logger.error("Node 3: Hybrid search encountered an error (%s); returning empty passages", exc)

    # 4. Neural Cross-Encoder Reranking using Clean Original Query
    if passages and len(passages) > 1:
        try:
            from src.utils.reranker import rerank_passages
            # Rerank against original natural user query to maximize cross-attention precision
            passages = rerank_passages(query_text=query_text, passages=passages, top_k=8)
            logger.info("Node 3: Cross-encoder successfully reranked to top %d passages", len(passages))
        except Exception as exc:
            logger.warning("Node 3: Reranker execution failed (%s); keeping vector order", exc)
            passages = passages[:8]

    # 5. Populate Candidate Citations
    citations: list[Citation] = [
        Citation(
            go_number=p.go_number,
            issuing_department=p.issuing_department,
            date=p.date,
            page_number=p.page_number,
            exact_text_excerpt=p.exact_text_excerpt,
            bounding_box_coordinates=p.bounding_box_coordinates,
        )
        for p in passages
    ]

    return {
        "retrieved_passages": passages,
        "candidate_citations": citations,
    }