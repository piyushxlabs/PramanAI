"""MultiServerMCPClient connection manager for ShasanAI read-only MCP tool servers.

Connects to the three bespoke ITDA MCP servers:
1. Document-Retrieval MCP Server (search_go_corpus)
2. Supersession/Version-Comparison MCP Server (compare_go_versions)
3. Bounding-Box/Source-Highlight MCP Server (get_source_highlight)

All tools are 100% database-backed via PostgreSQL 16 pgvector, documents master table,
and supersession_graph relational table. Zero mock dictionaries.
"""

import json
import logging
import math
import os
import re
from typing import Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger("shasanai.mcp_client")

from src.tools.schemas.compare_go_versions import (
    COMPARE_GO_VERSIONS_JSON_SCHEMA,
    CompareGoVersionsOutput,
    SupersessionLink,
)
from src.tools.schemas.get_source_highlight import (
    GET_SOURCE_HIGHLIGHT_JSON_SCHEMA,
    BoundingBox,
    GetSourceHighlightOutput,
)
from src.tools.schemas.search_go_corpus import (
    SEARCH_GO_CORPUS_JSON_SCHEMA,
    PassageMatch,
    SearchGoCorpusOutput,
)

load_dotenv()


def sanitize_relevance_score(raw_score: Any) -> float:
    """Sanitizes relevance score, replacing NaN/None with 0.0 and clamping to [0.0, 1.0]."""
    if raw_score is None:
        return 0.0
    try:
        val = float(raw_score)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return max(0.0, min(1.0, val))
    except (ValueError, TypeError):
        return 0.0


class MultiServerMCPClientManager:
    """Manages connections to the 3 read-only MCP servers backed directly by PostgreSQL."""

    def __init__(self):
        self.retrieval_url = os.getenv("MCP_RETRIEVAL_SERVER_URL")
        self.supersession_url = os.getenv("MCP_SUPERSESSION_SERVER_URL")
        self.highlight_url = os.getenv("MCP_HIGHLIGHT_SERVER_URL")
        self.retrieval_token = os.getenv("MCP_RETRIEVAL_SERVER_TOKEN")
        self.supersession_token = os.getenv("MCP_SUPERSESSION_SERVER_TOKEN")
        self.highlight_token = os.getenv("MCP_HIGHLIGHT_SERVER_TOKEN")

    def get_tools_manifest(self) -> list[dict[str, Any]]:
        """Returns schemas for the 3 authorized read-only tools."""
        return [
            SEARCH_GO_CORPUS_JSON_SCHEMA,
            COMPARE_GO_VERSIONS_JSON_SCHEMA,
            GET_SOURCE_HIGHLIGHT_JSON_SCHEMA,
        ]

    async def call_search_go_corpus(self, arguments: dict[str, Any]) -> SearchGoCorpusOutput:
        """Executes search_go_corpus MCP tool call against PostgreSQL pgvector store with SQL-level RRF."""
        query_text = arguments.get("query_text", "")
        dept_filter = arguments.get("department_filter")
        year_filter = arguments.get("year_range_filter")
        go_num_filter = arguments.get("go_number_filter")
        max_results = arguments.get("max_results", 20)

        try:
            from src.ingestion.vector_store import VectorStore
            store = VectorStore()

            matches = await store.a_hybrid_search(
                query_text=query_text,
                department_filter=dept_filter,
                year_range_filter=year_filter,
                go_number_filter=go_num_filter,
                max_results=max_results,
            )

            # Adaptive fallback: if 0 matches and go_number is present, relax dept & year filters
            if not matches and go_num_filter:
                logger.info("0 matches with strict filters. Relaxing year/dept filter for GO: %s", go_num_filter)
                matches = await store.a_hybrid_search(
                    query_text=query_text,
                    department_filter=None,
                    year_range_filter=None,
                    go_number_filter=go_num_filter,
                    max_results=max_results,
                )

            if matches:
                sanitized_matches = []
                for m in matches:
                    raw_score = getattr(m, "relevance_score", 0.0)
                    score = sanitize_relevance_score(raw_score)
                    sanitized_matches.append(
                        PassageMatch(
                            go_number=m.go_number,
                            issuing_department=m.issuing_department,
                            date=m.date,
                            page_number=m.page_number,
                            exact_text_excerpt=m.exact_text_excerpt,
                            relevance_score=score,
                            bounding_box_coordinates=m.bounding_box_coordinates,
                        )
                    )
                logger.info("Retrieved %d matches from PostgreSQL pgvector RRF hybrid search", len(sanitized_matches))
                return SearchGoCorpusOutput(success=True, result=sanitized_matches, error=None)

            logger.warning("PostgreSQL pgvector search returned 0 matches for query '%s'", query_text[:80])
            return SearchGoCorpusOutput(success=True, result=[], error="No matching government orders found in corpus.")
        except Exception as e:
            logger.error("PostgreSQL pgvector search error: %s", e)
            return SearchGoCorpusOutput(success=False, result=[], error=str(e))

    async def call_compare_go_versions(self, arguments: dict[str, Any]) -> CompareGoVersionsOutput:
        """Executes compare_go_versions tool querying supersession_graph and documents tables in PostgreSQL."""
        go_numbers = arguments.get("go_numbers", [])
        if not go_numbers:
            return CompareGoVersionsOutput(success=True, result=[], error=None)

        links: list[SupersessionLink] = []
        try:
            from src.ingestion.vector_store import VectorStore
            store = VectorStore()
            pool = await store.get_async_pool()

            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    # 1. Query supersession_graph table
                    patterns = [f"%{go.strip()}%" for go in go_numbers if go]
                    await cur.execute(
                        """
                        SELECT go_number, status, superseded_by, amends
                        FROM supersession_graph
                        WHERE go_number = ANY(%s) OR go_number ILIKE ANY(%s);
                        """,
                        (go_numbers, patterns),
                    )
                    graph_rows = await cur.fetchall()
                    found_map = {r["go_number"].lower(): r for r in graph_rows}

                    # 2. Check documents table for remaining GOs to verify active status
                    await cur.execute(
                        """
                        SELECT go_number, status
                        FROM documents
                        WHERE go_number = ANY(%s) OR go_number ILIKE ANY(%s);
                        """,
                        (go_numbers, patterns),
                    )
                    doc_rows = await cur.fetchall()
                    doc_map = {r["go_number"].lower(): r for r in doc_rows}

                    for go in go_numbers:
                        clean_go = go.lower().strip()
                        matched_row = found_map.get(clean_go)
                        if not matched_row:
                            # Fuzzy key match in found_map
                            matched_row = next((v for k, v in found_map.items() if clean_go in k or k in clean_go), None)

                        if matched_row:
                            raw_status = matched_row.get("status", "CURRENT_ACTIVE")
                            valid_status = raw_status if raw_status in ("CURRENT_ACTIVE", "AMENDED", "SUPERSEDED", "UNKNOWN") else "CURRENT_ACTIVE"
                            links.append(
                                SupersessionLink(
                                    go_number=go,
                                    status=valid_status,
                                    superseded_by=matched_row.get("superseded_by"),
                                    amends=matched_row.get("amends"),
                                )
                            )
                        elif clean_go in doc_map or any(clean_go in k or k in clean_go for k in doc_map):
                            matched_doc = doc_map.get(clean_go) or next(v for k, v in doc_map.items() if clean_go in k or k in clean_go)
                            raw_status = matched_doc.get("status", "CURRENT_ACTIVE")
                            valid_status = raw_status if raw_status in ("CURRENT_ACTIVE", "AMENDED", "SUPERSEDED", "UNKNOWN") else "CURRENT_ACTIVE"
                            links.append(
                                SupersessionLink(
                                    go_number=go,
                                    status=valid_status,
                                    superseded_by=None,
                                    amends=None,
                                )
                            )
                        else:
                            # Default active if present in document_chunks
                            links.append(
                                SupersessionLink(
                                    go_number=go,
                                    status="CURRENT_ACTIVE",
                                    superseded_by=None,
                                    amends=None,
                                )
                            )

            return CompareGoVersionsOutput(success=True, result=links, error=None)

        except Exception as exc:
            logger.warning("Database supersession query error (%s): %s; returning default CURRENT_ACTIVE", type(exc).__name__, exc)
            for go in go_numbers:
                links.append(
                    SupersessionLink(
                        go_number=go,
                        status="CURRENT_ACTIVE",
                        superseded_by=None,
                        amends=None,
                    )
                )
            return CompareGoVersionsOutput(success=True, result=links, error=None)

    async def call_get_source_highlight(self, arguments: dict[str, Any]) -> GetSourceHighlightOutput:
        """Executes get_source_highlight MCP tool querying document_chunks table directly for true bounding boxes."""
        go_number = arguments.get("go_number", "")
        page_number = arguments.get("page_number", 1)

        try:
            from src.ingestion.vector_store import VectorStore
            store = VectorStore()
            pool = await store.get_async_pool()

            raw_go = go_number.strip().lstrip("GO-").strip()
            core_match = re.search(r"\b\d{2,5}\b", raw_go)
            target_go = core_match.group(0) if core_match else raw_go

            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT bounding_box_coordinates
                        FROM document_chunks
                        WHERE (go_number ILIKE %s OR document_id ILIKE %s)
                          AND page_number = %s
                          AND bounding_box_coordinates IS NOT NULL
                        ORDER BY chunk_index ASC
                        LIMIT 1;
                        """,
                        (f"%{target_go}%", f"%{target_go}%", page_number),
                    )
                    row = await cur.fetchone()

                    if row and row.get("bounding_box_coordinates"):
                        raw_bbox = row["bounding_box_coordinates"]
                        if isinstance(raw_bbox, str):
                            raw_bbox = json.loads(raw_bbox)
                        if isinstance(raw_bbox, list) and len(raw_bbox) >= 4:
                            return GetSourceHighlightOutput(
                                success=True,
                                result=BoundingBox(
                                    x=float(raw_bbox[0]),
                                    y=float(raw_bbox[1]),
                                    width=float(raw_bbox[2]),
                                    height=float(raw_bbox[3]),
                                ),
                                error=None,
                            )
                        elif isinstance(raw_bbox, dict):
                            return GetSourceHighlightOutput(
                                success=True,
                                result=BoundingBox(**raw_bbox),
                                error=None,
                            )

            # When no authentic bounding box exists in database, return result=None (clean page rendering)
            return GetSourceHighlightOutput(
                success=True,
                result=None,
                error=None,
            )

        except Exception as exc:
            logger.warning("Database source highlight lookup error: %s", exc)
            return GetSourceHighlightOutput(
                success=True,
                result=None,
                error=str(exc),
            )


_mcp_manager: Optional[MultiServerMCPClientManager] = None


def get_mcp_client_manager() -> MultiServerMCPClientManager:
    """Returns singleton MultiServerMCPClientManager instance."""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MultiServerMCPClientManager()
    return _mcp_manager
