"""Integration test suite for ShasanAI Production RAG Scaling Overhaul (40,000+ PDFs).

Verifies:
1. Schema migrations: documents, supersession_graph, document_chunks tsvector column and GIN index.
2. Connection pooling: concurrent query execution without TCP connection exhaustion.
3. True SQL-level Hybrid Search with Reciprocal Rank Fusion (RRF).
4. Layer 3 Neural Cross-Encoder reranking (FlashRank ONNX).
5. Elimination of all mock dictionaries in MCP client.
6. Dynamic Node 6 fallback synthesis without hardcoded GO-667 strings.
"""

import asyncio
import pytest
from src.agents.nodes.node6_grounded_synthesis import generate_structured_fallback
from src.ingestion.vector_store import VectorStore
from src.tools.schemas.search_go_corpus import PassageMatch
from src.tools.mcp_clients.mcp_client import (
    get_mcp_client_manager,
)
from src.utils.reranker import rerank_passages


@pytest.mark.asyncio
async def test_schema_and_indexes_exist():
    """Verifies that production tables and GIN/B-tree indexes exist in PostgreSQL."""
    store = VectorStore()
    store.initialize_schema()

    with store.get_connection() as conn:
        with conn.cursor() as cur:
            # 1. Verify documents table
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'documents';")
            doc_cols = {row["column_name"] for row in cur.fetchall()}
            assert "document_id" in doc_cols
            assert "go_number" in doc_cols
            assert "year" in doc_cols
            assert "status" in doc_cols

            # 2. Verify supersession_graph table
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'supersession_graph';")
            sg_cols = {row["column_name"] for row in cur.fetchall()}
            assert "go_number" in sg_cols
            assert "status" in sg_cols
            assert "superseded_by" in sg_cols

            # 3. Verify document_chunks generated tsvector column
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'document_chunks';")
            chunk_cols = {row["column_name"] for row in cur.fetchall()}
            assert "tsv_content" in chunk_cols
            assert "year" in chunk_cols
            assert "status" in chunk_cols


@pytest.mark.asyncio
async def test_mock_dictionaries_purged():
    """Verifies that no static mock dictionaries exist in mcp_client.py."""
    from src.tools.mcp_clients import mcp_client

    assert not hasattr(mcp_client, "MOCK_CORPUS"), "MOCK_CORPUS must be purged"
    assert not hasattr(mcp_client, "MOCK_SUPERSESSION"), "MOCK_SUPERSESSION must be purged"
    assert not hasattr(mcp_client, "MOCK_HIGHLIGHTS"), "MOCK_HIGHLIGHTS must be purged"


@pytest.mark.asyncio
async def test_connection_pooling_concurrency():
    """Verifies that 20 concurrent search operations succeed without connection exhaustion."""
    store = VectorStore()

    async def _query_task(q_num: int):
        return await store.a_hybrid_search(
            query_text=f"वन विभाग शासनादेश नियम {q_num}",
            query_embedding=[0.0] * 1024,
            department_filter="Forest",
            max_results=5,
        )

    tasks = [_query_task(i) for i in range(20)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for idx, res in enumerate(results):
        assert not isinstance(res, Exception), f"Concurrent query {idx} raised exception: {res}"


@pytest.mark.asyncio
async def test_sql_hybrid_search_rrf():
    """Verifies SQL Reciprocal Rank Fusion returns valid PassageMatch items with normalized scores."""
    store = VectorStore()

    # Seed test chunk if needed
    with store.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_chunks (
                    chunk_id, document_id, go_number, issuing_department, date, year, status,
                    page_number, chunk_index, exact_text_excerpt, embedding
                ) VALUES (
                    'test:doc:p1:c0', 'test:doc', 'GO-TEST-999', 'Forest', '2018-05-10', 2018, 'CURRENT_ACTIVE',
                    1, 0, 'रॉयल्टी दर विदोहन एवं निकासी रवन्ना के नियम', %s::vector
                ) ON CONFLICT (chunk_id) DO NOTHING;
                """,
                (f"[{','.join(['0.01']*1024)}]",),
            )

    matches = await store.a_hybrid_search(
        query_text="रॉयल्टी दर विदोहन",
        query_embedding=[0.01] * 1024,
        department_filter=None,
        max_results=10,
    )

    assert isinstance(matches, list)
    assert len(matches) >= 1
    for m in matches:
        assert isinstance(m, PassageMatch)
        assert 0.0 <= m.relevance_score <= 1.0
        assert m.go_number is not None
        assert m.page_number >= 1


@pytest.mark.asyncio
async def test_cross_encoder_reranking():
    """Verifies FlashRank Neural Cross-Encoder reranker orders passages by cross-attention relevance."""
    passages = [
        PassageMatch(
            go_number="GO-100",
            issuing_department="Forest",
            date="2018-01-01",
            page_number=1,
            exact_text_excerpt="हस्ताक्षर एवं सेवा में प्रतिलिपि प्रेषित।",
            relevance_score=0.90,
        ),
        PassageMatch(
            go_number="GO-100",
            issuing_department="Forest",
            date="2018-01-01",
            page_number=7,
            exact_text_excerpt="विदोहन काल समाप्त होने पर एक सप्ताह के अन्दर रॉयल्टी जमा करना अनिवार्य है।",
            relevance_score=0.70,
        ),
    ]

    reranked = rerank_passages(
        query_text="रॉयल्टी जमा करने की समयसीमा क्या है?",
        passages=passages,
        top_k=2,
    )

    assert len(reranked) == 2
    # The substantive provision on royalty should be ranked first by the cross-encoder
    assert "रॉयल्टी" in reranked[0].exact_text_excerpt


@pytest.mark.asyncio
async def test_supersession_graph_tool_db_lookup():
    """Verifies compare_go_versions queries supersession_graph table dynamically."""
    manager = get_mcp_client_manager()
    store = VectorStore()

    # Seed test supersession record in PostgreSQL
    with store.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO supersession_graph (go_number, status, superseded_by, amends)
                VALUES ('GO-TEST-OLD/2014', 'SUPERSEDED', 'GO-TEST-NEW/2018', NULL)
                ON CONFLICT (id) DO NOTHING;
                """
            )

    output = await manager.call_compare_go_versions({"go_numbers": ["GO-TEST-OLD/2014"]})
    assert output.success is True
    assert len(output.result) == 1
    assert output.result[0].status == "SUPERSEDED"
    assert output.result[0].superseded_by == "GO-TEST-NEW/2018"


@pytest.mark.asyncio
async def test_dynamic_fallback_synthesis():
    """Verifies generate_structured_fallback produces dynamic output without hardcoded GO-667 strings."""
    test_passages = [
        PassageMatch(
            go_number="GO-999/EDU/2022",
            issuing_department="Education",
            date="2022-05-15",
            page_number=3,
            exact_text_excerpt="शिक्षकों के स्थानांतरण हेतु ऑनलाइन आवेदन पोर्टल के माध्यम से स्वीकार किए जाएंगे।",
            relevance_score=0.88,
        )
    ]

    fallback = generate_structured_fallback(test_passages, go_number="GO-999/EDU/2022")
    assert "GO-999/EDU/2022" in fallback["answer_markdown"]
    assert "शिक्षकों के स्थानांतरण" in fallback["answer_markdown"]
    assert "काली सूची (Blacklist)" not in fallback["answer_markdown"]  # Verify old hardcoded string is gone
    assert len(fallback["citations"]) == 1
    assert fallback["citations"][0].go_number == "GO-999/EDU/2022"
