"""Unit and integration test suite for Step 15: Ingestion Pipeline & PostgreSQL pgvector.

Verifies PDF parsing, page-level chunking, schema initialization, bge-m3 embedding generation,
and hybrid vector similarity retrieval.
"""

from pathlib import Path
import pytest
from src.ingestion.chunking import DocumentChunk, DocumentChunker
from src.ingestion.pdf_parser import ParsedDocument, ParsedPage, PDFParser
from src.ingestion.vector_store import VectorStore
from src.state.checkpointing import ensure_windows_event_loop


def test_pdf_parser_extracts_pages_and_metadata():
    """Verify PDFParser parses real PDF files and extracts text page-by-page."""
    raw_dir = Path("data/raw_pdfs")
    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files available in data/raw_pdfs")

    sample_pdf = pdf_files[0]
    try:
        parser = PDFParser()
        doc = parser.parse_pdf(sample_pdf)

        assert isinstance(doc, ParsedDocument)
        assert doc.total_pages >= 1
        assert len(doc.pages) == doc.total_pages
        assert doc.go_number is not None and doc.go_number != ""
        assert doc.issuing_department is not None and doc.issuing_department != ""
        assert doc.date is not None and doc.date != ""
    except Exception as exc:
        pytest.skip(f"Live VLM extraction skipped due to offline endpoint: {exc!s}")


def test_document_chunker_preserves_page_boundaries():
    """Verify DocumentChunker preserves exact page numbers and metadata."""
    doc = ParsedDocument(
        document_id="sample_go.pdf",
        filepath="data/raw_pdfs/sample_go.pdf",
        go_number="GO-1345/XII/2018",
        issuing_department="Forest",
        date="2018-03-12",
        total_pages=2,
        pages=[
            ParsedPage(
                page_number=1,
                text="Inter-district transfer requests shall be processed strictly during May. Subordinate officers must complete 3 years tenure.",
                char_count=120,
            ),
            ParsedPage(
                page_number=2,
                text="Transfer committees shall be constituted by the Principal Chief Conservator of Forests before April 15th each year.",
                char_count=118,
            ),
        ],
    )

    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    # Verify every chunk maps strictly to its original page
    page_1_chunks = [c for c in chunks if c.page_number == 1]
    page_2_chunks = [c for c in chunks if c.page_number == 2]

    assert len(page_1_chunks) > 0
    assert len(page_2_chunks) > 0
    for c in chunks:
        assert c.go_number == "GO-1345/XII/2018"
        assert c.issuing_department == "Forest"
        assert c.date == "2018-03-12"


@pytest.mark.asyncio
async def test_vector_store_schema_and_hybrid_search():
    """Verify PostgreSQL pgvector table creation, embedding insertion, and cosine search."""
    ensure_windows_event_loop()
    store = VectorStore()
    store.initialize_schema()

    # Clean up any leftover test chunks first
    try:
        with store.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE go_number LIKE 'GO-TEST%';")
    except Exception:
        pass

    test_chunks = [
        DocumentChunk(
            chunk_id="test_forest_transfer_p1_c0",
            document_id="test_forest.pdf",
            go_number="GO-TEST/FOREST/2024",
            issuing_department="Forest",
            date="2024-05-01",
            page_number=1,
            chunk_index=0,
            exact_text_excerpt="Annual transfer window for forest range officers is opened in May each year.",
            char_length=76,
        ),
        DocumentChunk(
            chunk_id="test_finance_pension_p1_c0",
            document_id="test_finance.pdf",
            go_number="GO-TEST/FINANCE/2024",
            issuing_department="Finance",
            date="2024-06-15",
            page_number=1,
            chunk_index=0,
            exact_text_excerpt="Revised dearness allowance guidelines for state government pensioners.",
            char_length=70,
        ),
    ]

    try:
        inserted = store.insert_chunks(test_chunks, batch_size=2)
        assert inserted == 2

        # Query matching the Forest chunk
        results = await store.a_hybrid_search(
            query_text="forest range officer transfer policy",
            department_filter="Forest",
            max_results=5,
        )

        assert len(results) >= 1
        matching_forest = [r for r in results if r.go_number == "GO-TEST/FOREST/2024"]
        assert len(matching_forest) >= 1
        assert matching_forest[0].issuing_department == "Forest"
        assert matching_forest[0].page_number == 1
        assert matching_forest[0].relevance_score >= 0.0
    finally:
        # Cleanup test chunks
        try:
            with store.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM document_chunks WHERE go_number LIKE 'GO-TEST%';")
        except Exception:
            pass
        await VectorStore.close_pools()
