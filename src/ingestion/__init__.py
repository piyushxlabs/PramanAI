"""ShasanAI Document Ingestion & Vector Indexing Pipeline.

Provides air-gapped PDF parsing, page-level chunking, pgvector embedding storage (bge-m3),
and hybrid similarity retrieval for Uttarakhand Government Orders.
"""

from src.ingestion.pdf_parser import ParsedDocument, ParsedPage, PDFParser
from src.ingestion.chunking import DocumentChunk, DocumentChunker
from src.ingestion.vector_store import VectorStore

__all__ = [
    "PDFParser",
    "ParsedPage",
    "ParsedDocument",
    "DocumentChunker",
    "DocumentChunk",
    "VectorStore",
]
