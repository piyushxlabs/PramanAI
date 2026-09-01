"""CLI Tool and Ingestion Engine re-export wrapper.

Exposes run_ingestion_pipeline, DocumentChunker, PDFParser, and KrutiDev-to-Unicode conversion.
"""

from src.ingestion.krutidev import convert_if_krutidev, is_likely_krutidev, krutidev_to_unicode
from src.ingestion.pdf_parser import PDFParser, ParsedDocument, ParsedPage
from src.ingestion.run_ingestion import run_ingestion_pipeline
from src.ingestion.vector_store import VectorStore

__all__ = [
    "convert_if_krutidev",
    "is_likely_krutidev",
    "krutidev_to_unicode",
    "PDFParser",
    "ParsedDocument",
    "ParsedPage",
    "run_ingestion_pipeline",
    "VectorStore",
]

if __name__ == "__main__":
    from src.ingestion.run_ingestion import main
    main()
