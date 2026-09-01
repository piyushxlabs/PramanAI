"""gov_pdf_extractor - 5-Layer Defense-in-Depth Document Ingestion & Parsing Pipeline.

Engineered specifically for Indian Government gazettes, financial budgets,
and administrative orders (शासनादेश).
"""

from src.gov_pdf_extractor.models import (
    BoundingBox,
    DocumentExtractionResult,
    PageType,
    ParsedPage,
    TableCell,
    TableData,
)
from src.gov_pdf_extractor.normalizer import DevanagariNormalizer
from src.gov_pdf_extractor.pipeline import GovPdfExtractor
from src.gov_pdf_extractor.preprocessor import ImagePreprocessor, enhance_gov_document_image
from src.gov_pdf_extractor.table_extractor import TableExtractor
from src.gov_pdf_extractor.triage import DocumentTriageEngine, KrutiDevToUnicodeConverter
from src.gov_pdf_extractor.validator import MathValidator
from src.gov_pdf_extractor.vlm_extractor import VlmDocumentExtractor, VlmPageExtractionResult

__all__ = [
    "GovPdfExtractor",
    "VlmDocumentExtractor",
    "VlmPageExtractionResult",
    "DocumentTriageEngine",
    "KrutiDevToUnicodeConverter",
    "ImagePreprocessor",
    "enhance_gov_document_image",
    "TableExtractor",
    "DevanagariNormalizer",
    "MathValidator",
    "PageType",
    "BoundingBox",
    "TableCell",
    "TableData",
    "ParsedPage",
    "DocumentExtractionResult",
]

