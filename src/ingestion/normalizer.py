"""Devanagari text normalizer export for ingestion pipeline."""

from src.gov_pdf_extractor.normalizer import (
    ADMIN_OCR_REPAIR_MAP,
    DevanagariNormalizer,
    normalize_devanagari_text,
)

__all__ = ["DevanagariNormalizer", "normalize_devanagari_text", "ADMIN_OCR_REPAIR_MAP"]
