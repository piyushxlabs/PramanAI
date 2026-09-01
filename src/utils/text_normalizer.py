"""Universal Devanagari text normalization and matra repair utility for ShasanAI.

Provides canonical NFC normalization, misplaced 'ि' (\u093F) phoneme reordering,
and 500+ Uttarakhand administrative governance glossary corrections.
"""

from src.gov_pdf_extractor.normalizer import (
    ADMIN_OCR_REPAIR_MAP,
    DevanagariNormalizer,
    normalize_devanagari_text,
)

__all__ = [
    "DevanagariNormalizer",
    "normalize_devanagari_text",
    "normalize_text",
    "ADMIN_OCR_REPAIR_MAP",
]


def normalize_text(text: str) -> str:
    """Standardizes Devanagari text using DevanagariNormalizer.normalize_text."""
    return normalize_devanagari_text(text)
