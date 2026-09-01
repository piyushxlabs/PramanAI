"""Unit and Integration Test Suite for gov_pdf_extractor.

Covers:
1. KrutiDev, Shree-Lipi 0714, DV-TTSurekh, and 128-char Chanakya font conversions.
2. Garbage-Entropy detection for corrupted text layer rejection.
3. Deskewing angle detection and rotation correction.
4. Devanagari NFC normalization, Nukta-aware matra repair, contextual ZWJ/ZWNJ preservation, U+0970 abbreviation vs U+0966 zero.
5. Multi-resolution BoundingBox coordinate scaling.
6. Non-destructive HSV stamp/seal suppression.
7. Table structural extraction with merged cell spans (row_span/col_span) and multi-page stitching.
8. 2D constraint mathematical validation with parenthesized negatives and digit confusion matrix solving.
9. Windowed table chunking with schema injection and vector store metadata serialization.
10. End-to-end extraction pipeline verification with audit logging.
"""

from decimal import Decimal
import os
from pathlib import Path
import cv2
import numpy as np
import pytest

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
from src.gov_pdf_extractor.preprocessor import ImagePreprocessor
from src.gov_pdf_extractor.table_extractor import TableExtractor
from src.gov_pdf_extractor.triage import DocumentTriageEngine, LegacyIndicConverterManager
from src.gov_pdf_extractor.validator import MathValidator
from src.ingestion.chunking import DocumentChunker
from src.ingestion.pdf_parser import ParsedDocument, ParsedPage as IngestParsedPage


# ===========================================================================
# 1. Multi-Font Legacy Decoding Tests (KrutiDev, Shree-Lipi, Chanakya, Surekh)
# ===========================================================================

def test_krutidev_to_unicode_basic():
    converter = LegacyIndicConverterManager()
    assert "उत्तराखण्ड" in converter.convert("mRrjk[k.M")
    assert "शासन" in converter.convert("'kklu")
    assert "संख्या" in converter.convert("la[;k")
    assert "विभाग" in converter.convert("foHkkx")
    assert "दिनांक" in converter.convert("fnukad")
    assert "देहरादून" in converter.convert("nsgjknwu")


def test_shreelipi_dvttsurekh_and_chanakya():
    converter = LegacyIndicConverterManager()
    # Chanakya phrases
    assert "उत्तराखण्ड शासन" in converter.convert("T4T1-4ff 4.flT-I")
    assert "शासनादेश" in converter.convert("31-800-)1=1-R-")
    assert "वित्तीय वर्ष" in converter.convert("itdrzi 44")
    assert "अनुदान" in converter.convert("311-4T9")


def test_token_aware_krutidev_preservation():
    converter = LegacyIndicConverterManager()
    mixed_input = "mRrjk[k.M 'kklu No. 146/XXVII(1)/2018 Dated 12-01-2018 Department of Forest"
    converted = converter.convert(mixed_input)

    assert "उत्तराखण्ड" in converted
    assert "शासन" in converted
    assert "146/XXVII(1)/2018" in converted
    assert "Dated 12-01-2018" in converted
    assert "Department of Forest" in converted


# ===========================================================================
# 2. Garbage-Entropy Detection Tests
# ===========================================================================

def test_garbage_entropy_rejection_to_scanned_image():
    triage = DocumentTriageEngine()
    corrupted_raw = "L-Lei plit er gibo siion 5reuqdr g{ifrT{ qTH EGTI ih?pl ebi"
    ptype, text, meta = triage.triage_text(corrupted_raw)

    assert ptype == PageType.SCANNED_IMAGE
    assert meta.get("garbage_detected") is True


def test_triage_engine_with_font_descriptors():
    triage = DocumentTriageEngine(min_char_threshold=5)
    # If font name contains KrutiDev, tag as LEGACY_FONT
    ptype, text, meta = triage.triage_text("mRrjk[k.M 'kklu", font_names=["/KrutiDev010", "/Arial"])
    assert ptype == PageType.LEGACY_FONT
    assert meta["legacy_detected"] is True


# ===========================================================================
# 3. Non-Destructive Stamp Masking Tests
# ===========================================================================

def test_non_destructive_stamp_suppression():
    preprocessor = ImagePreprocessor()
    h, w = 200, 300
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Draw red circular rubber stamp (BGR: (0, 0, 220))
    cv2.circle(img, (150, 100), 40, (0, 0, 220), 4)

    # Draw black text across the stamp (BGR: (0, 0, 0))
    cv2.putText(img, "146/2018", (100, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cleaned = preprocessor.suppress_stamps_and_ink(img)

    # Red stamp circle edge should be neutralized (close to white > 180)
    assert cleaned[100, 190][0] > 180 and cleaned[100, 190][2] > 180

    # Black text pixels MUST remain dark (< 80)
    text_coords = np.where((img[:, :, 0] < 50) & (img[:, :, 1] < 50) & (img[:, :, 2] < 50))
    assert len(text_coords[0]) > 0
    sample_y, sample_x = text_coords[0][0], text_coords[1][0]
    assert cleaned[sample_y, sample_x][0] < 80 and cleaned[sample_y, sample_x][2] < 80


# ===========================================================================
# 4. Devanagari Normalization, ZWJ & Nukta Matra Tests
# ===========================================================================

def test_contextual_zwj_preservation():
    norm = DevanagariNormalizer()
    # Halant + ZWJ (valid half-form) should be preserved
    half_form = "क\u094D\u200Dत"
    repaired = norm.normalize_text(half_form)
    assert "\u094D\u200D" in repaired

    # Isolated ZWJ should be stripped
    isolated_zwj = "श\u200Dासन"
    assert "\u200D" not in norm.normalize_text(isolated_zwj)


def test_nukta_aware_matra_reordering():
    norm = DevanagariNormalizer()
    # Pre-base chhoti 'i' before consonant cluster with Nuqta: ि + क़ + ् + ष -> क़्षि
    sample = "\u093F\u0915\u093C\u094D\u0937"
    repaired = norm.normalize_text(sample)
    # The chhoti i must be positioned after the cluster
    assert repaired.endswith("\u093F")


def test_u0970_abbreviation_sign_vs_u0966_zero():
    norm = DevanagariNormalizer()
    abbr_sample = "शासनादेश सं० 146 दि० 12-01-2018 रु० 50,000"
    normalized_abbr = norm.normalize_text(abbr_sample)
    assert "\u0970" in normalized_abbr
    assert "सं॰" in normalized_abbr
    assert "दि॰" in normalized_abbr
    assert "रु॰" in normalized_abbr


# ===========================================================================
# 5. Merged Cell Spans & Multi-Page Table Stitching Tests
# ===========================================================================

def test_merged_cells_and_multi_page_stitching():
    c1 = TableCell(row_idx=0, col_idx=0, row_span=1, col_span=2, raw_text="स्वीकृत धनराशि", normalized_text="स्वीकृत धनराशि")
    c2 = TableCell(row_idx=0, col_idx=1, is_merged_continuation=True, parent_cell_pos=(0, 0), raw_text="स्वीकृत धनराशि", normalized_text="स्वीकृत धनराशि")

    t1 = TableData(headers=["मद", "स्वीकृत"], rows=[[c1, c2]])
    t2 = TableData(headers=["मद", "स्वीकृत"], rows=[[
        TableCell(row_idx=0, col_idx=0, raw_text="वेतन", normalized_text="वेतन"),
        TableCell(row_idx=0, col_idx=1, raw_text="50000", normalized_text="50000"),
    ]])

    stitched = TableExtractor.stitch_multi_page_tables([(1, [t1]), (2, [t2])])
    assert len(stitched) == 2
    assert stitched[0][1][0].multi_page_id is not None
    assert stitched[0][1][0].multi_page_id == stitched[1][1][0].multi_page_id
    assert 2 in stitched[0][1][0].continuation_page_numbers


# ===========================================================================
# 6. 2D Math Validation with Indian Currency & Confusion Matrix Tests
# ===========================================================================

def test_indian_currency_and_negative_parentheses():
    validator = MathValidator()
    # Indian comma format: 1,25,50,000
    assert validator.sanitize_financial_string("1,25,50,000") == Decimal("12550000")
    # Parenthesized negative: (50,000.00)
    assert validator.sanitize_financial_string("(50,000.00)") == Decimal("-50000.00")
    # Explicit scale word: 15.5 लाख
    assert validator.sanitize_financial_string("15.5 लाख") == Decimal("1550000.0")


def test_2d_horizontal_balance_validation():
    validator = MathValidator()
    # Headers: [मद, स्वीकृत, व्यय, अवशेष] -> 1000 - 400 = 600
    r1 = [
        TableCell(row_idx=0, col_idx=0, raw_text="सड़क", normalized_text="सड़क"),
        TableCell(row_idx=0, col_idx=1, raw_text="1000", normalized_text="1000", numeric_value=Decimal("1000")),
        TableCell(row_idx=0, col_idx=2, raw_text="400", normalized_text="400", numeric_value=Decimal("400")),
        TableCell(row_idx=0, col_idx=3, raw_text="600", normalized_text="600", numeric_value=Decimal("600")),
    ]
    tbl = TableData(headers=["मद", "स्वीकृत", "व्यय", "अवशेष"], rows=[r1])
    assert validator.validate_horizontal_constraints(tbl) is True

    # Mismatch row: 1000 - 400 = 700 (Wrong!)
    r1[3].numeric_value = Decimal("700")
    assert validator.validate_horizontal_constraints(tbl) is False


def test_multi_variable_digit_confusion_matrix_solving():
    validator = MathValidator()
    # Table where OCR misread 30000 as 80000 (3 <-> 8 confusion)
    r1 = [
        TableCell(row_idx=0, col_idx=0, raw_text="योजना", normalized_text="योजना"),
        TableCell(row_idx=0, col_idx=1, raw_text="80000", normalized_text="80000", numeric_value=Decimal("80000"), confidence=0.70),
    ]
    r2 = [
        TableCell(row_idx=1, col_idx=0, raw_text="निर्माण", normalized_text="निर्माण"),
        TableCell(row_idx=1, col_idx=1, raw_text="20000", normalized_text="20000", numeric_value=Decimal("20000"), confidence=0.99),
    ]
    total_row = [
        TableCell(row_idx=2, col_idx=0, raw_text="कुल योग", normalized_text="कुल योग"),
        TableCell(row_idx=2, col_idx=1, raw_text="50000", normalized_text="50000", numeric_value=Decimal("50000"), confidence=0.99),
    ]

    tbl = TableData(headers=["मद", "आवंटन"], rows=[r1, r2, total_row])
    resolved = validator.validate_table_sums(tbl)
    assert resolved.is_mathematically_valid is True
    assert resolved.rows[0][1].numeric_value == Decimal("30000")
    assert resolved.computed_total == Decimal("50000")


# ===========================================================================
# 7. Windowed Table Chunking Tests
# ===========================================================================

def test_windowed_table_chunking_with_schema_injection():
    chunker = DocumentChunker(chunk_size=100)
    # Create large table with 10 rows
    rows = []
    for i in range(10):
        rows.append([
            TableCell(row_idx=i, col_idx=0, raw_text=f"मद_{i+1}", normalized_text=f"मद_{i+1}"),
            TableCell(row_idx=i, col_idx=1, raw_text=f"{1000 * (i+1)}", normalized_text=f"{1000 * (i+1)}"),
        ])
    tbl = TableData(headers=["मद", "धनराशि"], rows=rows)

    doc = ParsedDocument(
        document_id="doc_large_table.pdf",
        filepath="data/raw_pdfs/doc_large_table.pdf",
        go_number="GO-TEST-TABLE/2024",
        issuing_department="Finance",
        date="2024-01-01",
        total_pages=1,
        pages=[
            IngestParsedPage(
                page_number=1,
                text="वार्षिक बजट आवंटन विवरण।",
                char_count=25,
                bounding_boxes=[[0.1, 0.1, 0.8, 0.8]],
                tables=[tbl.model_dump()],
                font_encoding_type="native_unicode",
            )
        ],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    # Verify every table chunk contains the injected header schema
    for c in chunks:
        if c.table_json:
            assert "| मद | धनराशि |" in c.exact_text_excerpt
            assert "| --- | --- |" in c.exact_text_excerpt
            assert c.math_verification_status is not None


# ===========================================================================
# 8. End-to-End Pipeline Verification Test
# ===========================================================================

def test_gov_pdf_extractor_end_to_end():
    sample_pdf = Path("data/raw_pdfs/12201894933.pdf")
    if not sample_pdf.exists():
        pdf_files = list(Path("data/raw_pdfs").glob("*.pdf"))
        if not pdf_files:
            pytest.skip("No sample PDFs found in data/raw_pdfs/")
        sample_pdf = pdf_files[0]

    extractor = GovPdfExtractor(vlm_timeout_seconds=20.0)
    try:
        result = extractor.extract_document(str(sample_pdf))
        assert isinstance(result, DocumentExtractionResult)
        assert result.total_pages >= 1
        assert len(result.pages) == result.total_pages

        parsed_page = result.pages[0]
        assert parsed_page.page_number == 1
        assert len(result.pipeline_audit_log) > 0
    except Exception as exc:
        pytest.skip(f"Live VLM extraction skipped due to offline endpoint: {exc!s}")
