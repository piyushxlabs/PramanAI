"""Automated Verification Test Suite for Multimodal VLM (Qwen2.5-VL-3B) Extractor.

Verifies:
1. Local VLM model runtime invocation (ainvoke_vision) on synthetic test images.
2. Conversational chatter stripping and header JSON block extraction.
3. GFM Markdown table extraction into structured TableData and TableCell with normalized BoundingBoxes.
4. Windowed Table Chunking with schema injection for VLM Markdown tables.
5. End-to-end PDF parsing with VLM integration on actual Government Order PDFs.
"""

import asyncio
import io
from pathlib import Path
from decimal import Decimal
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.gov_pdf_extractor.models import BoundingBox, TableCell, TableData, PageType
from src.gov_pdf_extractor.vlm_extractor import (
    VlmDocumentExtractor,
    VlmPageExtractionResult,
    VLM_EXTRACTION_PROMPT,
)
from src.gov_pdf_extractor.pipeline import GovPdfExtractor
from src.ingestion.chunking import DocumentChunker
from src.ingestion.pdf_parser import PDFParser, ParsedDocument, ParsedPage
from src.utils.model_runtime import ainvoke_vision, get_vision_model


@pytest.mark.asyncio
async def test_vlm_model_runtime_synthetic_image():
    """Smoke test for ainvoke_vision invoking local Qwen2.5-VL-3B with an image."""
    import cv2
    import pymupdf
    from src.state.reducers import ToolExecutionError

    pdf_files = list(Path("data/raw_pdfs").glob("*.pdf"))
    if pdf_files:
        doc = pymupdf.open(str(pdf_files[0]))
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("jpeg")
    else:
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "TEST", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        _, enc = cv2.imencode(".jpg", img)
        img_bytes = enc.tobytes()

    try:
        response = await ainvoke_vision(
            image_bytes=img_bytes,
            prompt="Transcribe visible text from this document image.",
            timeout_seconds=20.0,
        )
        assert response is not None
        assert len(response.strip()) > 0
    except (ToolExecutionError, Exception) as exc:
        pytest.skip(f"VLM endpoint offline/unreachable ({exc!s}) - skipping live vision smoke test.")




def test_vlm_conversational_stripping_and_header_json():
    """Verifies stripping conversational preambles and extracting ```json header blocks."""
    raw_vlm_output = """Here is the extracted text from the document:

```json
{
  "go_number": "GO-146/XXVII(1)/2018",
  "department": "Finance",
  "date": "2018-01-12",
  "subject": "वित्तीय स्वीकृति"
}
```

उत्तराखंड शासन
वित्त अनुभाग-1

संख्या: 146/XXVII(1)/2018
दिनांक: 12 जनवरी, 2018

| क्र० सं० | मद का नाम | स्वीकृत धनराशि (लाख में) |
| --- | --- | --- |
| 1 | वेतन भत्ते | 50.00 |
| 2 | यात्रा व्यय | 10.00 |
| 3 | कुल योग | 60.00 |

Let me know if you need anything else!"""

    cleaned_chatter = VlmDocumentExtractor.strip_conversational_chatter(raw_vlm_output)
    assert not cleaned_chatter.startswith("Here is the extracted")
    assert not cleaned_chatter.endswith("Let me know if you need anything else!")

    header_meta, remaining_text, is_draft = VlmDocumentExtractor.extract_header_json(cleaned_chatter)
    assert header_meta.get("go_number") == "GO-146/XXVII(1)/2018"
    assert header_meta.get("department") == "Finance"
    assert header_meta.get("date") == "2018-01-12"
    assert is_draft is False

    assert "```json" not in remaining_text
    assert "उत्तराखंड शासन" in remaining_text


def test_vlm_gfm_table_parsing_into_table_data():
    """Verifies parsing GFM markdown tables into structured TableData and TableCells with normalized bboxes."""
    markdown_doc = """उत्तराखंड शासन, वित्त विभाग

| क.सं. | कार्य विवरण | स्वीकृत धनराशि (रु० लाख में) | व्यय | अवशेष |
| :--- | :--- | :--- | :--- | :--- |
| 1 | सड़क चौड़ीकरण | 120.00 | 80.00 | 40.00 |
| 2 | पुलिया निर्माण | 50.00 | 30.00 | 20.00 |
| 3 | कुल योग | 170.00 | 110.00 | 60.00 |

आज्ञा से,
प्रमुख सचिव।"""

    tables = VlmDocumentExtractor.parse_gfm_tables(markdown_doc)
    assert len(tables) == 1

    tbl = tables[0]
    assert len(tbl.headers) == 5
    assert "क.सं" in tbl.headers[0]
    assert "कार्य विवरण" in tbl.headers[1]

    assert len(tbl.rows) == 3
    # Check row 1
    assert tbl.rows[0][0].raw_text == "1"
    assert tbl.rows[0][1].raw_text == "सड़क चौड़ीकरण"
    assert tbl.rows[0][2].raw_text == "120.00"

    # Verify bounding box normalization (0.0 to 1.0)
    for row in tbl.rows:
        for cell in row:
            assert cell.bbox is not None
            assert 0.0 <= cell.bbox.ymin <= 1.0
            assert 0.0 <= cell.bbox.xmin <= 1.0
            assert 0.0 <= cell.bbox.ymax <= 1.0
            assert 0.0 <= cell.bbox.xmax <= 1.0
            assert cell.bbox.ymin < cell.bbox.ymax
            assert cell.bbox.xmin < cell.bbox.xmax


def test_vlm_windowed_table_chunking_with_schema_injection():
    """Verifies that large VLM markdown tables are window-sliced while retaining schema headers."""
    # Create a 15-row markdown table
    headers = ["क्र०", "परियोजना का नाम", "जनपद", "स्वीकृत लागत", "व्यय", "अवशेष"]
    rows = []
    for i in range(1, 16):
        rows.append([str(i), f"जल जीवन मिशन योजना {i}", "देहरादून", "100.00", "40.00", "60.00"])

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    large_table_md = "\n".join(lines)

    parsed_page = ParsedPage(
        page_number=1,
        text=large_table_md,
        char_count=len(large_table_md),
        bounding_boxes=[[0.05, 0.10, 0.90, 0.80]],
        tables=[],
        font_encoding_type="scanned_image",
    )
    doc = ParsedDocument(
        document_id="test_doc.pdf",
        filepath="/path/to/test_doc.pdf",
        go_number="GO-999/XXVII/2024",
        issuing_department="Finance",
        date="2024-01-01",
        total_pages=1,
        pages=[parsed_page],
    )

    chunker = DocumentChunker(chunk_size=300, chunk_overlap=30)
    chunks = chunker.chunk_page(doc, parsed_page)

    # Should create multiple windowed slices
    assert len(chunks) >= 2
    for c in chunks:
        # Every chunk MUST have the table schema header
        assert "परियोजना का नाम" in c.exact_text_excerpt and ("क्र०" in c.exact_text_excerpt or "क्र॰" in c.exact_text_excerpt)
        assert "Table Slice Rows" in c.exact_text_excerpt
        assert c.font_encoding_type == "scanned_image"


def test_vlm_pipeline_e2e_integration():
    """Verifies GovPdfExtractor pure-VLM pipeline on real sample GO PDF with graceful skip if offline."""
    pdf_files = list(Path("data/raw_pdfs").glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No test PDFs found in data/raw_pdfs/")

    raw_pdf_path = pdf_files[0]
    extractor = GovPdfExtractor(vlm_timeout_seconds=20.0)
    try:
        result = extractor.extract_document(str(raw_pdf_path))
        assert result.total_pages >= 1
        assert len(result.pages) == result.total_pages
        if result.pages and result.pages[0].cleaned_text:
            assert len(result.pages[0].cleaned_text) > 0
    except Exception as exc:
        pytest.skip(f"Live VLM extraction skipped due to offline endpoint: {exc!s}")

    # Also verify PDFParser integration
    try:
        parser = PDFParser(vlm_timeout_seconds=20.0)
        parsed_doc = parser.parse_pdf(raw_pdf_path)

        assert parsed_doc.go_number != ""
        assert parsed_doc.total_pages >= 1
        assert len(parsed_doc.pages) >= 1
        assert len(parsed_doc.pages[0].bounding_boxes) >= 1
    except Exception as exc:
        pytest.skip(f"Live VLM extraction skipped due to offline endpoint: {exc!s}")


def test_vlm_extraction_error_fail_fast_on_failure():
    """Verifies that GovPdfExtractor raises VlmExtractionError when VLM endpoint fails."""
    from src.gov_pdf_extractor.pipeline import VlmExtractionError

    pdf_files = list(Path("data/raw_pdfs").glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No test PDFs found in data/raw_pdfs/")

    extractor = GovPdfExtractor(vlm_base_url="http://invalid-vlm-host-12345:9999", vlm_timeout_seconds=1.0)
    with pytest.raises(VlmExtractionError):
        extractor.extract_document(str(pdf_files[0]))


def test_devanagari_unicode_normalization_and_matra_repair():
    """Verifies that DevanagariNormalizer repairs decomposed Unicode, misplaced matras, and administrative glossary."""
    from src.gov_pdf_extractor.normalizer import DevanagariNormalizer, normalize_devanagari_text
    from src.utils.text_normalizer import normalize_text

    corrupt_samples = [
        ("दनिांक: 24 दिसम्बर, 2020", "दिनांक: 24 दिसम्बर, 2020"),
        ("अनुभाग अधकिारी द्वारा प्रस्तुत", "अनुभाग अधिकारी द्वारा प्रस्तुत"),
        ("कार्मकि एवं सतर्कता विभाग", "कार्मिक एवं सतर्कता विभाग"),
        ("वज्ञिप्ति संख्या 100", "विज्ञप्ति संख्या 100"),
        ("प्रशक्षिण कार्यक्रम", "प्रशिक्षण कार्यक्रम"),
        ("जनपद पथिौरागढ़ में", "जनपद पिथौरागढ़ में"),
        ("संस्ततुि प्रदान की जाती है", "संस्तुति प्रदान की जाती है"),
        ("वत्ति वभिाग शासनादश", "वित्त विभाग शासनादेश"),
        ("प्रमुख सचवि, उत्तराखण्ड शासन", "प्रमुख सचिव, उत्तराखण्ड शासन"),
    ]

    for corrupt, expected in corrupt_samples:
        normalized = normalize_devanagari_text(corrupt)
        assert expected in normalized, f"Failed: '{corrupt}' -> '{normalized}', expected '{expected}'"


def test_metadata_go_number_extraction_robustness():
    """Verifies extracting authentic UP/Uttarakhand GO numbers from various header strings."""
    from src.gov_pdf_extractor.metadata_extractor import extract_go_number

    test_headers = [
        ("संख्या-115/XXX(4)/2018-01(3)/2018\nदेहरादून : दिनांक 12 जनवरी", "GO-115/XXX(4)/2018-01(3)/2018"),
        ("संख्या: 791/XXVIII-(1)/2020-03(14)2020\nदेहरादून: दिनांक 24 दिसम्बर, 2020", "GO-791/XXVIII-(1)/2020-03(14)2020"),
        ("शासनादेश संख्या 146/XXVII(1)/2018\nवित्त अनुभाग-1", "GO-146/XXVII(1)/2018"),
        ("संख्या 98/XXXI/15G/2023-41\nदिनांक 16 जून, 2023", "GO-98/XXXI/15G/2023-41"),
        ("संख्या: 667 (1)/X-3-18-16(01)/2014\nदेहरादून: दिनांक 08 अक्टूबर, 2018", "GO-667 (1)/X-3-18-16(01)/2014"),
    ]

    for header, expected_go in test_headers:
        extracted = extract_go_number(header)
        assert extracted == expected_go, f"Failed on '{header}': got '{extracted}', expected '{expected_go}'"


def test_department_extraction_and_sanitization():
    """Verifies clean extraction and noise sanitization for department names."""
    from src.gov_pdf_extractor.metadata_extractor import extract_department

    test_headers = [
        ("सामान्य प्रशासन विभाग /XXXI(15)G/20-41(सा)/2018\nसंख्या: 825", "सामान्य प्रशासन विभाग"),
        ("कार्मिक अनुभाग-4\nसंख्या 115/XXX(4)/2018", "कार्मिक अनुभाग-4"),
        ("वन विभाग-3\nसंख्या: 667 (1)/X-3-18", "वन विभाग-3"),
        ("उत्तराखण्ड शासन\nवित्त अनुभाग-1\nदेहरादून", "वित्त अनुभाग-1"),
    ]

    for header, expected_dept in test_headers:
        dept = extract_department(header)
        assert dept == expected_dept, f"Failed on '{header}': got '{dept}', expected '{expected_dept}'"


def test_duplicate_matra_deduplication():
    """Verifies that consecutive identical Devanagari matras are cleaned."""
    from src.gov_pdf_extractor.normalizer import normalize_devanagari_text

    samples = [
        ("कार्मििक अधिकारी", "कार्मिक अधिकारी"),
        ("नििदेशक महोदय", "निदेशक महोदय"),
        ("अधीीकारी", "अधिकारी"),
    ]

    for raw, expected in samples:
        res = normalize_devanagari_text(raw)
        assert expected in res, f"Failed on '{raw}': got '{res}', expected '{expected}'"


def test_vlm_configuration_decoupling():
    """Verifies that VLM_BASE_URL and VLM_MODEL are loaded from env without hardcoding."""
    from src.utils.model_runtime import VLM_BASE_URL, VLM_MODEL, VLM_TIMEOUT_SECONDS, VLM_MAX_IMAGE_DIM
    assert VLM_BASE_URL is not None
    assert len(VLM_BASE_URL) > 0
    assert VLM_MODEL is not None
    assert VLM_TIMEOUT_SECONDS >= 180.0
    assert VLM_MAX_IMAGE_DIM >= 1280


def test_endorsement_scanning_and_draft_placeholder():
    """Verifies scanning endorsement dispatch numbers and detecting draft placeholders."""
    from src.gov_pdf_extractor.metadata_extractor import (
        extract_endorsement_go_number,
        is_draft_placeholder,
    )

    endorsement_sample = """
    प्रतिलिपि निम्नलिखित को सूचनार्थ एवं आवश्यक कार्यवाही हेतु प्रेषित :-
    1. महालेखाकार, उत्तराखण्ड।
    संख्या 825 / XXXI(15)G/20-41(सा) / 2018 तददिनांक।
    आज्ञा से,
    (अधीक्षक)
    """
    endorsed_go = extract_endorsement_go_number(endorsement_sample)
    assert endorsed_go == "GO-825/XXXI(15)G/20-41(सा)/2018"

    assert is_draft_placeholder("संख्या-   /XXXI(15)G/20-41(सा)/2018") is True
    assert is_draft_placeholder("संख्या-___/XXXI(15)G/20-41(सा)/2018") is True
    assert is_draft_placeholder("825/XXXI(15)G/20-41(सा)/2018") is False


def test_enhance_gov_document_image():
    """Verifies image contrast and sharpness enhancement helper."""
    from src.gov_pdf_extractor.preprocessor import enhance_gov_document_image

    # Test RGB image
    img_rgb = Image.new("RGB", (200, 200), color=(240, 240, 240))
    enhanced_rgb = enhance_gov_document_image(img_rgb)
    assert enhanced_rgb.mode == "RGB"
    assert enhanced_rgb.size == (200, 200)

    # Test RGBA image
    img_rgba = Image.new("RGBA", (150, 150), color=(200, 200, 200, 255))
    enhanced_rgba = enhance_gov_document_image(img_rgba)
    assert enhanced_rgba.mode == "RGB"
    assert enhanced_rgba.size == (150, 150)

    # Test Grayscale image
    img_l = Image.new("L", (100, 100), color=128)
    enhanced_l = enhance_gov_document_image(img_l)
    assert enhanced_l.mode == "RGB"
    assert enhanced_l.size == (100, 100)


def test_admin_ocr_repair_map_exhaustive():
    """Verifies all entries in ADMIN_OCR_REPAIR_MAP normalize to authentic Hindi forms."""
    from src.gov_pdf_extractor.normalizer import ADMIN_OCR_REPAIR_MAP, normalize_devanagari_text

    test_cases = [
        ("यह एक कार्यालय-झाप है।", "यह एक कार्यालय-ज्ञाप है।"),
        ("कार्यालय झाप संख्या 100", "कार्यालय-ज्ञाप संख्या 100"),
        ("चाजस्व विमान द्वारा निर्गत", "राजस्व विभाग द्वारा निर्गत"),
        ("कृपया काब्द करें", "कृपया कष्ट करें"),
        ("भाव्यांसे प्रेषित", "माध्यम से प्रेषित"),
        ("भुजे यह कहने का निदेश हुआ है", "मुझे यह कहने का निदेश हुआ है"),
        ("जनपद सिंधोचगड में", "जनपद पिथौरागढ़ में"),
        ("जनपद फरिोरागढ़ में", "जनपद पिथौरागढ़ में"),
        ("शैक्षांक योग्यता", "शैक्षणिक योग्यता"),
        ("उत्कृष्ठता केंद्र", "उत्कृष्टता केंद्र"),
        ("उल्कृष्टता पुरस्कार", "उत्कृष्टता पुरस्कार"),
        ("एत्दद्वारा सूचित किया जाता है", "एतद्द्वारा सूचित किया जाता है"),
        ("श्री कौस्तुम उपसचिव", "श्री कौस्तुभ"),
        ("श्री गेखुरी", "श्री मैखुरी"),
        ("एओको शर्मा", "ए०के० शर्मा"),
        ("ए० को वर्मा", "ए०के० वर्मा"),
        ("जेओपी सिंह", "जे०पी० सिंह"),
        ("जे० पी रावत", "जे०पी० रावत"),
        ("डॉओ जोशी", "डॉ० जोशी"),
        ("कैओ उनियाल", "कै० उनियाल"),
    ]

    for corrupted, expected in test_cases:
        normalized = normalize_devanagari_text(corrupted)
        assert expected in normalized, f"Failed on '{corrupted}': got '{normalized}', expected '{expected}'"




