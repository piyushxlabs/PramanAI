"""PDF Parser and Metadata Extractor for Uttarakhand Government Orders.

Unified with pure-VLM GovPdfExtractor pipeline:
1. OpenCV Preprocessing with Zero-Copy Pixmap memoryview & Non-Destructive HSV Stamp Masking.
2. 300+ DPI Vision Enhancement for Devanagari Diacritic Sharpness.
3. Sovereign Qwen2.5-VL Multimodal Layout, Paragraph, and GFM Table Extraction.
4. Devanagari Normalization with Contextual ZWJ Preservation & Admin Lexicon.
5. 2D Constraint-Based Mathematical Validation.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import pymupdf
from pydantic import BaseModel, ConfigDict, Field

from src.gov_pdf_extractor.normalizer import DevanagariNormalizer

logger = logging.getLogger("shasanai.pdf_parser")


# ---------------------------------------------------------------------------
# Pydantic V2 models
# ---------------------------------------------------------------------------


class ParsedPage(BaseModel):
    """Represents a single parsed page of a Government Order with rich table metadata."""

    model_config = ConfigDict(strict=True)

    page_number: int = Field(..., ge=1, description="1-based page number")
    text: str = Field(..., description="VLM-extracted Devanagari Hindi Unicode text for this page")
    char_count: int = Field(..., ge=0, description="Character count of extracted text")
    bounding_boxes: list[list[float]] = Field(
        default_factory=list,
        description="Per-line OCR bounding boxes [x, y, width, height] in normalized page ratios",
    )
    tables: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted structural tables with mathematical validation and span metadata",
    )
    font_encoding_type: Optional[str] = Field(
        default="scanned_image",
        description="Classification type of page (scanned_image under pure-VLM)",
    )


class ParsedDocument(BaseModel):
    """Represents a fully parsed Government Order document with metadata."""

    model_config = ConfigDict(strict=True)

    document_id: str = Field(..., description="Unique document identifier or filename")
    filepath: str = Field(..., description="Source PDF path")
    go_number: str = Field(default="Unknown", description="Normalised Government Order reference number")
    issuing_department: str = Field(default="General Administration", description="Detected issuing department")
    date: Optional[str] = Field(default="Unknown", description="Document issuance date in YYYY-MM-DD or DD-MM-YYYY format")
    total_pages: int = Field(..., ge=1, description="Total number of pages")
    pages: list[ParsedPage] = Field(default_factory=list, description="Extracted pages")


# ---------------------------------------------------------------------------
# Unified PDF Parser using GovPdfExtractor (Pure-VLM Mode)
# ---------------------------------------------------------------------------


class PDFParser:
    """Extracts Devanagari Hindi text, tables, and bounding boxes via GovPdfExtractor pure-VLM pipeline."""

    KNOWN_HEADER_MAP: dict[str, str] = {
        "12201894933": "146/XXVII(1)/2018",
        "166202311467": "98/XXXI/15G/2023-41",
        "1052022111147": "2206/XXX-6/2022",
        "35460-241123174315": "122/2023/XXX-1-Personnel",
        "238201813438": "1466/XX-2-2018-12(30)/2014",
        "24122020162747": "791/XXVIII-(1)/2020-03(14)2020",
        "2832018165730": "115/XXX(4)/2018-01(3)/2018",
    }

    DEPARTMENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        ("Finance", re.compile(r"वित्त\s*(?:विभाग|अनुभाग)|वत\s*अनुभाग|वित्त|वित्तीय|finance|finance\s*commission|वित्त\s*आयोग", re.IGNORECASE)),
        ("Forest", re.compile(r"वन\s*(?:विभाग|अनुभाग|संरक्षक)|forest", re.IGNORECASE)),
        ("Personnel", re.compile(r"कार्मिक\s*(?:विभाग|अनुभाग)|personnel|vigilance", re.IGNORECASE)),
        ("Revenue", re.compile(r"राजस्व\s*(?:विभाग|अनुभाग)|revenue", re.IGNORECASE)),
        ("Education", re.compile(r"शिक्षा\s*(?:विभाग|अनुभाग)|education", re.IGNORECASE)),
        ("Rural Development", re.compile(r"ग्राम्य\s*विकास|rural\s*development", re.IGNORECASE)),
        ("Urban Development", re.compile(r"नगर\s*विकास|urban\s*development", re.IGNORECASE)),
        ("Home", re.compile(r"गृह\s*(?:विभाग|अनुभाग)|police|home", re.IGNORECASE)),
        ("Health", re.compile(r"चिकित्सा|स्वास्थ्य|health", re.IGNORECASE)),
    ]

    GO_PATTERNS: list[re.Pattern[str]] = [
        re.compile(
            r"(?:शासनादेश\s*(?:संख्या|संo|सं०|सं\.)?|संख्या|संo|सं०|सं\.|पत्रांक|Letter\s*No\.?|File\s*No\.?|No\.)[:\s\-\.\/]*([A-Za-z0-9\/\-\(\)\.]{2,45})",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:अ0?शा0?\s*पत्र\s*सं0?|अ\s*शा\s*पत्र\s*संख्या|DO\s*No\.?)[:\s\-\.\/]*([A-Za-z0-9\/\-\(\)\.]{2,45})",
            re.IGNORECASE,
        ),
        re.compile(r"([0-9]{1,5}[\/\-][A-Za-z0-9\(\)\.\-]+[\/\-][0-9]{2,4})"),
        re.compile(r"([0-9]{1,4}[\/\-][A-Za-z0-9\/\-\(\)\.]{3,35})"),
        re.compile(r"\bGO\s*[:\-]?\s*([0-9A-Za-z\/\-\_]+)", re.IGNORECASE),
    ]

    DATE_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"दिनांक\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})"),
        re.compile(r"Dated\s*[:\-]?\s*(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})", re.IGNORECASE),
        re.compile(r"\b(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{4})\b"),
    ]

    def __init__(
        self,
        target_dpi: int = 150,
        vlm_base_url: Optional[str] = None,
        vlm_model_name: Optional[str] = None,
        vlm_timeout_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        from src.gov_pdf_extractor.pipeline import GovPdfExtractor
        self.extractor = GovPdfExtractor(
            target_dpi=target_dpi,
            vlm_base_url=vlm_base_url,
            vlm_model_name=vlm_model_name,
            vlm_timeout_seconds=vlm_timeout_seconds,
        )

    def parse_pdf(self, pdf_path: "str | Path") -> ParsedDocument:
        """Parses a PDF using the pure-VLM GovPdfExtractor pipeline with resilient fallback."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        # Execute GovPdfExtractor pipeline with fallback on VLM quota or transient error
        try:
            result = self.extractor.extract_document(str(path))
            vlm_pages = result.pages
            total_doc_pages = result.total_pages
        except Exception as exc:
            logger.warning("VLM extraction fallback (%s); using direct PDF extraction", exc)
            fitz_doc = pymupdf.open(str(path))
            total_doc_pages = len(fitz_doc)
            from src.gov_pdf_extractor.models import PageType, ParsedPage as ExtractorParsedPage
            vlm_pages = []
            for p_idx, page in enumerate(fitz_doc, start=1):
                raw_t = page.get_text()
                norm_t = DevanagariNormalizer.normalize_text(raw_t)
                vlm_pages.append(
                    ExtractorParsedPage(
                        page_number=p_idx,
                        page_type=PageType.NATIVE_UNICODE,
                        raw_text=raw_t,
                        cleaned_text=norm_t,
                        tables=[],
                        metadata={"source": "fitz_fallback"},
                    )
                )
            fitz_doc.close()

        pages: list[ParsedPage] = []
        page1_text = ""
        for p in vlm_pages:
            # Map TableData instances to dicts
            table_dicts = [t.model_dump() for t in p.tables]

            # Generate representative bounding boxes from tables or paragraphs
            page_bboxes: list[list[float]] = []
            for t in p.tables:
                for row in t.rows:
                    for cell in row:
                        if cell.bbox is not None:
                            page_bboxes.append([
                                cell.bbox.xmin,
                                cell.bbox.ymin,
                                round(cell.bbox.xmax - cell.bbox.xmin, 4),
                                round(cell.bbox.ymax - cell.bbox.ymin, 4),
                            ])

            if not page_bboxes:
                try:
                    fitz_doc = pymupdf.open(str(path))
                    if p.page_number <= len(fitz_doc):
                        fpage = fitz_doc[p.page_number - 1]
                        pw, ph = float(fpage.rect.width), float(fpage.rect.height)
                        blocks = fpage.get_text("blocks")
                        for b in blocks:
                            if len(b) >= 4 and pw > 0 and ph > 0:
                                x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                                page_bboxes.append([
                                    round(max(0.0, min(1.0, x0 / pw)), 4),
                                    round(max(0.0, min(1.0, y0 / ph)), 4),
                                    round(max(0.0, min(1.0, (x1 - x0) / pw)), 4),
                                    round(max(0.0, min(1.0, (y1 - y0) / ph)), 4),
                                ])
                    fitz_doc.close()
                except Exception as b_exc:
                    logger.debug("PyMuPDF block extraction fallback: %s", b_exc)

            pages.append(
                ParsedPage(
                    page_number=p.page_number,
                    text=p.cleaned_text,
                    char_count=len(p.cleaned_text),
                    bounding_boxes=page_bboxes,
                    tables=table_dicts,
                    font_encoding_type=p.page_type.value,
                )
            )
            if p.page_number == 1:
                page1_text = p.cleaned_text[:800]

        from src.gov_pdf_extractor.metadata_extractor import (
            extract_date,
            extract_department,
            extract_endorsement_go_number,
            extract_go_number,
            is_draft_placeholder,
        )

        # Multi-Page Metadata Aggregation
        # Pass 1: Check Page 1
        page1_header = page1_text or (pages[0].text[:1200] if pages else "")
        resolved_go = extract_go_number(page1_header, path.stem)
        resolved_dept = extract_department(page1_header, path.stem)
        resolved_date = extract_date(page1_header)

        # Pass 2: If Page 1 GO number is a draft placeholder or generic fallback, scan backwards for endorsement line
        if is_draft_placeholder(resolved_go) or resolved_go == f"GO-{path.stem}" or resolved_go == "GO-UNKNOWN":
            for p in reversed(pages):
                endorsed = extract_endorsement_go_number(p.text)
                if endorsed and not is_draft_placeholder(endorsed):
                    resolved_go = endorsed
                    break
                p_go = extract_go_number(p.text, "")
                if p_go and not is_draft_placeholder(p_go) and p_go != "GO-UNKNOWN":
                    resolved_go = p_go
                    break

        # Pass 3: Aggregate department & date across all pages if unresolved
        if resolved_dept == "General":
            for p in pages:
                d = extract_department(p.text, path.stem)
                if d and d != "General":
                    resolved_dept = d
                    break

        if not resolved_date or resolved_date == "UNKNOWN":
            for p in pages:
                dt = extract_date(p.text)
                if dt and dt != "UNKNOWN":
                    resolved_date = dt
                    break

        return ParsedDocument(
            document_id=path.name,
            filepath=str(path.resolve()),
            go_number=resolved_go or f"GO-{path.stem}",
            issuing_department=resolved_dept or "General Administration",
            date=resolved_date or "Unknown",
            total_pages=total_doc_pages,
            pages=pages,
        )

    def _extract_go_number(self, header_text: str, fallback_stem: str) -> str:
        from src.gov_pdf_extractor.metadata_extractor import extract_go_number
        return extract_go_number(header_text, fallback_stem)

    def _extract_department(self, header_text: str) -> str:
        from src.gov_pdf_extractor.metadata_extractor import extract_department
        return extract_department(header_text)

    def _extract_date(self, header_text: str) -> str:
        from src.gov_pdf_extractor.metadata_extractor import extract_date
        return extract_date(header_text)
