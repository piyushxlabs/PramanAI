"""Document Chunking Strategy for Uttarakhand Government Orders.

Preserves exact page-boundary isolation, hierarchical Markdown section context,
VLM bounding box coordinates, rich table JSON structures, math verification status,
and page-level metadata.
"""

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from src.gov_pdf_extractor.normalizer import DevanagariNormalizer
from src.ingestion.pdf_parser import ParsedDocument, ParsedPage


class DocumentChunk(BaseModel):
    """Represents an atomic, searchable text passage with page-level attribution, visual grounding, and table metadata."""
    model_config = ConfigDict(strict=True)

    chunk_id: str = Field(..., description="Unique chunk identifier: doc_id:page:index")
    document_id: str = Field(..., description="Source document identifier")
    file_path: Optional[str] = Field(default=None, description="Absolute/relative path to source PDF")
    go_number: str = Field(..., description="Government Order identifier")
    issuing_department: str = Field(..., description="Issuing department")
    date: str = Field(..., description="Document date YYYY-MM-DD")
    page_number: int = Field(..., ge=1, description="1-based page number")
    chunk_index: int = Field(..., ge=0, description="Chunk index within page")
    exact_text_excerpt: str = Field(..., description="Verbatim text passage or structured Markdown table")
    char_length: int = Field(..., ge=0, description="Character count")
    bounding_box_coordinates: Optional[list[float]] = Field(
        default=None,
        description="Representative bounding box [x, y, w, h] in normalized page ratios for visual grounding"
    )
    table_json: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured table JSON if this chunk represents a tabular record"
    )
    math_verification_status: Optional[Dict[str, Any]] = Field(
        default=None, description="Arithmetic validation status and metadata (declared total, computed sum, delta)"
    )
    font_encoding_type: Optional[str] = Field(
        default=None, description="Font classification of source page (native_unicode, legacy_font, scanned_image)"
    )


def _is_footer_or_watermark_bbox(bbox: list[float]) -> bool:
    """Detects CamScanner watermark or page footer bboxes (e.g. y > 0.90, [0.70, 0.97, ...])."""
    if not bbox or len(bbox) < 4:
        return False
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    if y >= 0.90:
        return True
    return False


def _filter_content_bboxes(bboxes: list[list[float]]) -> list[list[float]]:
    """Filters out footer / watermark bounding boxes."""
    if not bboxes:
        return []
    content_boxes = [b for b in bboxes if not _is_footer_or_watermark_bbox(b)]
    return content_boxes if content_boxes else bboxes


def assign_chunk_bbox(chunk_text: str, page_idx: int, clause_idx: int, total_clauses: int) -> list[float]:
    """
    Computes a clean, horizontal paragraph bounding box [ymin, xmin, ymax, xmax] (0.0 to 1.0).
    Matches standard PDF canvas overlay expectations.
    """
    # Horizontal page margins (leaving 10% margin on left/right)
    xmin = 0.10
    xmax = 0.90

    total = max(total_clauses, 1)
    vertical_step = 0.80 / total

    # Calculate vertical position
    ymin = max(0.08, min(0.08 + (clause_idx * vertical_step), 0.82))

    # Calculate height based on character volume
    approx_lines = max(1, len(chunk_text) // 75) if chunk_text else 2
    line_height = 0.032
    calculated_height = approx_lines * line_height
    ymax = min(ymin + max(calculated_height, vertical_step * 0.85), 0.92)

    return [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)]


def _compute_clause_bbox(chunk_idx: int, total_chunks: int) -> list[float]:
    """Computes a full-width paragraph bounding box spanning x in [0.10, 0.90]
    and distributes ymin/ymax proportionally down the page based on clause index.
    """
    return assign_chunk_bbox(chunk_text="", page_idx=1, clause_idx=chunk_idx, total_clauses=total_chunks)


def _representative_bbox(bboxes: list[list[float]], chunk_idx: int, total_chunks: int) -> Optional[list[float]]:
    """Assigns representative visual bounding box anchor for a content chunk, ignoring footers."""
    content_boxes = _filter_content_bboxes(bboxes)
    if not content_boxes:
        return _compute_clause_bbox(chunk_idx, total_chunks)
    idx = min(int((chunk_idx / max(total_chunks, 1)) * len(content_boxes)), len(content_boxes) - 1)
    return content_boxes[idx]


def _format_table_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """Formats headers and rows into clean GitHub-flavored Markdown table."""
    if not headers and not rows:
        return ""
    col_count = max(len(headers), max((len(r) for r in rows), default=0))
    norm_headers = [headers[i] if i < len(headers) and headers[i] else f"Col_{i+1}" for i in range(col_count)]

    header_line = "| " + " | ".join(norm_headers) + " |"
    sep_line = "| " + " | ".join(["---"] * col_count) + " |"
    row_lines = []
    for r in rows:
        padded = [str(r[i] if i < len(r) else "").replace("|", "-").strip() for i in range(col_count)]
        row_lines.append("| " + " | ".join(padded) + " |")

    return "\n".join([header_line, sep_line] + row_lines)


def _split_into_clauses_and_paragraphs(text: str) -> list[str]:
    """Splits administrative text into clause-level units based on numbering patterns
    such as (1), (2), (10), 1., 2- or double newlines.
    """
    if not text or not text.strip():
        return []

    import re
    # Lookahead pattern matching (1), (2), (10), 1., 2- at newline/start
    clause_pattern = r"(?=(?:^|\n)\s*(?:\(\d+\)|\d+[\.\-]\s*))"

    try:
        raw_splits = re.split(clause_pattern, text)
        chunks = [c.strip() for c in raw_splits if c and c.strip()]
        return chunks if chunks else [text.strip()]
    except Exception:
        # Fallback to paragraph splitting if regex encounters unexpected edge-cases
        return [p.strip() for p in text.split("\n\n") if p.strip()]


class DocumentChunker:
    """Chunks parsed Government Orders with boundary preservation and Windowed Table Chunking."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 60) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """Splits all pages of a ParsedDocument into searchable DocumentChunk objects."""
        chunks: list[DocumentChunk] = []
        for page in doc.pages:
            page_chunks = self.chunk_page(doc, page)
            chunks.extend(page_chunks)
        return chunks

    def chunk_page(self, doc: ParsedDocument, page: ParsedPage) -> list[DocumentChunk]:
        """Chunks a single page, prioritizing structured table windows and boundary-safe text splits."""
        page_chunks: list[DocumentChunk] = []
        chunk_idx = 0

        # Extract font type if available
        font_type = getattr(page, "font_encoding_type", None) or "native_unicode"
        page_tables = list(getattr(page, "tables", []) or [])

        # If page_tables is empty, check if page.text contains inline GFM Markdown tables
        if not page_tables and "|" in page.text:
            from src.gov_pdf_extractor.vlm_extractor import VlmDocumentExtractor
            detected_tables = VlmDocumentExtractor.parse_gfm_tables(page.text)
            if detected_tables:
                page_tables = [t.model_dump() for t in detected_tables]

        # 1. Process Structured Tables with Windowed Table Chunking
        for tbl_idx, tbl in enumerate(page_tables):
            headers = tbl.get("headers", []) if isinstance(tbl, dict) else getattr(tbl, "headers", [])
            raw_rows = tbl.get("rows", []) if isinstance(tbl, dict) else getattr(tbl, "rows", [])

            # Extract row string matrix
            str_rows: List[List[str]] = []
            for row in raw_rows:
                row_strs = []
                for c in row:
                    txt = c.get("normalized_text", "") if isinstance(c, dict) else getattr(c, "normalized_text", str(c))
                    row_strs.append(txt)
                str_rows.append(row_strs)

            if not str_rows:
                continue

            math_meta = {
                "is_valid": tbl.get("is_mathematically_valid", False) if isinstance(tbl, dict) else getattr(tbl, "is_mathematically_valid", False),
                "declared_total": str(tbl.get("declared_total", "")) if isinstance(tbl, dict) else str(getattr(tbl, "declared_total", "")),
                "computed_total": str(tbl.get("computed_total", "")) if isinstance(tbl, dict) else str(getattr(tbl, "computed_total", "")),
                "unit": tbl.get("unit_name", "") if isinstance(tbl, dict) else getattr(tbl, "unit_name", ""),
            }

            tbl_dict = tbl if isinstance(tbl, dict) else tbl.model_dump() if hasattr(tbl, "model_dump") else {}

            full_md = _format_table_markdown(headers, str_rows)

            # If full table fits in chunk_size, create single atomic table chunk
            if len(full_md) <= self.chunk_size * 1.5:
                norm_table_md = DevanagariNormalizer.normalize_text(full_md)
                page_chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc.document_id}:p{page.page_number}:t{tbl_idx}:c0",
                        document_id=doc.document_id,
                        file_path=doc.filepath,
                        go_number=doc.go_number,
                        issuing_department=doc.issuing_department,
                        date=doc.date,
                        page_number=page.page_number,
                        chunk_index=chunk_idx,
                        exact_text_excerpt=norm_table_md,
                        char_length=len(norm_table_md),
                        bounding_box_coordinates=page.bounding_boxes[0] if page.bounding_boxes else None,
                        table_json=tbl_dict,
                        math_verification_status=math_meta,
                        font_encoding_type=font_type,
                    )
                )
                chunk_idx += 1
            else:
                # Windowed Table Chunking: slice rows while injecting schema header into every chunk
                rows_per_window = max(2, int(len(str_rows) / max(int(len(full_md) / self.chunk_size), 1)))
                for w_idx in range(0, len(str_rows), rows_per_window):
                    window_rows = str_rows[w_idx:w_idx + rows_per_window]
                    window_md = _format_table_markdown(headers, window_rows)
                    slice_tag = f" [Table Slice Rows {w_idx+1}-{w_idx+len(window_rows)} of {len(str_rows)}]"
                    norm_slice_md = DevanagariNormalizer.normalize_text(window_md) + slice_tag

                    page_chunks.append(
                        DocumentChunk(
                            chunk_id=f"{doc.document_id}:p{page.page_number}:t{tbl_idx}:c{w_idx}",
                            document_id=doc.document_id,
                            file_path=doc.filepath,
                            go_number=doc.go_number,
                            issuing_department=doc.issuing_department,
                            date=doc.date,
                            page_number=page.page_number,
                            chunk_index=chunk_idx,
                            exact_text_excerpt=norm_slice_md,
                            char_length=len(norm_slice_md),
                            bounding_box_coordinates=page.bounding_boxes[0] if page.bounding_boxes else None,
                            table_json=tbl_dict,
                            math_verification_status=math_meta,
                            font_encoding_type=font_type,
                        )
                    )
                    chunk_idx += 1

        # 2. Process Narrative Paragraph Text
        text = page.text.strip()
        if page_tables:
            # Strip lines that belong to markdown tables to prevent duplicate unformatted chunking
            non_table_lines = [
                line
                for line in text.splitlines()
                if not (line.strip().startswith("|") and line.strip().endswith("|"))
            ]
            text = "\n".join(non_table_lines).strip()

        if not text and not page_chunks:
            return []

        if not text:
            return page_chunks

        # If page text fits in chunk_size and no tables
        if len(text) <= self.chunk_size and not page_chunks:
            bbox = assign_chunk_bbox(text, page.page_number, 0, 1)
            norm_text = DevanagariNormalizer.normalize_text(text)
            return [
                DocumentChunk(
                    chunk_id=f"{doc.document_id}:p{page.page_number}:c0",
                    document_id=doc.document_id,
                    file_path=doc.filepath,
                    go_number=doc.go_number,
                    issuing_department=doc.issuing_department,
                    date=doc.date,
                    page_number=page.page_number,
                    chunk_index=0,
                    exact_text_excerpt=norm_text,
                    char_length=len(norm_text),
                    bounding_box_coordinates=bbox,
                    table_json=None,
                    math_verification_status=None,
                    font_encoding_type=font_type,
                )
            ]

        # 2. Process Narrative Clause and Paragraph Text with strict boundary preservation
        clause_units = _split_into_clauses_and_paragraphs(text)
        current_chunk_text = ""

        for unit in clause_units:
            if current_chunk_text and (len(current_chunk_text) + len(unit) > self.chunk_size):
                bbox = assign_chunk_bbox(current_chunk_text, page.page_number, chunk_idx, max(len(clause_units), 1))
                norm_chunk = DevanagariNormalizer.normalize_text(current_chunk_text)
                page_chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc.document_id}:p{page.page_number}:c{chunk_idx}",
                        document_id=doc.document_id,
                        file_path=doc.filepath,
                        go_number=doc.go_number,
                        issuing_department=doc.issuing_department,
                        date=doc.date,
                        page_number=page.page_number,
                        chunk_index=chunk_idx,
                        exact_text_excerpt=norm_chunk,
                        char_length=len(norm_chunk),
                        bounding_box_coordinates=bbox,
                        table_json=None,
                        math_verification_status=None,
                        font_encoding_type=font_type,
                    )
                )
                chunk_idx += 1
                current_chunk_text = unit
            else:
                current_chunk_text = f"{current_chunk_text}\n\n{unit}".strip() if current_chunk_text else unit

        if current_chunk_text:
            bbox = assign_chunk_bbox(current_chunk_text, page.page_number, chunk_idx, max(len(clause_units), 1))
            norm_chunk = DevanagariNormalizer.normalize_text(current_chunk_text)
            page_chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc.document_id}:p{page.page_number}:c{chunk_idx}",
                    document_id=doc.document_id,
                    file_path=doc.filepath,
                    go_number=doc.go_number,
                    issuing_department=doc.issuing_department,
                    date=doc.date,
                    page_number=page.page_number,
                    chunk_index=chunk_idx,
                    exact_text_excerpt=norm_chunk,
                    char_length=len(norm_chunk),
                    bounding_box_coordinates=bbox,
                    table_json=None,
                    math_verification_status=None,
                    font_encoding_type=font_type,
                )
            )

        return page_chunks
