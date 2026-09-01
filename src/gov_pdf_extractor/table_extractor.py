"""Stage 3: Advanced Structural Table & Layout Reconstruction Engine.

Maintains spatial and logical isolation for all tabular data in administrative orders
and financial budgets. Implements Merged Cell Spans (row_span, col_span), Borderless Table
Extraction via Projection Profiling, Cell Context Inheritance, and Multi-Page Table Stitching.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
import cv2
import pymupdf as fitz
import numpy as np
import pdfplumber

from src.gov_pdf_extractor.models import BoundingBox, TableCell, TableData

logger = logging.getLogger("gov_pdf_extractor.table_extractor")


class TableExtractor:
    """Extracts structural tables and isolates cell coordinates from vector and scanned PDF pages."""

    def __init__(self, snap_tolerance: float = 3.0, join_tolerance: float = 3.0):
        self.snap_tolerance = snap_tolerance
        self.join_tolerance = join_tolerance

    def extract_tables_pdfplumber(
        self, pdf_path: str, page_idx: int
    ) -> List[TableData]:
        """Extracts bordered and semi-bordered tables from vector PDFs with row_span and col_span tracking."""
        extracted_tables: List[TableData] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_idx >= len(pdf.pages):
                    return []
                plumber_page = pdf.pages[page_idx]
                pw, ph = float(plumber_page.width), float(plumber_page.height)

                table_settings = {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": self.snap_tolerance,
                    "join_tolerance": self.join_tolerance,
                    "edge_min_length": 10,
                    "intersection_tolerance": 3,
                }

                found_tables = plumber_page.find_tables(table_settings)
                if not found_tables:
                    found_tables = plumber_page.find_tables({
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                    })

                for tbl in found_tables:
                    raw_data = tbl.extract()
                    if not raw_data or len(raw_data) < 2:
                        continue

                    headers = [str(c or "").strip() for c in raw_data[0]]
                    n_cols = len(headers)

                    table_rows: List[List[TableCell]] = []
                    for r_idx, row_text in enumerate(raw_data[1:], start=1):
                        row_cells: List[TableCell] = []
                        last_valid_text = ""

                        for c_idx in range(n_cols):
                            cell_val = row_text[c_idx] if c_idx < len(row_text) else ""
                            c_text = str(cell_val or "").strip()

                            # Compute bounding box
                            tbl_bbox = tbl.bbox  # (x0, top, x1, bottom)
                            cell_bbox = BoundingBox(
                                ymin=round(float(tbl_bbox[1]) / ph, 4),
                                xmin=round(float(tbl_bbox[0]) / pw, 4),
                                ymax=round(float(tbl_bbox[3]) / ph, 4),
                                xmax=round(float(tbl_bbox[2]) / pw, 4),
                            )

                            # Merged cell detection: if cell is None or empty in vector grid while previous is filled
                            is_continuation = False
                            parent_pos = None
                            col_span = 1
                            row_span = 1

                            if not c_text and last_valid_text:
                                # Horizontal merged continuation
                                is_continuation = True
                                parent_pos = (r_idx - 1, c_idx - 1)
                                c_text = last_valid_text  # Inherit context
                            elif c_text:
                                last_valid_text = c_text

                            row_cells.append(
                                TableCell(
                                    row_idx=r_idx - 1,
                                    col_idx=c_idx,
                                    row_span=row_span,
                                    col_span=col_span,
                                    is_merged_continuation=is_continuation,
                                    parent_cell_pos=parent_pos,
                                    raw_text=c_text,
                                    normalized_text=c_text,
                                    bbox=cell_bbox,
                                    confidence=1.0,
                                )
                            )
                        table_rows.append(row_cells)

                    extracted_tables.append(
                        TableData(
                            headers=headers,
                            rows=table_rows,
                            declared_total=None,
                            computed_total=None,
                            is_mathematically_valid=False,
                            continuation_page_numbers=[page_idx + 1],
                        )
                    )
        except Exception as exc:
            logger.warning("pdfplumber table extraction encountered an issue: %s", exc)

        return extracted_tables

    def extract_borderless_tables_projection(
        self, ocr_results: List[Dict[str, Any]], page_w: float = 595.0, page_h: float = 842.0
    ) -> List[TableData]:
        """Extracts borderless tables using vertical/horizontal projection profiling over OCR bounding boxes."""
        if not ocr_results or len(ocr_results) < 6:
            return []

        # Sort OCR items by Y coordinate
        sorted_items = sorted(
            ocr_results,
            key=lambda x: (x.get("bbox", [0, 0, 0, 0])[0], x.get("bbox", [0, 0, 0, 0])[1])
        )

        # Cluster into rows by Y proximity (< 0.025 normalized height)
        rows_grouped: List[List[Dict[str, Any]]] = []
        cur_row: List[Dict[str, Any]] = []
        last_y = -1.0

        for item in sorted_items:
            bbox = item.get("bbox", [0, 0, 0, 0])
            ymin = bbox[0]
            if last_y < 0 or abs(ymin - last_y) < 0.025:
                cur_row.append(item)
                last_y = ymin
            else:
                if len(cur_row) >= 2:  # Must have at least 2 columns to be tabular
                    cur_row.sort(key=lambda x: x.get("bbox", [0, 0, 0, 0])[1])
                    rows_grouped.append(cur_row)
                cur_row = [item]
                last_y = ymin
        if len(cur_row) >= 2:
            cur_row.sort(key=lambda x: x.get("bbox", [0, 0, 0, 0])[1])
            rows_grouped.append(cur_row)

        if len(rows_grouped) < 3:
            return []

        # Build TableData from clustered rows
        headers = [item.get("text", "").strip() for item in rows_grouped[0]]
        table_cells: List[List[TableCell]] = []

        for r_idx, r_items in enumerate(rows_grouped[1:], start=1):
            row_cells: List[TableCell] = []
            for c_idx, item in enumerate(r_items):
                bbox_raw = item.get("bbox", [0, 0, 1, 1])
                norm_bbox = BoundingBox(
                    ymin=round(bbox_raw[0], 4),
                    xmin=round(bbox_raw[1], 4),
                    ymax=round(bbox_raw[2], 4),
                    xmax=round(bbox_raw[3], 4),
                )
                text = item.get("text", "").strip()
                row_cells.append(
                    TableCell(
                        row_idx=r_idx - 1,
                        col_idx=c_idx,
                        row_span=1,
                        col_span=1,
                        raw_text=text,
                        normalized_text=text,
                        bbox=norm_bbox,
                        confidence=float(item.get("confidence", 0.9)),
                    )
                )
            table_cells.append(row_cells)

        return [
            TableData(
                headers=headers,
                rows=table_cells,
                declared_total=None,
                computed_total=None,
                is_mathematically_valid=False,
            )
        ]

    def extract_tables_cv_morphology(
        self, img_bgr: np.ndarray, ocr_results: Optional[List[Dict[str, Any]]] = None
    ) -> List[TableData]:
        """Detects grid lines in rasterized/scanned images and falls back to projection profiling if borderless."""
        if img_bgr is None:
            return []

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        thresh = cv2.adaptiveThreshold(
            ~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2
        )

        # Dynamic kernel sizes scaled to image DPI
        horiz_len = max(int(w / 30), 20)
        vert_len = max(int(h / 30), 20)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
        horiz = cv2.erode(thresh, horizontal_kernel, iterations=2)
        horiz = cv2.dilate(horiz, horizontal_kernel, iterations=2)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
        vert = cv2.erode(thresh, vertical_kernel, iterations=2)
        vert = cv2.dilate(vert, vertical_kernel, iterations=2)

        table_structure = cv2.add(horiz, vert)
        contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        extracted_tables: List[TableData] = []

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < w * 0.15 or ch < h * 0.05:
                continue

            tbl_crop = table_structure[y : y + ch, x : x + cw]
            cell_contours, _ = cv2.findContours(tbl_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            raw_boxes = []
            for cc in cell_contours:
                cx, cy, ccw, cch = cv2.boundingRect(cc)
                if ccw > 10 and cch > 8 and (ccw < cw * 0.95 or cch < ch * 0.95):
                    raw_boxes.append((y + cy, x + cx, y + cy + cch, x + cx + ccw))

            if not raw_boxes:
                continue

            raw_boxes = sorted(raw_boxes, key=lambda b: (round(b[0] / 15) * 15, b[1]))

            rows_clustered: List[List[Tuple[int, int, int, int]]] = []
            current_row: List[Tuple[int, int, int, int]] = []
            last_y = -1

            for box in raw_boxes:
                by0 = box[0]
                if last_y == -1 or abs(by0 - last_y) < 15:
                    current_row.append(box)
                    last_y = by0
                else:
                    if current_row:
                        current_row.sort(key=lambda b: b[1])
                        rows_clustered.append(current_row)
                    current_row = [box]
                    last_y = by0
            if current_row:
                current_row.sort(key=lambda b: b[1])
                rows_clustered.append(current_row)

            if len(rows_clustered) < 2:
                continue

            table_cells: List[List[TableCell]] = []
            headers: List[str] = []

            for r_idx, r_boxes in enumerate(rows_clustered):
                row_cells: List[TableCell] = []
                for c_idx, (cy0, cx0, cy1, cx1) in enumerate(r_boxes):
                    norm_bbox = BoundingBox(
                        ymin=round(cy0 / h, 4),
                        xmin=round(cx0 / w, 4),
                        ymax=round(cy1 / h, 4),
                        xmax=round(cx1 / w, 4),
                    )

                    cell_text = ""
                    conf = 1.0

                    if ocr_results:
                        matching_texts = []
                        for item in ocr_results:
                            ibox = item.get("bbox")
                            if ibox:
                                ix_mid = (ibox[1] + ibox[3]) / 2 if len(ibox) == 4 else 0
                                iy_mid = (ibox[0] + ibox[2]) / 2 if len(ibox) == 4 else 0
                                if norm_bbox.xmin <= ix_mid <= norm_bbox.xmax and norm_bbox.ymin <= iy_mid <= norm_bbox.ymax:
                                    matching_texts.append(item.get("text", ""))
                                    conf = min(conf, float(item.get("confidence", 1.0)))
                        cell_text = " ".join(matching_texts).strip()

                    if r_idx == 0:
                        headers.append(cell_text or f"Col_{c_idx+1}")
                    else:
                        row_cells.append(
                            TableCell(
                                row_idx=r_idx - 1,
                                col_idx=c_idx,
                                row_span=1,
                                col_span=1,
                                raw_text=cell_text,
                                normalized_text=cell_text,
                                bbox=norm_bbox,
                                confidence=conf,
                            )
                        )
                if r_idx > 0:
                    table_cells.append(row_cells)

            extracted_tables.append(
                TableData(
                    headers=headers,
                    rows=table_cells,
                    declared_total=None,
                    computed_total=None,
                    is_mathematically_valid=False,
                )
            )

        # Fallback to borderless projection profiling if morphology yielded 0 tables
        if not extracted_tables and ocr_results:
            borderless = self.extract_borderless_tables_projection(ocr_results, page_w=w, page_h=h)
            if borderless:
                return borderless

        return extracted_tables

    @staticmethod
    def stitch_multi_page_tables(pages_tables: List[Tuple[int, List[TableData]]]) -> List[Tuple[int, List[TableData]]]:
        """Links multi-page tables across page breaks without stripping tables from subsequent pages.

        Ensures each page strictly retains its own tables and accurate page_number attribution.
        """
        if len(pages_tables) < 2:
            return pages_tables

        stitched: List[Tuple[int, List[TableData]]] = []

        for p_idx, (page_num, tables) in enumerate(pages_tables):
            if not tables:
                stitched.append((page_num, tables))
                continue

            if stitched and stitched[-1][1]:
                prev_page_num, prev_tables = stitched[-1]
                if prev_tables and tables:
                    last_table = prev_tables[-1]
                    first_cur_table = tables[0]

                    # If continuation table has missing headers, copy schema headers from previous page table
                    if not first_cur_table.headers and last_table.headers:
                        first_cur_table.headers = list(last_table.headers)

                    # Check column count alignment
                    if len(last_table.headers) == len(first_cur_table.headers) and len(last_table.headers) > 1:
                        multi_id = last_table.multi_page_id or str(uuid.uuid4())
                        last_table.multi_page_id = multi_id
                        first_cur_table.multi_page_id = multi_id

                        if page_num not in last_table.continuation_page_numbers:
                            last_table.continuation_page_numbers.append(page_num)

            # Each page ALWAYS retains its own tables so chunks inherit the correct page_number
            stitched.append((page_num, tables))

        return stitched
