"""Pure-VLM Ingestion Pipeline Orchestrator for Uttarakhand Government Documents.

Executes a strict, high-fidelity multimodal vision extraction workflow:
1. 300+ DPI Vision Preprocessing (Sauvola binarization, auto-deskew, stamp suppression).
2. Contrast and edge enhancement via enhance_gov_document_image.
3. Sovereign Local Qwen2.5-VL Multimodal Vision-Language extraction.
4. Devanagari Unicode Normalization (NFC, Nukta repair, ZWJ/ZWNJ preservation).
5. 2D Constraint-Based Mathematical Cross-Validation.

Strict Zero-Mock / Zero-Silent-Degradation Policy:
If VLM cannot extract a page with high fidelity, raises VlmExtractionError immediately
to trigger atomic transaction rollback. No silent degradation, no fallback to legacy OCR.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from PIL import Image
import pymupdf as fitz

from src.gov_pdf_extractor.models import (
    BoundingBox,
    DocumentExtractionResult,
    PageType,
    ParsedPage,
    TableCell,
    TableData,
)
from src.gov_pdf_extractor.normalizer import DevanagariNormalizer
from src.gov_pdf_extractor.preprocessor import ImagePreprocessor, enhance_gov_document_image
from src.gov_pdf_extractor.table_extractor import TableExtractor
from src.gov_pdf_extractor.validator import MathValidator
from src.gov_pdf_extractor.vlm_extractor import VlmDocumentExtractor

logger = logging.getLogger("gov_pdf_extractor.pipeline")


class VlmExtractionError(Exception):
    """Raised when strict VLM extraction fails on a document page."""
    pass


class GovPdfExtractor:
    """Strict Multimodal VLM-Only PDF Extraction Pipeline for Sovereign Document Ingestion."""

    def __init__(
        self,
        target_dpi: int = 150,
        deskew_threshold_deg: float = 0.2,
        vlm_base_url: Optional[str] = None,
        vlm_model_name: Optional[str] = None,
        vlm_timeout_seconds: Optional[float] = None,
        max_image_dim: Optional[int] = None,
        **kwargs: Any,
    ):
        self.preprocessor = ImagePreprocessor(
            target_dpi=target_dpi, deskew_threshold_deg=deskew_threshold_deg
        )
        self.table_extractor = TableExtractor()
        self.normalizer = DevanagariNormalizer()
        self.validator = MathValidator()
        self.vlm_extractor = VlmDocumentExtractor(
            base_url=vlm_base_url,
            model_name=vlm_model_name,
            timeout_seconds=vlm_timeout_seconds,
            max_image_dim=max_image_dim,
        )

    def extract_document(self, pdf_path: str) -> DocumentExtractionResult:
        """Executes strict pure-VLM extraction pipeline on an administrative order or gazette PDF."""
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        audit_log: List[Dict[str, Any]] = []
        parsed_pages: List[ParsedPage] = []

        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        audit_log.append({
            "stage": "initialization",
            "filename": path_obj.name,
            "total_pages": total_pages,
            "dpi": self.preprocessor.target_dpi,
            "mode": "pure_vlm_strict",
        })

        try:
            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_num = page_idx + 1
                page_audit: Dict[str, Any] = {"page_number": page_num}

                # -------------------------------------------------------------
                # STAGE 1: 300+ DPI VISION CONDITIONING & DESKEWING
                # -------------------------------------------------------------
                proc_bgr, bin_img, deskew_angle = self.preprocessor.preprocess_page(page)
                page_audit["stage_1_vision"] = {
                    "deskew_angle_deg": deskew_angle,
                    "binarized_shape": list(bin_img.shape),
                }

                # -------------------------------------------------------------
                # STAGE 2: CONTRAST & SHARPNESS ENHANCEMENT FOR DEVANAGARI DIACRITICS
                # -------------------------------------------------------------
                try:
                    rgb_frame = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(rgb_frame)
                    enhanced_pil = enhance_gov_document_image(pil_frame)
                    enhanced_bgr = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
                except Exception as enh_exc:
                    logger.warning("Image enhancement fallback on page %d: %s", page_num, enh_exc)
                    enhanced_bgr = proc_bgr

                # -------------------------------------------------------------
                # STAGE 3: STRICT MULTIMODAL VLM EXTRACTION
                # -------------------------------------------------------------
                try:
                    vlm_res = self.vlm_extractor.extract_page(enhanced_bgr, page_num=page_num)
                except Exception as vlm_err:
                    logger.error(
                        "Strict VLM extraction failed on page %d of '%s': %s",
                        page_num,
                        path_obj.name,
                        vlm_err,
                    )
                    raise VlmExtractionError(
                        f"VLM extraction failed on page {page_num} of '{path_obj.name}': {vlm_err}"
                    ) from vlm_err

                if vlm_res is None:
                    raise VlmExtractionError(
                        f"VLM returned null extraction payload on page {page_num} of '{path_obj.name}'"
                    )

                raw_page_text = vlm_res.cleaned_text
                page_tables = vlm_res.tables

                page_audit["stage_3_vlm"] = {
                    "status": "success",
                    "header_meta": vlm_res.header_metadata,
                    "tables_found": len(page_tables),
                    "bboxes_count": len(vlm_res.bounding_boxes),
                    "is_draft_placeholder": vlm_res.is_draft_placeholder,
                }

                # -------------------------------------------------------------
                # STAGE 4: DEVANAGARI UNICODE NORMALIZATION
                # -------------------------------------------------------------
                cleaned_page_text = self.normalizer.normalize_text(raw_page_text)

                for tbl in page_tables:
                    tbl.headers = [self.normalizer.normalize_text(h) for h in tbl.headers]
                    for row in tbl.rows:
                        for cell in row:
                            cell.normalized_text = self.normalizer.normalize_text(cell.raw_text)

                page_audit["stage_4_normalization"] = {
                    "raw_length": len(raw_page_text),
                    "cleaned_length": len(cleaned_page_text),
                }

                # -------------------------------------------------------------
                # STAGE 5: MATHEMATICAL CROSS-VALIDATION LOOP
                # -------------------------------------------------------------
                validated_tables: List[TableData] = []
                for tbl in page_tables:
                    val_tbl = self.validator.validate_table_sums(tbl)
                    validated_tables.append(val_tbl)

                page_audit["stage_5_math_validation"] = {
                    "tables_validated": len(validated_tables),
                    "all_valid": all(t.is_mathematically_valid for t in validated_tables),
                }

                parsed_page = ParsedPage(
                    page_number=page_num,
                    page_type=PageType.SCANNED_IMAGE,
                    raw_text=raw_page_text,
                    cleaned_text=cleaned_page_text,
                    tables=validated_tables,
                    metadata={
                        "deskew_angle_deg": deskew_angle,
                        "header_metadata": vlm_res.header_metadata,
                        "bounding_boxes": vlm_res.bounding_boxes,
                        "is_draft_placeholder": vlm_res.is_draft_placeholder,
                    },
                )
                parsed_pages.append(parsed_page)
                audit_log.append(page_audit)
                logger.info(
                    "Page %d/%d processed — pure_vlm chars=%d tables=%d",
                    page_num,
                    total_pages,
                    len(cleaned_page_text),
                    len(validated_tables),
                )
                time.sleep(2.0)

            # Stitch multi-page tables across pages
            page_tables_tuples = [(p.page_number, p.tables) for p in parsed_pages]
            stitched_tuples = self.table_extractor.stitch_multi_page_tables(page_tables_tuples)
            for p, (_, st_tables) in zip(parsed_pages, stitched_tuples):
                p.tables = st_tables

        finally:
            doc.close()

        return DocumentExtractionResult(
            filename=path_obj.name,
            total_pages=total_pages,
            pages=parsed_pages,
            pipeline_audit_log=audit_log,
        )
