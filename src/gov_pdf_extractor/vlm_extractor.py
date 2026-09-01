"""High-Resolution Multimodal Gemini 3.5 Flash Document and Table Extractor for PramanAI.

Provides vision-based layout parsing, table extraction, and header metadata recognition
from complex and scanned Government Order PDFs using Google Gemini 3.5 Flash.
"""

import asyncio
import json
import logging
import os
import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.gov_pdf_extractor.models import BoundingBox, TableCell, TableData
from src.gov_pdf_extractor.normalizer import DevanagariNormalizer
from src.utils.model_runtime import (
    VLM_MAX_IMAGE_DIM,
    VLM_MODEL,
    VLM_TIMEOUT_SECONDS,
    ainvoke_vision,
)

logger = logging.getLogger("gov_pdf_extractor.vlm")

VLM_SYSTEM_PROMPT = (
    "You are an administrative document OCR expert. You must extract ALL sections, names, "
    "tables, and paragraphs from the top header to the bottom footer without summarizing, "
    "omitting, or truncating any content. Maintain strict verbatim accuracy in Devanagari Hindi."
)

VLM_EXTRACTION_PROMPT = """Extract and transcribe all text, tables, and metadata from this Uttarakhand Government Order page with 100% verbatim fidelity in Devanagari Hindi.

You MUST respond with a valid JSON object in the following format:
```json
{
  "go_number": "<Complete GO number if visible, including pen-written/handwritten digits, or null>",
  "is_draft_placeholder": <true if the GO number is a blank/draft placeholder like संख्या-  / or contains blanks/underscores, otherwise false>,
  "date_raw": "<Date string exactly as written on the page, e.g. 24 दिसम्बर, 2020 or null>",
  "date_iso": "<YYYY-MM-DD format if resolvable, or null>",
  "issuing_department": "<Administrative department or section, e.g. कार्मिक अनुभाग-4, सामान्य प्रशासन विभाग>",
  "page_text": "<Exhaustive verbatim Devanagari Hindi text of the ENTIRE page from top header to bottom footer, including all clauses, names, designations, and tables formatted as GFM markdown tables>",
  "tables": [
    {
      "headers": ["header1", "header2"],
      "rows": [["cell1", "cell2"]]
    }
  ]
}
```

RULES:
1. Transcribe ALL administrative clauses, paragraph text, order numbers, dates, department names, designations, person names, and categorized sections (such as सामूहिक श्रेणी 1, सामूहिक श्रेणी 2, eligibility terms, rules, and notes) completely and verbatim from top to bottom.
2. Output all tables as standard GitHub-Flavored Markdown (GFM) tables in `page_text`.
3. Do NOT summarize, omit, truncate, or skip any sections or clauses.
4. Do NOT generate ASCII noise, English gibberish, conversational commentary, or preambles.
"""


class VlmPageExtractionResult(BaseModel):
    """Structured extraction result from VLM processing of a single page."""

    model_config = ConfigDict(strict=True)

    raw_markdown: str = Field(..., description="Full verbatim response from VLM")
    cleaned_text: str = Field(..., description="Cleaned body text with headers and markdown tables")
    tables: List[TableData] = Field(default_factory=list, description="Extracted structured TableData instances")
    header_metadata: Dict[str, Any] = Field(default_factory=dict, description="Extracted header metadata dict")
    bounding_boxes: List[List[float]] = Field(
        default_factory=list, description="Normalized [x, y, w, h] representative bounding boxes"
    )
    is_draft_placeholder: bool = Field(default=False, description="Whether GO number on page is an unassigned draft placeholder")



class VlmDocumentExtractor:
    """Extracts text, structural GFM tables, and header metadata from page images using Google Gemini 3.5 Flash."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_image_dim: Optional[int] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name or VLM_MODEL
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else VLM_TIMEOUT_SECONDS
        )
        self.max_image_dim = (
            max_image_dim if max_image_dim is not None else VLM_MAX_IMAGE_DIM
        )

    def encode_image_bytes(self, bgr_image: np.ndarray) -> bytes:
        """Encodes BGR numpy image to RGB, resizes longest edge to max_image_dim, and returns JPEG bytes (quality=85)."""
        h, w = bgr_image.shape[:2]
        target_img = bgr_image

        if max(h, w) > self.max_image_dim:
            scale = self.max_image_dim / max(h, w)
            new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
            target_img = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB for clean Vision ViT patch encoding
        rgb_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        import io
        from PIL import Image

        pil_img = Image.fromarray(rgb_img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    @staticmethod
    def strip_conversational_chatter(raw_text: str) -> str:
        """Strips conversational preambles, greetings, trailing commentary, and repetitive padding tokens."""
        text = raw_text.strip()
        # Remove repetitive @ padding tokens if generated by VLM
        text = re.sub(r"@+", "", text).strip()
        # Remove common preamble patterns
        text = re.sub(
            r"^(?:Here is the (?:extracted |markdown |verbatim )?text[^\n]*\n+|Sure,?[^\n]*\n+|Certainly,?[^\n]*\n+)",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        # Remove trailing commentary patterns
        text = re.sub(
            r"\n+(?:Let me know if you need anything else|I hope this helps|Note: The above text is extracted)[^\n]*$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        # Strip outer markdown code blocks if the whole response is wrapped in ```markdown ... ``` or ``` ... ```
        fence_match = re.match(r"^```(?:markdown|text)?\s*\n(.*)\n```$", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        return text

    @staticmethod
    def extract_header_json(raw_text: Any) -> Tuple[Dict[str, Any], str, bool]:
        """Extracts JSON metadata object if present, unpacks page_text, and detects draft placeholders."""
        if isinstance(raw_text, list):
            raw_text = "\n".join(str(x) for x in raw_text)
        elif not isinstance(raw_text, str):
            raw_text = str(raw_text) if raw_text is not None else ""

        # Check if raw_text starts with python list of dict representation "[{'type': 'text', 'text': ...}]"
        if raw_text.strip().startswith("[{'type': 'text'") or raw_text.strip().startswith('[{"type": "text"'):
            try:
                import ast
                evaluated = ast.literal_eval(raw_text.strip())
                if isinstance(evaluated, list) and evaluated and isinstance(evaluated[0], dict) and "text" in evaluated[0]:
                    raw_text = evaluated[0]["text"]
            except Exception:
                pass

        header_meta: Dict[str, Any] = {}
        remaining_text = raw_text
        is_draft_placeholder = False

        # Attempt 1: Check for fenced ```json ... ``` block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        json_str = ""
        if json_match:
            json_str = json_match.group(1).strip()
            remaining_text = raw_text[: json_match.start()] + raw_text[json_match.end() :]
            remaining_text = remaining_text.strip()
        elif raw_text.strip().startswith("{") and raw_text.strip().endswith("}"):
            # Attempt 2: Whole response is a raw JSON object
            json_str = raw_text.strip()
            remaining_text = ""

        if json_str:
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    header_meta = parsed
                    # If page_text is in JSON payload, use it as remaining_text
                    if "page_text" in header_meta and header_meta["page_text"]:
                        remaining_text = str(header_meta["page_text"]).strip()
            except Exception as e:
                logger.debug("Could not parse VLM JSON payload: %s; preserving raw text", e)

        # Detect draft placeholder status
        if "is_draft_placeholder" in header_meta and isinstance(header_meta["is_draft_placeholder"], bool):
            is_draft_placeholder = header_meta["is_draft_placeholder"]
        else:
            go_num = str(header_meta.get("go_number") or "")
            if not go_num or re.search(r"संख्या\s*[\—\-]?\s*[\n\r]*\s*\/|संख्या\s*[\—\-]?\s*[\s_]{2,}\/|[\s_]{2,}\/|संख्या\s*[\—\-]?\s*$", go_num):
                is_draft_placeholder = True

        if not remaining_text.strip() and header_meta:
            lines = [f"{k}: {v}" for k, v in header_meta.items() if v and k != "page_text"]
            remaining_text = "\n".join(lines)

        if not remaining_text.strip() and raw_text.strip():
            remaining_text = raw_text.strip()

        return header_meta, remaining_text, is_draft_placeholder

    @staticmethod
    def parse_gfm_tables(markdown_text: str) -> List[TableData]:
        """Parses all GitHub-Flavored Markdown tables from text into structured TableData models."""
        tables: List[TableData] = []
        lines = markdown_text.splitlines()
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            # Table row start detection
            if line.startswith("|") and line.endswith("|") and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Check for standard markdown table delimiter | --- | :--- | --- |
                if re.match(r"^\|(?:\s*:?-+:?\s*\|)+$", next_line):
                    raw_headers = [c.strip() for c in line.strip("|").split("|")]
                    headers = [DevanagariNormalizer.normalize_text(h) for h in raw_headers]
                    table_rows: List[List[str]] = []
                    j = i + 2

                    while j < len(lines):
                        row_line = lines[j].strip()
                        if row_line.startswith("|") and row_line.endswith("|"):
                            row_cells = [c.strip() for c in row_line.strip("|").split("|")]
                            # Align column length with header
                            if len(row_cells) < len(headers):
                                row_cells.extend([""] * (len(headers) - len(row_cells)))
                            table_rows.append(row_cells[: len(headers)])
                            j += 1
                        else:
                            break

                    if table_rows:
                        total_lines = max(len(lines), 1)
                        t_ymin = max(0.05, min(0.90, i / total_lines))
                        t_ymax = max(t_ymin + 0.05, min(0.98, j / total_lines))

                        cell_matrix: List[List[TableCell]] = []
                        for r_idx, row in enumerate(table_rows):
                            row_cells_objs: List[TableCell] = []
                            r_ymin = t_ymin + (t_ymax - t_ymin) * (r_idx / max(len(table_rows), 1))
                            r_ymax = t_ymin + (t_ymax - t_ymin) * ((r_idx + 1) / max(len(table_rows), 1))

                            for c_idx, val in enumerate(row):
                                c_xmin = 0.05 + 0.90 * (c_idx / max(len(headers), 1))
                                c_xmax = 0.05 + 0.90 * ((c_idx + 1) / max(len(headers), 1))
                                cell_bbox = BoundingBox(
                                    ymin=round(r_ymin, 4),
                                    xmin=round(c_xmin, 4),
                                    ymax=round(r_ymax, 4),
                                    xmax=round(c_xmax, 4),
                                    page_number=1,
                                )
                                norm_val = DevanagariNormalizer.normalize_text(val)
                                row_cells_objs.append(
                                    TableCell(
                                        row_idx=r_idx,
                                        col_idx=c_idx,
                                        raw_text=val,
                                        normalized_text=norm_val,
                                        bbox=cell_bbox,
                                    )
                                )
                            cell_matrix.append(row_cells_objs)

                        tbl_bbox = BoundingBox(
                            ymin=round(t_ymin, 4),
                            xmin=0.05,
                            ymax=round(t_ymax, 4),
                            xmax=0.95,
                            page_number=1,
                        )
                        tables.append(
                            TableData(
                                headers=headers,
                                rows=cell_matrix,
                                bbox=tbl_bbox,
                            )
                        )
                    i = j
                    continue
            i += 1

        return tables

    async def a_extract_page(
        self,
        bgr_image: np.ndarray,
        page_num: int = 1,
    ) -> VlmPageExtractionResult:
        """Asynchronously parses a preprocessed page image using Qwen2.5-VL with automatic Devanagari normalization."""
        img_bytes = self.encode_image_bytes(bgr_image)

        # Call multimodal VLM (Google Gemini 3.5 Flash) with 0.0 temperature
        raw_response = await ainvoke_vision(
            image_bytes=img_bytes,
            prompt=VLM_EXTRACTION_PROMPT,
            system_prompt=VLM_SYSTEM_PROMPT,
            model=self.model_name,
            timeout_seconds=self.timeout_seconds,
            max_image_dim=self.max_image_dim,
        )

        cleaned_response = self.strip_conversational_chatter(raw_response)
        header_meta, cleaned_text, is_draft = self.extract_header_json(cleaned_response)

        # Automatically normalize Devanagari text to eliminate decomposed Unicode and misplaced matras
        cleaned_text = DevanagariNormalizer.normalize_text(cleaned_text)
        tables = self.parse_gfm_tables(cleaned_text)

        # Handle genuinely blank pages or scanner watermark noise (< 30 meaningful characters)
        meaningful_chars = re.sub(r"[\s\-_=@~|`#*]", "", cleaned_text)
        if len(meaningful_chars) < 30:
            logger.info(
                "Page %d has only %d meaningful characters (blank scan or watermark noise); recording clean empty page",
                page_num,
                len(meaningful_chars),
            )
            cleaned_text = ""
            tables = []

        # Extract authentic bounding boxes from tables if detected
        bboxes: List[List[float]] = []
        for tbl in tables:
            if tbl.bbox is not None:
                bboxes.append([
                    round(tbl.bbox.xmin, 4),
                    round(tbl.bbox.ymin, 4),
                    round(tbl.bbox.xmax - tbl.bbox.xmin, 4),
                    round(tbl.bbox.ymax - tbl.bbox.ymin, 4),
                ])

        return VlmPageExtractionResult(
            raw_markdown=raw_response,
            cleaned_text=cleaned_text,
            tables=tables,
            header_metadata=header_meta,
            bounding_boxes=bboxes,
            is_draft_placeholder=is_draft,
        )

    def extract_page(
        self,
        bgr_image: np.ndarray,
        page_num: int = 1,
    ) -> VlmPageExtractionResult:
        """Synchronous wrapper for a_extract_page."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio

                nest_asyncio.apply()
                return loop.run_until_complete(self.a_extract_page(bgr_image, page_num=page_num))
            return asyncio.run(self.a_extract_page(bgr_image, page_num=page_num))
        except Exception:
            return asyncio.run(self.a_extract_page(bgr_image, page_num=page_num))
