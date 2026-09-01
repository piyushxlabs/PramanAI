"""Pydantic V2 Data Models for gov_pdf_extractor pipeline.

Defines strict type-safe models for document pages, structural table grids,
bounding boxes, and mathematical verification results with multi-resolution coordinate scaling.
"""

from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field


class PageType(str, Enum):
    """Classification type of a PDF page."""

    NATIVE_UNICODE = "native_unicode"
    LEGACY_FONT = "legacy_font"
    SCANNED_IMAGE = "scanned_image"


class BoundingBox(BaseModel):
    """Bounding box coordinates with multi-resolution coordinate scaling methods."""

    model_config = ConfigDict(strict=True)

    ymin: float = Field(..., description="Top boundary coordinate")
    xmin: float = Field(..., description="Left boundary coordinate")
    ymax: float = Field(..., description="Bottom boundary coordinate")
    xmax: float = Field(..., description="Right boundary coordinate")

    @property
    def is_normalized(self) -> bool:
        """Returns True if coordinates are in normalized 0.0 to 1.0 space."""
        return self.xmax <= 1.05 and self.ymax <= 1.05 and self.xmin >= -0.05 and self.ymin >= -0.05

    def to_normalized(self, page_width_pt: float, page_height_pt: float) -> "BoundingBox":
        """Normalizes bounding box coordinates into [0.0, 1.0] unit square."""
        if self.is_normalized:
            return self
        pw = max(page_width_pt, 1.0)
        ph = max(page_height_pt, 1.0)
        return BoundingBox(
            ymin=max(0.0, min(1.0, round(self.ymin / ph, 6))),
            xmin=max(0.0, min(1.0, round(self.xmin / pw, 6))),
            ymax=max(0.0, min(1.0, round(self.ymax / ph, 6))),
            xmax=max(0.0, min(1.0, round(self.xmax / pw, 6))),
        )

    def to_points(self, page_width_pt: float, page_height_pt: float) -> "BoundingBox":
        """Converts bounding box to 72 DPI PDF point space."""
        if not self.is_normalized:
            return self
        return BoundingBox(
            ymin=round(self.ymin * page_height_pt, 2),
            xmin=round(self.xmin * page_width_pt, 2),
            ymax=round(self.ymax * page_height_pt, 2),
            xmax=round(self.xmax * page_width_pt, 2),
        )

    def to_raster_pixels(
        self, page_width_pt: float, page_height_pt: float, dpi: int = 300
    ) -> "BoundingBox":
        """Scales bounding box to raster image pixels at target DPI (e.g. 300 DPI)."""
        pts_box = self.to_points(page_width_pt, page_height_pt)
        scale = dpi / 72.0
        return BoundingBox(
            ymin=round(pts_box.ymin * scale, 1),
            xmin=round(pts_box.xmin * scale, 1),
            ymax=round(pts_box.ymax * scale, 1),
            xmax=round(pts_box.xmax * scale, 1),
        )

    def to_crop_rect(
        self,
        page_width_pt: float,
        page_height_pt: float,
        dpi: int = 450,
        padding_pt: float = 5.0,
    ) -> Tuple[float, float, float, float]:
        """Returns clamped (x0, y0, x1, y1) in PDF points for localized 450 DPI re-crop."""
        pts_box = self.to_points(page_width_pt, page_height_pt)
        x0 = max(0.0, pts_box.xmin - padding_pt)
        y0 = max(0.0, pts_box.ymin - padding_pt)
        x1 = min(page_width_pt, pts_box.xmax + padding_pt)
        y1 = min(page_height_pt, pts_box.ymax + padding_pt)
        return (round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2))


class TableCell(BaseModel):
    """Represents a single cell within a structural table grid with span support."""

    model_config = ConfigDict(strict=True)

    row_idx: int = Field(..., ge=0, description="0-indexed row position")
    col_idx: int = Field(..., ge=0, description="0-indexed column position")
    row_span: int = Field(default=1, ge=1, description="Number of rows this cell spans")
    col_span: int = Field(default=1, ge=1, description="Number of columns this cell spans")
    is_merged_continuation: bool = Field(
        default=False, description="True if cell position is occupied by a span from a parent cell"
    )
    parent_cell_pos: Optional[Tuple[int, int]] = Field(
        default=None, description="Coordinates (row_idx, col_idx) of top-left parent cell if merged"
    )
    raw_text: str = Field(..., description="Raw extracted cell text before normalization")
    normalized_text: str = Field(..., description="Cleaned, normalized Devanagari/English text")
    numeric_value: Optional[Decimal] = Field(
        default=None, description="Parsed decimal numeric value if cell contains financial/budget figures"
    )
    bbox: Optional[BoundingBox] = Field(default=None, description="Spatial coordinates of the cell")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="OCR / extraction confidence score"
    )


class TableData(BaseModel):
    """Represents an extracted structural table with mathematical validation and multi-page metadata."""

    model_config = ConfigDict(strict=True)

    headers: List[str] = Field(default_factory=list, description="Column header strings")
    rows: List[List[TableCell]] = Field(
        default_factory=list, description="Matrix of table cells organized by row and column"
    )
    unit_multiplier: Decimal = Field(
        default=Decimal("1"), description="Unit scale parsed from headers (e.g. 100000 for Lakhs, 10000000 for Crores)"
    )
    unit_name: Optional[str] = Field(
        default=None, description="Detected scale unit name (e.g. 'लाख', 'करोड़', 'हज़ार')"
    )
    declared_total: Optional[Decimal] = Field(
        default=None, description="Explicit total extracted from summary row or text"
    )
    computed_total: Optional[Decimal] = Field(
        default=None, description="Sum of individual row numeric values calculated by validator"
    )
    is_mathematically_valid: bool = Field(
        default=False, description="True if declared_total == computed_total or no total required"
    )
    validation_error_delta: Optional[Decimal] = Field(
        default=None, description="Absolute difference between computed sum and declared total"
    )
    multi_page_id: Optional[str] = Field(
        default=None, description="Unique identifier linking pages of a continuous multi-page table"
    )
    continuation_page_numbers: List[int] = Field(
        default_factory=list, description="List of page numbers this table spans across"
    )
    bbox: Optional[BoundingBox] = Field(
        default=None, description="Spatial coordinates of the table"
    )


class ParsedPage(BaseModel):
    """Represents a single parsed document page."""

    model_config = ConfigDict(strict=True)

    page_number: int = Field(..., ge=1, description="1-indexed page number")
    page_type: PageType = Field(..., description="Classification route of the page")
    raw_text: str = Field(..., description="Raw extracted text from text-layer or OCR")
    cleaned_text: str = Field(..., description="Devanagari normalized and dictionary-corrected text")
    tables: List[TableData] = Field(default_factory=list, description="Extracted structural tables")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Page-level audit metadata (DPI, deskew angle, font info)"
    )


class DocumentExtractionResult(BaseModel):
    """Root result returned by GovPdfExtractor pipeline."""

    model_config = ConfigDict(strict=True)

    filename: str = Field(..., description="Source PDF filename")
    total_pages: int = Field(..., ge=0, description="Total number of pages processed")
    pages: List[ParsedPage] = Field(default_factory=list, description="Processed page results")
    pipeline_audit_log: List[Dict[str, Any]] = Field(
        default_factory=list, description="Audit trace of decisions, conversions, and checks"
    )
