"""Structured Output Schema for Document OCR and Region-Level Classification.

Provides a unified, strict schema for pages, regions, timing telemetry,
and calibrated confidence verification without text or score fabrication.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RegionType(str, Enum):
    WORD = "word"
    LINE = "line"
    FIELD = "field"
    PAGE = "page"


class ScriptType(str, Enum):
    KANNADA = "Kannada"
    ENGLISH = "English"
    MIXED = "Mixed"
    UNKNOWN = "Unknown"


class OCRRegion(BaseModel):
    """Granular region with deterministic routing, calibration, and review tracking."""
    region_id: str = Field(..., description="Unique region identifier within the page")
    bbox: List[int] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    region_type: RegionType = Field(default=RegionType.WORD, description="Granularity: word, line, field, or page")
    script: ScriptType = Field(default=ScriptType.KANNADA, description="Detected or requested script")
    handwriting: bool = Field(default=False, description="True if region was classified as handwritten")
    recognizer: str = Field(..., description="Engine ID used (e.g., 'easyocr', 'iitb_kannada_v002')")
    text: str = Field(default="", description="Recognized text in original script")
    recognizer_confidence_raw: Optional[float] = Field(
        default=None,
        description="Raw recognizer score or posterior probability (uncalibrated)"
    )
    calibrated_confidence: Optional[float] = Field(
        default=None,
        description="Calibrated empirical probability of correctness; null if uncalibrated"
    )
    needs_review: bool = Field(default=False, description="True if human review is strictly required")
    review_reason: Optional[str] = Field(default=None, description="Explicit actionable reason for review requirement")
    routing_reason: Optional[str] = Field(default=None, description="Deterministic rule applied to route to this recognizer")


class PageTimings(BaseModel):
    """Granular per-stage processing latency in milliseconds."""
    file_decode_ms: float = 0.0
    page_rendering_ms: float = 0.0
    layout_detection_ms: float = 0.0
    crop_generation_ms: float = 0.0
    ocr_inference_ms: float = 0.0
    reconstruction_ms: float = 0.0
    translation_ms: float = 0.0
    total_ms: float = 0.0


class OCRPage(BaseModel):
    """Page-level container maintaining sequential regions and page-specific timings."""
    page_number: int = Field(..., ge=1, description="1-indexed document page number")
    width: int = Field(..., ge=1, description="Page pixel width")
    height: int = Field(..., ge=1, description="Page pixel height")
    regions: List[OCRRegion] = Field(default_factory=list, description="Ordered regions on this page")
    merged_text: str = Field(default="", description="Reading-order merged text for this page")
    translated_text: Optional[str] = Field(default=None, description="English translation if eligible")
    translation_status: str = Field(default="pending", description="'completed', 'flagged_medium_confidence', or 'skipped_low_confidence'")
    timings: PageTimings = Field(default_factory=PageTimings, description="Per-stage latency breakdown")


class StructuredDocumentOCR(BaseModel):
    """Top-level standardized OCR output for an entire document."""
    document_id: str = Field(..., description="Document canonical identifier")
    document_type: str = Field(default="Unknown / Not classified", description="Verified document type or 'Unknown / Not classified'")
    is_land_record: Optional[bool] = Field(default=None, description="Gate classification result")
    pages: List[OCRPage] = Field(default_factory=list, description="Ordered list of processed pages")
    overall_confidence_calibrated: Optional[float] = Field(
        default=None,
        description="Empirically calibrated document confidence; null if uncalibrated"
    )
    status: str = Field(default="completed", description="Document status ('completed', 'flagged_for_review', 'rejected')")
    requires_human_review: bool = Field(default=False, description="True if any page or region requires review")
    total_timings: PageTimings = Field(default_factory=PageTimings, description="Aggregate latency across all pages")
