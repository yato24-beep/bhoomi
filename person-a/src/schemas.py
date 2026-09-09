"""person-a/src/schemas.py
Pydantic v2 schemas for Person A vision, preprocessing, layout, and printed OCR pipeline.
Designed for clean serializability and seamless mapping to downstream Person C contracts.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator


class BlockType(str, Enum):
    HEADER = "header"
    TITLE = "title"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    TABLE_CELL = "table_cell"
    HANDWRITTEN = "handwritten"
    SIGNATURE = "signature"
    STAMP = "stamp"
    FOOTER = "footer"
    UNKNOWN = "unknown"


class OCREngineType(str, Enum):
    PADDLE_OCR = "PaddleOCR-PP-OCRv5"
    PP_STRUCTURE = "PP-StructureV3-MorphologicalFallback"
    TROCR = "TrOCR-PersonB"
    MOCK_ENGINE = "mock_engine"


class BoundingBox(BaseModel):
    """Spatial bounding box representation [x_min, y_min, x_max, y_max]."""
    x_min: float = Field(..., description="Left coordinate")
    y_min: float = Field(..., description="Top coordinate")
    x_max: float = Field(..., description="Right coordinate")
    y_max: float = Field(..., description="Bottom coordinate")
    normalized: bool = Field(default=False, description="Whether coordinates are normalized in [0, 1]")

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoundingBox":
        if self.x_max < self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must be >= x_min ({self.x_min})")
        if self.y_max < self.y_min:
            raise ValueError(f"y_max ({self.y_max}) must be >= y_min ({self.y_min})")
        if self.normalized:
            for val, name in [(self.x_min, "x_min"), (self.y_min, "y_min"),
                              (self.x_max, "x_max"), (self.y_max, "y_max")]:
                if not (0.0 <= val <= 1.05):
                    raise ValueError(f"Normalized coordinate {name}={val} must be in [0.0, 1.0]")
        return self

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height


class ImageQualityAssessment(BaseModel):
    """Calculated physical and digital quality metrics for an image page."""
    dpi: float = Field(default=300.0, description="Estimated dots-per-inch")
    blur_score: float = Field(..., description="Laplacian variance sharpness score")
    is_blurry: bool = Field(default=False, description="True if blur_score falls below blur threshold")
    contrast_score: float = Field(..., description="RMS pixel intensity standard deviation")
    illumination_uniformity: float = Field(default=1.0, description="Quadrant intensity uniformity [0.0, 1.0]")
    skew_angle: float = Field(default=0.0, description="Detected skew angle in degrees")
    rotation_needed: int = Field(default=0, description="Detected 90-degree step rotation (0, 90, 180, 270)")
    is_blank: bool = Field(default=False, description="True if page contains virtually no foreground ink")
    preprocessing_applied: List[str] = Field(default_factory=list, description="List of operations applied")


class LanguageDetectionResult(BaseModel):
    """Detected language probabilities, fallback metadata, and selected OCR script."""
    requested_language: str = Field(default="en", description="Requested ISO 639-1 code")
    actual_language: str = Field(default="en", description="Actual language model executed")
    engine: str = Field(default="PaddleOCR-PP-OCRv5", description="Underlying OCR engine")
    fallback_occurred: bool = Field(default=False, description="True if fallback occurred due to missing checkpoint")
    fallback_reason: Optional[str] = Field(default=None, description="Detailed explanation of fallback rationale")
    primary_language: str = Field(default="en", description="Primary detected/effective language code")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    secondary_languages: List[str] = Field(default_factory=list)
    script: str = Field(default="Devanagari", description="Script name")


class OCRWord(BaseModel):
    """Fine-grained token/word bounding box and confidence."""
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    language: Optional[str] = None


class OCRLine(BaseModel):
    """Single line of recognized text composed of words."""
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    words: List[OCRWord] = Field(default_factory=list)
    language: Optional[str] = None


class OCRBlock(BaseModel):
    """Segmented structural block composed of multiple lines."""
    block_type: BlockType = BlockType.PARAGRAPH
    bbox: BoundingBox
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    lines: List[OCRLine] = Field(default_factory=list)
    text: str = ""


class TableCell(BaseModel):
    """Individual cell in a recognized table grid."""
    row_index: int
    col_index: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    bbox: Optional[BoundingBox] = None
    confidence: float = 1.0


class TableStructure(BaseModel):
    """Detected table structure with cells, headers, and markdown preview."""
    table_id: str
    page_number: int = 1
    bbox: BoundingBox
    rows_count: int
    cols_count: int
    cells: List[TableCell] = Field(default_factory=list)
    headers: List[str] = Field(default_factory=list)
    rows_data: List[List[str]] = Field(default_factory=list)
    markdown: Optional[str] = None
    dataframe_json: Optional[str] = None
    confidence: float = 1.0


class ClassificationResult(BaseModel):
    """Classification of document type, jurisdiction state, and supporting evidence."""
    predicted_type: str = Field(default="unknown", description="Canonical document type (e.g., 'land_record_7_12', 'rtc')")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    state: Optional[str] = Field(default=None, description="Identified State code ('MH', 'KA', 'UP', 'TN', etc.)")
    matched_signals: List[str] = Field(default_factory=list)
    all_scores: Dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


class OCRPageResult(BaseModel):
    """Complete multi-modal extraction for a single document page."""
    page_number: int
    width: int
    height: int
    text: str = ""
    confidence: float = 1.0
    ocr_engine: str = OCREngineType.PADDLE_OCR.value
    quality: ImageQualityAssessment
    language_info: Optional[LanguageDetectionResult] = None
    blocks: List[OCRBlock] = Field(default_factory=list)
    tables: List[TableStructure] = Field(default_factory=list)


class OCROutput(BaseModel):
    """Person A Primary Pipeline Result across all pages."""
    document_id: str
    sha256_hash: str
    ocr_engine: str = OCREngineType.PADDLE_OCR.value
    overall_confidence: float = 1.0
    classification: ClassificationResult = Field(default_factory=ClassificationResult)
    pages: List[OCRPageResult] = Field(default_factory=list)
    full_text: str = ""
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentInput(BaseModel):
    """Polymorphic input definition for Person A process_document(...) orchestrator."""
    document_id: str
    file_path: Optional[str] = None
    file_bytes: Optional[bytes] = None
    state_hint: Optional[str] = None
    target_languages: List[str] = Field(default_factory=lambda: ["en"])
    metadata: Dict[str, Any] = Field(default_factory=dict)
