"""Public Integration Schemas for Land Record Digitization OCR & Handwriting Pipeline.

Defines stable, strongly-typed contracts for:
1. Person A (Layout Analysis & Region Detection) -> Person B Input
2. Person B -> Person C (Information Extraction & Database Storage) Output
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from pydantic import BaseModel, ConfigDict, Field

from schemas import BoundingBox, DocumentPage, OCRResult


class RegionType(str, Enum):
    """Semantic region types recognized in land records."""
    TEXT = "text"
    TABLE_CELL = "table_cell"
    HEADER = "header"
    SIGNATURE = "signature"
    STAMP = "stamp"
    THUMBPRINT = "thumbprint"
    MAP_OR_DIAGRAM = "map_or_diagram"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Document and region processing execution status."""
    SUCCESS = "success"
    LOW_CONFIDENCE = "low_confidence"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    EMPTY = "empty"
    ERROR = "error"


class RegionRequest(BaseModel):
    """Input specification for a discrete document sub-region or bounding box (from Person A)."""
    model_config = ConfigDict(protected_namespaces=())

    region_id: str = Field(..., description="Unique identifier for the region (e.g. 'reg_001')")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates (x_min, y_min, x_max, y_max)")
    language: Optional[str] = Field("kannada", description="Language code or script name (e.g. 'kannada', 'english')")
    script: Optional[str] = Field("Kannada", description="Script name (e.g. 'Kannada', 'Latin')")
    is_handwritten: Optional[bool] = Field(None, description="True=Handwritten, False=Printed, None=Auto-infer")
    region_type: RegionType = Field(RegionType.TEXT, description="Semantic region classification")
    layout_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score from upstream layout model")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary upstream metadata from layout analysis")


class DocumentProcessingRequest(BaseModel):
    """Full-document processing request submitted to Person B pipeline."""
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    image: Any = Field(..., description="Input image (file path, PIL Image, numpy array, or bytes)")
    document_id: Optional[str] = Field(None, description="Optional caller-supplied document identifier")
    page_number: int = Field(1, ge=1, description="1-indexed page number within the record")
    regions: Optional[List[RegionRequest]] = Field(None, description="Candidate bounding box regions from Person A")
    language: str = Field("kannada", description="Default document language (e.g. 'kannada', 'english')")
    script: str = Field("Kannada", description="Default document script")
    is_handwritten: Optional[bool] = Field(None, description="Default handwriting flag if not specified per region")
    apply_preprocessing: bool = Field(True, description="Whether to run non-destructive enhancement (deskew/denoise)")
    apply_normalization: bool = True
    document_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata passed through to final output")


class RecognizedRegionResult(BaseModel):
    """Strongly-typed output contract for each recognized region (consumed by Person C)."""
    model_config = ConfigDict(protected_namespaces=())

    region_id: str = Field(..., description="Identifier matching or assigned to the input region")
    raw_text: str = Field(..., description="Verbatim, unedited text directly produced by the OCR/handwriting model")
    normalized_text: str = Field(..., description="Conservative Unicode-normalized and cleaned text")
    bbox: Optional[BoundingBox] = Field(None, description="Region bounding box in document coordinate space")
    page_number: int = Field(1, description="1-indexed page number")
    language: str = Field(..., description="Recognized language identifier")
    script: str = Field(..., description="Recognized script identifier")
    is_handwritten: Optional[bool] = Field(None, description="Flag indicating if the text was recognized as handwritten")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model confidence score in [0.0, 1.0]")
    model_name: Optional[str] = Field(None, description="Engine or checkpoint used (e.g. TrOCR Kannada, PaddleOCR)")
    model_version: Optional[str] = Field(None, description="Model version or checkpoint identifier")
    inference_time_ms: float = Field(0.0, description="Inference execution duration in milliseconds")
    preprocessing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Details of image filters applied")
    requires_human_review: bool = Field(False, description="Flagged if confidence is low, language unsupported, or text empty")
    status: ProcessingStatus = Field(ProcessingStatus.SUCCESS, description="Execution status code")
    candidate_suggestions: List[str] = Field(default_factory=list, description="Candidate land record dictionary suggestions if applicable")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Audit trail and raw token probabilities")


class DocumentProcessingResponse(BaseModel):
    """Structured document-level response returned to Person C and downstream storage."""
    model_config = ConfigDict(protected_namespaces=())

    document_id: Optional[str] = Field(None, description="Document identifier if supplied in request")
    page_number: int = Field(1, description="1-indexed document page number")
    image_path: str = Field(..., description="Source image path or reference identifier")
    ordered_regions: List[RecognizedRegionResult] = Field(default_factory=list, description="Regions sorted in reading order")
    merged_text: str = Field(..., description="Full document text concatenated in natural reading order")
    document_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Character-length weighted mean confidence")
    status: str = Field("completed", description="High-level processing status ('completed', 'flagged_for_review', 'failed')")
    requires_human_review: bool = Field(False, description="True if any region requires human verification")
    warnings: List[str] = Field(default_factory=list, description="Diagnostic warnings and review triggers")
    engine_breakdown: Dict[str, int] = Field(default_factory=dict, description="Count of regions processed per engine")
    processing_time_ms: float = Field(0.0, description="Total pipeline latency in milliseconds")
    page: Optional[DocumentPage] = Field(None, description="DocumentPage schema for backwards compatibility with Person C")

    @property
    def full_text(self) -> str:
        """Backward-compatible alias for merged_text."""
        return self.merged_text


    @property
    def review_reasons(self) -> List[str]:
        """Backward-compatible alias for warnings."""
        return self.warnings


    @property
    def review_reason(self) -> Optional[str]:
        """Backward-compatible human review reason."""
        if not self.requires_human_review:
            return None

        return self.warnings[0] if self.warnings else None


    def to_dict(self) -> Dict[str, Any]:
        """Backward-compatible dictionary serialization."""
        return self.model_dump()