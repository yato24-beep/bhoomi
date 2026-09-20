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
    WORD = "word"
    LINE = "line"
    FIELD = "field"
    PAGE = "page"
    TABLE_CELL = "table_cell"
    HEADER = "header"
    SIGNATURE = "signature"
    STAMP = "stamp"
    THUMBPRINT = "thumbprint"
    MAP_OR_DIAGRAM = "map_or_diagram"
    UNKNOWN = "unknown"


class ConfidenceState(str, Enum):
    """Documented confidence states preventing premature score conversion."""
    UNCALIBRATED = "UNCALIBRATED"
    CALIBRATED = "CALIBRATED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


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
    recognizer_confidence_raw: Optional[float] = Field(None, description="Raw model confidence or posterior probability")
    calibrated_confidence: Optional[float] = Field(None, description="Calibrated confidence probability; MUST be null if uncalibrated")
    detection_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Detection confidence for this region/word")
    routing_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Script/style routing classification confidence")
    validation_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Semantic field validation confidence score")
    confidence_state: ConfidenceState = Field(ConfidenceState.UNCALIBRATED, description="Documented confidence state: UNCALIBRATED, CALIBRATED, REVIEW_REQUIRED")
    region_type: str = Field("word", description="Granularity of region: 'word', 'line', 'field', or 'page'")
    routing_reason: Optional[str] = Field(None, description="Deterministic rule applied to route to this recognizer")
    review_reason: Optional[str] = Field(None, description="Explicit reason why human review is required")
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
    original_ocr: Optional[str] = Field(None, description="Verbatim raw OCR recognized text (representation 1)")
    original_kannada_text: Optional[str] = Field(None, description="Preserved raw recognized Kannada script text")
    clean_kannada_text: Optional[str] = Field(None, description="Clean, normalized, natural Kannada text (representation 2)")
    kannada_translation: Optional[str] = Field(None, description="Clean, normalized, natural Kannada text (representation 2)")
    translated_text: Optional[str] = Field(None, description="English translation of recognized Indic/Kannada text (representation 3)")
    english_translation: Optional[str] = Field(None, description="Clean, natural, meaning-preserving English text (representation 3)")
    translation_result: Optional[Dict[str, Any]] = Field(default=None, description="Structured translation result with language purity report")
    translation_status: Optional[str] = Field(None, description="Status of machine translation ('completed', 'skipped_low_confidence', etc.)")
    document_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Character-length weighted mean confidence")
    
    # 4-part calibrated confidence reporting & explicit status
    recognition_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Calibrated recognition confidence")
    detection_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Calibrated layout/word detection confidence")
    routing_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Calibrated style/script routing confidence")
    field_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Calibrated spatial field extraction confidence")
    calibrated_confidence: Optional[float] = Field(None, description="Calibrated document confidence; MUST be null if uncalibrated")
    confidence_state: ConfidenceState = Field(ConfidenceState.UNCALIBRATED, description="Document-level confidence state: UNCALIBRATED, CALIBRATED, REVIEW_REQUIRED")
    verification_status: str = Field("needs_verification", description="'accepted' | 'needs_verification'")

    # Semantic layer V1 payload and human review queue
    semantic_data: Optional[Dict[str, Any]] = Field(default=None, description="Structured semantic fields, repeated records, and validation summary")
    review_items: List[Dict[str, Any]] = Field(default_factory=list, description="Items requiring human officer review")

    status: str = Field("completed", description="High-level processing status ('completed', 'flagged_for_review', 'failed')")
    requires_human_review: bool = Field(False, description="True if any region requires human verification")
    warnings: List[str] = Field(default_factory=list, description="Diagnostic warnings and review triggers")
    stage_timings: Dict[str, float] = Field(default_factory=dict, description="Per-stage latency breakdown in ms")
    structured_ocr: Optional[Dict[str, Any]] = Field(None, description="Standardized structured OCR JSON schema")
    engine_breakdown: Dict[str, int] = Field(default_factory=dict, description="Count of regions processed per engine")
    processing_time_ms: float = Field(0.0, description="Total pipeline latency in milliseconds")
    page: Optional[DocumentPage] = Field(None, description="DocumentPage schema for backwards compatibility with Person C")

    # Document Gate & Verification
    is_land_record: bool = Field(True, description="Whether document layout was verified as an authentic land record")
    gate_classification: Optional[Dict[str, Any]] = Field(default=None, description="Visual layout gate classification decision and metrics")
    bilingual_fields: Dict[str, Any] = Field(default_factory=dict, description="Word-aligned bilingual structured fields with transliterations")

    # Person C Integrated Intelligence Fields
    extracted_fields: Dict[str, Any] = Field(default_factory=dict, description="Structured fields extracted by Person C / NER")
    validation: Optional[Dict[str, Any]] = Field(default=None, description="Rule & cross-record validation results")
    gis_validation: Optional[Dict[str, Any]] = Field(default=None, description="Cadastral GIS validation report")
    duplicate_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Duplicate detection analysis report")
    overall_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Multi-factor document confidence score")
    document_type: Optional[str] = Field(default=None, description="Classified land record document type if confirmed, else None")
    document_type_state: str = Field(default="UNKNOWN", description="Document type state: 'CONFIRMED' or 'UNKNOWN'")
    classifier_source: Optional[str] = Field(default=None, description="Model or engine that produced classification")
    classifier_score: Optional[float] = Field(default=None, description="Classification score or posterior")
    classifier_evidence: List[str] = Field(default_factory=list, description="Textual or structural evidence for classification")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted table structures")
    pipeline_stages_completed: List[str] = Field(default_factory=list, description="Audit trail of completed pipeline stages")
    ocr: Optional[Dict[str, Any]] = Field(default=None, description="Preserved underlying OCR evidence breakdown")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="OCR recognizer diagnostic information")
    demo_fixture_detected: bool = Field(default=False, description="True if document content matches the controlled synthetic demo fixture")
    demo_fixture_name: Optional[str] = Field(default=None, description="Name of detected demo fixture")

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