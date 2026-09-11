"""Unified Data Contracts and Shared Schemas for the Document Processing Platform.

Jointly owned by Person A, Person B, and Person C.
Provides stable, strongly-typed contracts for:
- Preprocessing & Document Layout (Person A)
- OCR & Handwriting Recognition (Person B)
- Information Extraction, Validation, GIS & Confidence (Person C)
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# Core Enums
# ============================================================================

class ExtractionMethod(str, Enum):
    REGEX = "regex"
    KEY_VALUE = "key_value"
    TABLE_LOOKUP = "table_lookup"
    HANDWRITING_FUSION = "handwriting_fusion"
    STATE_RULE = "state_rule"
    FALLBACK = "fallback"


class ValidationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    UNVERIFIED = "unverified"
    CONFLICT = "conflict"


class OCREngineType(str, Enum):
    PADDLE_OCR = "paddleocr"
    PP_STRUCTURE = "pp_structure"
    TROCR = "trocr"
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"
    MOCK_ENGINE = "mock_engine"


class GISStatus(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class DocumentType(str, Enum):
    KHATAUNI = "khatauni"
    KHASRA = "khasra"
    SATBARA = "7_12_satbara"
    BHOOMI_RTC = "bhoomi_rtc"
    KHATA_CERTIFICATE = "khata_certificate"
    E_KHATA = "e_khata"
    KHATA_EXTRACT = "khata_extract"
    PATTA_CHITTA = "patta_chitta"
    A_REGISTER = "a_register"
    KHATIYAN = "khatiyan"
    JAMABANDI = "jamabandi"
    SALE_DEED = "sale_deed"
    MUTATION_REGISTER = "mutation_register"
    UNKNOWN = "unknown"


class DuplicateMatchType(str, Enum):
    EXACT_HASH = "exact_hash"
    VECTOR_SIMILARITY = "vector_similarity"
    FIELD_EXACT = "field_exact"
    NONE = "none"


# ============================================================================
# Geometry & Bounding Box
# ============================================================================

class BoundingBox(BaseModel):
    """Standard bounding box represented as [x_min, y_min, x_max, y_max].

    Compatible with both integer pixel coordinates and normalized float coordinates.
    """
    x_min: float = Field(..., description="Top-left X coordinate")
    y_min: float = Field(..., description="Top-left Y coordinate")
    x_max: float = Field(..., description="Bottom-right X coordinate")
    y_max: float = Field(..., description="Bottom-right Y coordinate")
    normalized: bool = Field(default=False, description="Whether coordinates are normalized between 0 and 1")

    model_config = ConfigDict(frozen=True)

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    def overlaps_with(self, other: "BoundingBox", iou_threshold: float = 0.3) -> bool:
        """Check intersection-over-union (IoU) with another bounding box."""
        inter_x_min = max(self.x_min, other.x_min)
        inter_y_min = max(self.y_min, other.y_min)
        inter_x_max = min(self.x_max, other.x_max)
        inter_y_max = min(self.y_max, other.y_max)

        inter_w = max(0.0, inter_x_max - inter_x_min)
        inter_h = max(0.0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h

        if inter_area <= 0:
            return False

        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return False

        return (inter_area / union_area) >= iou_threshold


# ============================================================================
# Person A/B Legacy & Pipeline Schemas
# ============================================================================

class OCRResult(BaseModel):
    """Result of an OCR or handwritten text recognition operation (Person B)."""
    model_config = ConfigDict(protected_namespaces=())

    text: str = Field(..., description="Recognized text string")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0, or None if unavailable")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box of the recognized text region")
    is_handwritten: Optional[bool] = Field(None, description="Flag indicating if the text was identified as handwritten")
    page_number: Optional[int] = Field(None, description="1-indexed page number if applicable")
    model_name: Optional[str] = Field(None, description="Name or identifier of the recognition model")
    model_version: Optional[str] = Field(None, description="Version or checkpoint tag of the recognition model")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional audit details, raw token confidences, or preprocessing info")


class DocumentPage(BaseModel):
    """Represents a single processed page of a land record document."""
    page_number: int = Field(..., description="1-indexed page number")
    image_path: str = Field(..., description="Path to the page image file")
    ocr_results: List[OCRResult] = Field(default_factory=list, description="List of recognized text lines/regions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata such as resolution or rotation")


class TrainingSample(BaseModel):
    """Schema for OCR / handwritten text model training samples."""
    image_path: str = Field(..., description="Path to the cropped text image")
    ground_truth: str = Field(..., description="Ground truth text transcription")
    split: str = Field("train", description="Dataset split: train, val, or test")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Sample metadata such as source document or script")


# ============================================================================
# Person A Schemas: Preprocessing, OCR & Layout
# ============================================================================

class DocumentInput(BaseModel):
    """Raw document input submitted to the pipeline."""
    document_id: str = Field(..., description="Unique document identifier or UUID")
    file_path: Optional[str] = Field(default=None, description="Local or remote path to original file")
    file_bytes: Optional[bytes] = Field(default=None, repr=False, description="Raw file binary content")
    sha256_hash: Optional[str] = Field(default=None, description="SHA-256 hash of original document bytes")
    selected_state: Optional[str] = Field(default=None, description="User-selected state code (e.g. 'UP', 'KA') or None for auto-detect")
    mime_type: str = Field(default="application/pdf", description="Document MIME type (pdf, png, jpeg, tiff)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional upload-time metadata")


class PreprocessingMetadata(BaseModel):
    """Metadata describing image preprocessing operations performed by Person A."""
    deskew_angle: float = Field(default=0.0, description="Angle rotated to correct skew in degrees")
    super_resolution_applied: bool = Field(default=False, description="Whether Real-ESRGAN super-resolution was applied")
    denoised: bool = Field(default=False, description="Whether denoising filters were executed")
    contrast_enhanced: bool = Field(default=False, description="Whether adaptive histogram equalization / contrast enhancement was run")
    original_resolution: Optional[Tuple[int, int]] = Field(default=None, description="(width, height) before processing")
    processed_resolution: Optional[Tuple[int, int]] = Field(default=None, description="(width, height) after processing")
    rotation_degrees: int = Field(default=0, description="Orientation correction in degrees (0, 90, 180, 270)")
    page_count: int = Field(default=1, description="Total number of pages processed")


class OCRTextLine(BaseModel):
    """Single OCR text line or detected text block."""
    text: str = Field(..., description="Recognized text string")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score between 0.0 and 1.0")
    bbox: BoundingBox = Field(..., description="Bounding box for the text line")
    page_number: int = Field(default=1, ge=1, description="1-indexed page number")
    language: str = Field(default="hi", description="Detected language code (e.g. 'hi', 'kn', 'en', 'mr')")
    engine: OCREngineType = Field(default=OCREngineType.PADDLE_OCR, description="OCR Engine used")
    source_region: Optional[str] = Field(default="body", description="Region type: 'header', 'footer', 'table_cell', 'stamp', 'margin'")


class TableCell(BaseModel):
    """Single cell within a detected table."""
    row_index: int
    col_index: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    bbox: Optional[BoundingBox] = None
    confidence: float = 1.0


class TableStructure(BaseModel):
    """Table detected by layout analysis."""
    table_id: str
    page_number: int = 1
    bbox: BoundingBox
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    cells: List[TableCell] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class DocumentClassificationResult(BaseModel):
    """Classification of document type and jurisdiction."""
    document_type: DocumentType = DocumentType.UNKNOWN
    state: str = Field(default="UNKNOWN", description="Detected or confirmed state code (e.g. 'UP', 'KA')")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    signals_matched: List[str] = Field(default_factory=list, description="Keywords or structural rules matched")
    language: str = Field(default="hi", description="Primary document language")


class DocumentOCRResult(BaseModel):
    """Standard output produced by Person A's process_document(...) interface."""
    document_id: str
    sha256_hash: str
    preprocessing: PreprocessingMetadata = Field(default_factory=PreprocessingMetadata)
    classification: DocumentClassificationResult = Field(default_factory=DocumentClassificationResult)
    text_lines: List[OCRTextLine] = Field(default_factory=list)
    tables: List[TableStructure] = Field(default_factory=list)
    raw_full_text: str = Field(default="", description="Concatenated raw full text")
    pages_processed: int = 1
    processing_time_ms: float = 0.0


# ============================================================================
# Person B Schemas: Handwriting Recognition
# ============================================================================

class HandwritingRegionResult(BaseModel):
    """Recognized handwritten region output from Person B's TrOCR pipeline."""
    region_id: str = Field(..., description="Unique region identifier")
    text: str = Field(..., description="Recognized handwritten text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="TrOCR model confidence")
    page_number: int = Field(default=1, ge=1)
    bbox: BoundingBox = Field(..., description="Bounding box of the handwritten crop")
    model_version: str = Field(default="trocr-base-landrecords-v1", description="TrOCR checkpoint/version identifier")
    source_region: Optional[str] = Field(default="handwriting_fill", description="Type: 'signature', 'stamp_text', 'margin_note', 'field_entry'")
    preprocessing_applied: List[str] = Field(default_factory=list, description="Pre-filtering applied before TrOCR")


class HandwritingResult(BaseModel):
    """Standard output produced by Person B's recognize_handwriting(...) interface."""
    document_id: str
    regions: List[HandwritingRegionResult] = Field(default_factory=list)
    model_version: str = "trocr-base-landrecords-v1"
    processing_time_ms: float = 0.0


# ============================================================================
# Person C Schemas: Extraction, Normalization, Validation, GIS, Confidence
# ============================================================================

class FieldEvidence(BaseModel):
    """Full provenance tracking object answering: 'Where did this value come from?'"""
    page_number: int = Field(default=1)
    bbox: Optional[BoundingBox] = None
    raw_ocr_text: str = Field(default="", description="Exact matching raw OCR line or snippet")
    handwriting_text: Optional[str] = Field(default=None, description="Handwriting recognition snippet if fused")
    ocr_engine: str = Field(default="paddleocr")
    extraction_rule_id: str = Field(default="regex_default", description="State config regex ID or rule name")
    source_record: Optional[str] = Field(default=None, description="Table name, line index, or key-value source")
    validation_notes: List[str] = Field(default_factory=list)


class ExtractedField(BaseModel):
    """Standard contract for an extracted field.

    Preserves raw_value, normalized_value, confidence, and provenance evidence.
    Includes backward-compatible properties for Person A/B consumers.
    """
    field_name: str = Field(..., description="Canonical field name (e.g. 'khasra_number', 'owner_name', 'land_area')")
    raw_value: str = Field(..., description="Unaltered raw extracted string from the document")
    normalized_value: Any = Field(..., description="Standardized, typed representation (float for area, ISO date, cleaned string)")
    raw_unit: Optional[str] = Field(default=None, description="Original unit if applicable (e.g. 'Bigha', 'Gunta', 'Acre')")
    normalized_unit: Optional[str] = Field(default=None, description="Standardized unit (e.g. 'hectare', 'sq_meter')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Field-level multi-factor confidence score")
    page: int = Field(default=1, ge=1)
    bbox: Optional[BoundingBox] = None
    evidence: FieldEvidence = Field(default_factory=FieldEvidence)
    extraction_method: ExtractionMethod = ExtractionMethod.REGEX
    validation_status: ValidationStatus = ValidationStatus.UNVERIFIED
    validation_messages: List[str] = Field(default_factory=list)

    @property
    def field_value(self) -> str:
        """Backward-compatible alias returning string representation of normalized value."""
        return str(self.normalized_value) if self.normalized_value is not None else self.raw_value

    @property
    def source_text(self) -> str:
        """Backward-compatible alias for raw source text."""
        return self.raw_value

    @property
    def is_validated(self) -> bool:
        """Backward-compatible boolean indicator for passing validation."""
        return self.validation_status == ValidationStatus.VALID


class RuleValidationItem(BaseModel):
    """Result of a single rule or constraint validation."""
    rule_name: str
    field_name: str
    passed: bool
    severity: str = Field(default="error", description="'error', 'warning', 'info'")
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None


class CrossRecordValidationResult(BaseModel):
    """Cross-record consistency validation (e.g. table sum vs header total)."""
    passed: bool = True
    inconsistencies: List[str] = Field(default_factory=list)
    area_sum_matches: Optional[bool] = None
    share_sum_matches: Optional[bool] = None
    khasra_count_matches: Optional[bool] = None


class GISValidationResult(BaseModel):
    """Result of PostGIS / Cadastral GIS spatial verification."""
    gis_status: GISStatus = Field(default=GISStatus.UNKNOWN, description="'MATCH', 'MISMATCH', 'UNKNOWN', 'INSUFFICIENT_DATA'")
    is_verified: bool = Field(default=False, description="True if geometry and jurisdiction matched PostGIS records")
    has_mismatch: bool = Field(default=False, description="True if spatial discrepancy detected")
    parcel_id_found: bool = Field(default=False)
    state: str = Field(default="")
    district: str = Field(default="")
    tehsil: str = Field(default="")
    village: str = Field(default="")
    khasra_number: str = Field(default="")
    gis_recorded_area_hectares: Optional[float] = None
    extracted_area_hectares: Optional[float] = None
    area_deviation_percent: Optional[float] = None
    coordinates_matched: Optional[bool] = None
    point_in_polygon_passed: Optional[bool] = Field(default=None, description="True if provided GPS coordinates fall inside parcel polygon")
    spatial_deviation_meters: Optional[float] = Field(default=None, description="Distance from centroid if coordinates supplied")
    flag_reasons: List[str] = Field(default_factory=list)
    source_gis_layer: Optional[str] = None
    data_source_label: str = Field(default="synthetic_demo_cadastral", description="'official_source_derived' or 'synthetic_demo_cadastral'")


class DuplicateResult(BaseModel):
    """Result of SHA-256 and vector duplicate analysis."""
    is_duplicate: bool = Field(default=False)
    match_type: DuplicateMatchType = DuplicateMatchType.NONE
    matched_document_id: Optional[str] = None
    sha256_match: bool = False
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    potential_duplicate_ids: List[str] = Field(default_factory=list)
    evidence: Optional[str] = None


class ConfidenceBreakdown(BaseModel):
    """Detailed multi-factor component breakdown for a field's confidence score."""
    ocr_confidence: float = 1.0
    pattern_strength: float = 1.0
    rule_validation_score: float = 1.0
    cross_record_consistency: float = 1.0
    gis_consistency: float = 1.0
    disagreement_penalty: float = 0.0
    final_score: float = 1.0


class FinalDocumentResult(BaseModel):
    """The complete, final structured document output of the entire pipeline.

    Produced by Person C's extract_and_validate interface.
    """
    document_id: str
    sha256_hash: str
    state: str
    document_type: DocumentType
    processing_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Structured Fields
    fields: Dict[str, ExtractedField] = Field(default_factory=dict)
    tables: List[TableStructure] = Field(default_factory=list)

    # Validation & Quality Reports
    validation_status: ValidationStatus = ValidationStatus.VALID
    validation_errors: List[RuleValidationItem] = Field(default_factory=list)
    cross_record_validation: CrossRecordValidationResult = Field(default_factory=CrossRecordValidationResult)
    gis_validation: GISValidationResult = Field(default_factory=GISValidationResult)
    duplicate_analysis: DuplicateResult = Field(default_factory=DuplicateResult)

    # Global & Field-Level Confidence
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    requires_human_review: bool = Field(default=False)
    review_reasons: List[str] = Field(default_factory=list)

    # Traceability & Audit Trail
    pipeline_stages_completed: List[str] = Field(default_factory=list)
    disagreements_resolved: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_total_ms: float = 0.0


class LandRecord(BaseModel):
    """Normalized structured land record entity for database storage (Person A)."""
    record_id: str = Field(..., description="Unique identifier for the record")
    document_type: Optional[str] = Field(None, description="Type of land record document")
    fields: Dict[str, ExtractedField] = Field(default_factory=dict, description="Extracted key-value fields")
    pages: List[DocumentPage] = Field(default_factory=list, description="Associated document pages")
    status: str = Field("pending", description="Processing status (e.g. pending, validated, flagged)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional ingestion or audit metadata")
