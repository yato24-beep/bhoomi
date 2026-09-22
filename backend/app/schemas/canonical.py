"""Canonical Document Processing Schemas & Data Contract.

Defines the single source of truth for document data structures across:
- Document metadata & storage references
- Raw and normalized OCR results
- Language detection & translation outputs
- Extracted cadastral fields
- Validation, confidence calibration, and human review items
- Provenance and processing stage metadata
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    """Canonical processing lifecycle states."""
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    FAILED = "FAILED"


class ReviewStatus(str, Enum):
    """Human review decision status."""
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    CORRECTED = "CORRECTED"
    REJECTED = "REJECTED"


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    unit: str = "pixel"


class OCRResult(BaseModel):
    """Raw and normalized text extracted from document."""
    raw_ocr_text: str = Field(..., description="Verbatim recognized source text")
    normalized_text: str = Field(..., description="Unicode NFC normalized and cleaned text")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    execution_provider: str = Field(default="wasm", description="wasm or webgpu")
    latency_ms: int = Field(default=0)
    tokens: Optional[List[int]] = None

    # Backward compatibility properties
    @property
    def original_ocr(self) -> str:
        return self.raw_ocr_text

    @property
    def clean_kannada_text(self) -> str:
        return self.normalized_text


class TranslationContract(BaseModel):
    """Translation metadata with strict language separation."""
    source_text: str
    translated_text: Optional[str] = None
    source_lang: str = "kn"
    target_lang: str = "en"
    status: str = "COMPLETED"
    engine: str = "domain_glossary_v1"
    error: Optional[str] = None


class ExtractedFieldContract(BaseModel):
    """Single structured cadastral field."""
    field_name: str
    raw_value: str
    normalized_value: str
    english_value: Optional[str] = None
    confidence: float = 0.85
    validation_status: str = "valid"
    translation_status: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None


class ReviewItemContract(BaseModel):
    """Item flagged for human inspection or correction."""
    review_id: str
    field_name: Optional[str] = None
    raw_ocr_text: str
    suggested_value: Optional[str] = None
    confidence: float = 0.85
    review_reason: str
    status: ReviewStatus = ReviewStatus.PENDING
    decision: Optional[str] = None
    corrected_text: Optional[str] = None


class ValidationContract(BaseModel):
    """Business rule and schema validation result."""
    is_valid: bool = True
    requires_human_review: bool = False
    verification_status: str = "accepted"
    checks_passed: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CanonicalDocumentPayload(BaseModel):
    """Canonical document processing result representation."""
    document_id: int
    filename: str
    file_hash: str
    storage_path: Optional[str] = None
    status: ProcessingStatus = ProcessingStatus.COMPLETED

    ocr: OCRResult
    translation: TranslationContract
    extracted_fields: Dict[str, ExtractedFieldContract] = Field(default_factory=dict)
    bilingual_fields: Dict[str, Dict[str, Optional[str]]] = Field(default_factory=dict)
    validation: ValidationContract = Field(default_factory=ValidationContract)
    review_items: List[ReviewItemContract] = Field(default_factory=list)

    provenance: str = Field(default="BROWSER_HYBRID_OCR", description="BROWSER_HYBRID_OCR | VERIFIED_SAMPLE | SERVER_PIPELINE")
    overall_confidence: float = 0.85
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
