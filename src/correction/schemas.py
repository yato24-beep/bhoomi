"""Schemas for Human Corrections and Active Learning Data Collection.

Defines the contract for capturing human verification, ground-truth corrections,
and preparing high-quality data for future active learning cycles without auto-retraining.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from schemas import BoundingBox


class HumanCorrectionRecord(BaseModel):
    """Schema for recording a single human-verified correction for a recognized region."""
    model_config = ConfigDict(protected_namespaces=())

    correction_id: str = Field(..., description="Unique UUID or identifier for this correction entry")
    document_id: Optional[str] = Field(None, description="Source document identifier if available")
    region_id: str = Field(..., description="Region identifier matching the OCRResult")
    image_path: str = Field(..., description="Reference path or URI to the crop image")
    crop_bbox: Optional[BoundingBox] = Field(None, description="Bounding box of the crop within the document")
    raw_prediction: str = Field(..., description="Verbatim AI recognition output before human review")
    corrected_text: str = Field(..., description="Verified ground-truth transcription provided by human reviewer")
    ai_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI model confidence score at inference time")
    language: str = Field("kannada", description="Language of the corrected text (e.g. 'kannada')")
    script: str = Field("Kannada", description="Script of the corrected text")
    is_handwritten: bool = Field(True, description="Whether the sample is handwritten text")
    model_version: Optional[str] = Field(None, description="Model version or checkpoint that produced raw_prediction")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of the correction",
    )
    reviewer_id: Optional[str] = Field(None, description="Identifier of the human reviewer / operator")
    notes: Optional[str] = Field(None, description="Optional diagnostic notes or error classification")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional context")

    @field_validator("corrected_text")
    @classmethod
    def validate_corrected_text(cls, v: str) -> str:
        """Ensures human ground-truth correction is not empty or pure whitespace."""
        if not v or not v.strip():
            raise ValueError("Corrected ground-truth text cannot be empty or whitespace-only")
        return v.strip()

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        """Ensures image path reference is provided."""
        if not v or not v.strip():
            raise ValueError("Image path reference cannot be empty")
        return v.strip()
