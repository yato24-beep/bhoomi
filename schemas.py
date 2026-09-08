"""Core data schemas for the Land Record Digitization System.

These schemas define common data structures exchanged across components:
- Preprocessing & Document Layout
- OCR & Handwriting Recognition
- Information Extraction & Field Validation
- Database & Integration
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates for a region or text line in an image."""
    x_min: int = Field(..., description="Top-left x-coordinate")
    y_min: int = Field(..., description="Top-left y-coordinate")
    x_max: int = Field(..., description="Bottom-right x-coordinate")
    y_max: int = Field(..., description="Bottom-right y-coordinate")


class OCRResult(BaseModel):
    """Result of an OCR or handwritten text recognition operation."""
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


class ExtractedField(BaseModel):
    """An individual structured field extracted from the document."""
    field_name: str = Field(..., description="Name or key of the extracted field")
    field_value: str = Field(..., description="Normalized value of the extracted field")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence score")
    source_text: Optional[str] = Field(None, description="Raw source text before normalization")
    is_validated: bool = Field(False, description="Whether the field passed domain validation checks")


class LandRecord(BaseModel):
    """Normalized structured land record entity."""
    record_id: str = Field(..., description="Unique identifier for the record")
    document_type: Optional[str] = Field(None, description="Type of land record document")
    fields: Dict[str, ExtractedField] = Field(default_factory=dict, description="Extracted key-value fields")
    pages: List[DocumentPage] = Field(default_factory=list, description="Associated document pages")
    status: str = Field("pending", description="Processing status (e.g. pending, validated, flagged)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional ingestion or audit metadata")


class TrainingSample(BaseModel):
    """Schema for OCR / handwritten text model training samples."""
    image_path: str = Field(..., description="Path to the cropped text image")
    ground_truth: str = Field(..., description="Ground truth text transcription")
    split: str = Field("train", description="Dataset split: train, val, or test")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Sample metadata such as source document or script")
