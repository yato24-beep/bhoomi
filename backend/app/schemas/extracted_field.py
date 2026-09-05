from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


class ExtractedFieldRead(BaseModel):
    """Schema for individual extracted field record with spatial coordinates."""
    id: int = Field(..., description="Unique field ID in PostgreSQL")
    document_id: int = Field(..., description="ID of parent document")
    field_name: str = Field(..., description="Canonical field key name")
    original_value: Optional[str] = Field(default=None, description="Raw OCR value")
    normalized_value: Optional[str] = Field(default=None, description="Cleaned, standardized value")
    confidence_score: float = Field(..., description="Extraction confidence score (0.0 to 1.0)")
    source_page: int = Field(..., description="1-indexed source document page")
    bounding_box: Optional[Dict[str, Any]] = Field(default=None, description="Normalized spatial coordinates")
    created_at: datetime = Field(..., description="Timestamp when the field was saved")

    model_config = ConfigDict(from_attributes=True)


class ExtractedFieldsSummary(BaseModel):
    """Aggregated response containing all extracted fields for a document."""
    document_id: int = Field(..., description="ID of the document")
    total_fields: int = Field(..., description="Total count of extracted fields")
    average_confidence: float = Field(..., description="Average confidence score across all fields")
    fields: List[ExtractedFieldRead] = Field(..., description="List of granular extracted fields")
