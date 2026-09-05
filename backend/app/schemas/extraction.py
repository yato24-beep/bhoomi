from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ExtractionResultRead(BaseModel):
    """Pydantic schema for reading extraction results from the API."""
    
    id: int = Field(..., description="Unique extraction result ID")
    document_id: int = Field(..., description="Referenced document ID")
    extracted_data: Dict[str, Any] = Field(..., description="Extracted structured data payload")
    confidence_score: float = Field(..., description="Extraction confidence score (0.0 to 1.0)")
    is_valid: bool = Field(..., description="Business validation status")
    validation_info: Dict[str, Any] = Field(..., description="Validation rules detail")
    processing_time_ms: int = Field(..., description="Processing duration in milliseconds")
    created_at: datetime = Field(..., description="Timestamp when results were saved")

    model_config = ConfigDict(from_attributes=True)
