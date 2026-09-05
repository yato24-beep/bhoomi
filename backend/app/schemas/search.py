from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from app.schemas.extracted_field import ExtractedFieldRead


class DocumentSearchItem(BaseModel):
    """Document search result item with matching field snippets."""
    id: int = Field(..., description="Document ID")
    filename: str = Field(..., description="Document filename")
    file_hash: str = Field(..., description="SHA-256 hash of the file")
    status: str = Field(..., description="Document status (UPLOADED, PROCESSING, COMPLETED, FAILED)")
    storage_path: Optional[str] = Field(default=None, description="MinIO storage path")
    created_at: datetime = Field(..., description="Creation timestamp")
    matched_fields: List[ExtractedFieldRead] = Field(
        default_factory=list,
        description="Extracted key-value fields matching the search query",
    )
    match_source: str = Field(
        default="both",
        description="Source of match: 'filename', 'extracted_fields', or 'both'",
    )

    model_config = ConfigDict(from_attributes=True)


class DocumentSearchResponse(BaseModel):
    """Paginated search response."""
    query: str = Field(..., description="Search query string")
    total_results: int = Field(..., description="Total count of matching documents")
    skip: int = Field(default=0, description="Offset used")
    limit: int = Field(default=50, description="Limit used")
    results: List[DocumentSearchItem] = Field(..., description="Matching document records")
