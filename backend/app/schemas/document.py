from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class DocumentBase(BaseModel):
    """Base schema for document fields."""
    filename: str = Field(..., max_length=255, description="Original name of the uploaded document")
    file_hash: str = Field(..., min_length=64, max_length=64, description="SHA-256 hash of the document content")
    status: str = Field(default="UPLOADED", max_length=50, description="Processing status (UPLOADED, PROCESSING, COMPLETED, FAILED)")
    storage_path: Optional[str] = Field(default=None, max_length=500, description="Storage location/key for the document file")


class DocumentCreate(DocumentBase):
    """Schema for creating a new document record."""
    pass


class DocumentUpdate(BaseModel):
    """Schema for updating an existing document record."""
    status: Optional[str] = Field(default=None, max_length=50)
    storage_path: Optional[str] = Field(default=None, max_length=500)


class DocumentRead(DocumentBase):
    """Schema for reading document records returned by the API."""
    id: int = Field(..., description="Unique database identifier")
    created_at: datetime = Field(..., description="Timestamp when the document record was created")

    model_config = ConfigDict(from_attributes=True)


class DocumentUploadResponse(BaseModel):
    """Response model returned when uploading a document."""
    message: str = Field(..., description="Descriptive status message of the upload result")
    is_duplicate: bool = Field(..., description="Indicates whether this document was previously ingested")
    document: DocumentRead = Field(..., description="The saved or existing document record")
    task_id: Optional[str] = Field(default=None, description="Celery background task ID if a new job was dispatched")
