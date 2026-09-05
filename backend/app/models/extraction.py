from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from sqlalchemy import Integer, Float, Boolean, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ExtractionResult(Base):
    """Stores structured data, validation info, and confidence scores from document processing."""

    __tablename__ = "extraction_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
        doc="Foreign key reference to parent document",
    )
    
    # Structured extracted data (Key-value pairs, line items, parsed text)
    extracted_data: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        doc="Structured extracted fields and parsed payload",
    )
    
    # Model / Pipeline quality metrics
    confidence_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
        doc="Overall pipeline confidence score (0.0 to 1.0)",
    )
    
    # Validation flags
    is_valid: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether extracted fields passed business validation checks",
    )
    
    validation_info: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        doc="Details on validation checks, errors, and warnings",
    )
    
    processing_time_ms: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Execution duration in milliseconds",
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    document = relationship("Document", back_populates="extraction_result")

    def __repr__(self) -> str:
        return f"<ExtractionResult(id={self.id}, document_id={self.document_id}, confidence={self.confidence_score:.2f}, is_valid={self.is_valid})>"
