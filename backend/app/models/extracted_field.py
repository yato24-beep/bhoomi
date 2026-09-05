from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy import Integer, String, Float, Text, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ExtractedField(Base):
    """Stores individual extracted key-value fields with bounding boxes and normalization."""

    __tablename__ = "extracted_fields"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Foreign key reference to parent document",
    )
    field_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        doc="Canonical field name (e.g. invoice_number, total_amount, vendor_name)",
    )
    original_value: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Raw OCR string exactly as parsed from the document page",
    )
    normalized_value: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Standardized value (e.g. ISO date '2026-08-20', decimal '1350.00')",
    )
    confidence_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
        doc="Confidence score of this specific field (0.0 to 1.0)",
    )
    source_page: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
        doc="1-indexed document page number where the field was located",
    )
    bounding_box: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        doc="Coordinates on page: {'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float}",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    # Relationship back to Document
    document = relationship("Document", back_populates="extracted_fields")

    def __repr__(self) -> str:
        return (
            f"<ExtractedField(id={self.id}, doc_id={self.document_id}, "
            f"name='{self.field_name}', val='{self.normalized_value}', conf={self.confidence_score:.2f})>"
        )
