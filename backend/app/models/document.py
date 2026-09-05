from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import Integer, String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Document(Base):
    """Document model representing ingested files and their metadata."""
    
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False, doc="SHA-256 hash of the file contents")
    status: Mapped[str] = mapped_column(String(50), default="UPLOADED", nullable=False, index=True, doc="UPLOADED, PROCESSING, COMPLETED, FAILED")
    storage_path: Mapped[str] = mapped_column(String(500), nullable=True, doc="Path or MinIO object key in storage")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    # 1-to-1 Relationship with ExtractionResult (aggregate document summary)
    extraction_result = relationship(
        "ExtractionResult",
        back_populates="document",
        uselist=False,
        cascade="all, delete-orphan",
    )

    # 1-to-Many Relationship with ExtractedField (granular key-value fields with bounding boxes)
    extracted_fields = relationship(
        "ExtractedField",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="ExtractedField.id",
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}', hash='{self.file_hash[:8]}...')>"
