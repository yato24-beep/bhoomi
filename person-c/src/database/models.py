"""
src/database/models.py
SQLAlchemy ORM models for PostgreSQL, PostGIS spatial tables, pgvector embeddings, and Active Learning corrections.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class DocumentModel(Base):
    """Stores metadata and status of processed land documents."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(128), unique=True, index=True, nullable=False)
    sha256_hash = Column(String(64), index=True, nullable=False)
    state = Column(String(32), index=True, nullable=False)
    document_type = Column(String(64), default="unknown")
    raw_full_text = Column(Text, default="")
    overall_confidence = Column(Float, default=0.0)
    validation_status = Column(String(32), default="unverified")
    requires_human_review = Column(Boolean, default=False)
    review_reasons = Column(JSON, default=list)
    processing_time_ms = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    fields = relationship("ExtractedFieldModel", back_populates="document", cascade="all, delete-orphan")
    corrections = relationship("CorrectionModel", back_populates="document", cascade="all, delete-orphan")


class ExtractedFieldModel(Base):
    """Stores individual extracted structured fields with full provenance and confidence."""
    __tablename__ = "extracted_fields"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(128), ForeignKey("documents.document_id", ondelete="CASCADE"), index=True, nullable=False)
    field_name = Column(String(64), index=True, nullable=False)
    raw_value = Column(Text, nullable=False)
    normalized_value = Column(Text, nullable=False)
    raw_unit = Column(String(32), nullable=True)
    normalized_unit = Column(String(32), nullable=True)
    confidence = Column(Float, nullable=False)
    page_number = Column(Integer, default=1)
    bbox_json = Column(JSON, nullable=True)
    evidence_json = Column(JSON, nullable=True)
    extraction_method = Column(String(32), default="regex")
    validation_status = Column(String(32), default="unverified")
    validation_messages = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    document = relationship("DocumentModel", back_populates="fields")


class CadastralParcelModel(Base):
    """
    Cadastral parcel spatial table for PostGIS spatial queries.
    Stores administrative hierarchy, Khasra number, polygon geometry, and registered land area.
    """
    __tablename__ = "cadastral_parcels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    state = Column(String(32), index=True, nullable=False)
    district = Column(String(64), index=True, nullable=False)
    tehsil = Column(String(64), index=True, nullable=False)
    village = Column(String(64), index=True, nullable=False)
    khasra_number = Column(String(64), index=True, nullable=False)
    area_hectares = Column(Float, nullable=False)
    gis_layer_name = Column(String(128), default="cadastral_layer_v1")
    # Spatial metadata / GeoJSON representation
    geometry_geojson = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class CorrectionModel(Base):
    """
    Active Learning and Human-in-the-loop corrections repository.
    Stores original prediction alongside verified human ground truth for retraining loops.
    """
    __tablename__ = "corrections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(128), ForeignKey("documents.document_id", ondelete="CASCADE"), index=True, nullable=False)
    field_name = Column(String(64), index=True, nullable=False)
    original_prediction = Column(Text, nullable=False)
    corrected_value = Column(Text, nullable=False)
    page_number = Column(Integer, default=1)
    bbox_json = Column(JSON, nullable=True)
    model_version = Column(String(64), default="v1.0.0")
    dataset_batch_id = Column(String(64), default="batch_001")
    is_promoted = Column(Boolean, default=False)
    corrected_by = Column(String(64), default="human_annotator")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    document = relationship("DocumentModel", back_populates="corrections")
