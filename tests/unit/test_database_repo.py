"""
Unit tests for Database Repository and persistence layer.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from schemas import (
    BoundingBox,
    DocumentClassificationResult,
    DocumentOCRResult,
    DocumentType,
    ExtractedField,
    FieldEvidence,
    FinalDocumentResult,
    ValidationStatus,
)
from src.database.models import Base, DocumentModel, ExtractedFieldModel
from src.database.repository import DocumentRepository


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_save_and_retrieve_document(db_session):
    repo = DocumentRepository(db_session)

    bbox = BoundingBox(x_min=10, y_min=10, x_max=100, y_max=50)
    field = ExtractedField(
        field_name="khasra_number",
        raw_value="142/1",
        normalized_value="142/1",
        confidence=0.98,
        page=1,
        bbox=bbox,
        evidence=FieldEvidence(page_number=1, bbox=bbox, raw_ocr_text="142/1"),
        validation_status=ValidationStatus.VALID,
    )

    final_doc = FinalDocumentResult(
        document_id="DOC_DB_TEST_01",
        sha256_hash="hash_db_01",
        state="UP",
        document_type=DocumentType.KHATAUNI,
        fields={"khasra_number": field},
        overall_confidence=0.98,
        validation_status=ValidationStatus.VALID,
    )

    # Save to database
    saved_doc = repo.save_final_document(final_doc)
    assert saved_doc.document_id == "DOC_DB_TEST_01"

    # Query from database
    retrieved = repo.get_document("DOC_DB_TEST_01")
    assert retrieved is not None
    assert retrieved.sha256_hash == "hash_db_01"
    assert len(retrieved.fields) == 1
    assert retrieved.fields[0].field_name == "khasra_number"
    assert retrieved.fields[0].normalized_value == "142/1"
