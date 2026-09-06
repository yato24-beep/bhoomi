"""
src/database/repository.py
Repository pattern for document persistence, field storage, GIS lookup, and Active Learning corrections.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from loguru import logger

from schemas import ExtractedField, FinalDocumentResult
from src.database.models import (
    CadastralParcelModel,
    CorrectionModel,
    DocumentModel,
    ExtractedFieldModel,
)


class DocumentRepository:
    """Handles CRUD operations for processed documents, structured fields, and corrections."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def save_final_document(self, result: FinalDocumentResult) -> DocumentModel:
        """Persists a FinalDocumentResult and all its extracted fields."""
        # Check if document already exists
        doc = self.db.query(DocumentModel).filter(DocumentModel.document_id == result.document_id).first()
        if not doc:
            doc = DocumentModel(
                document_id=result.document_id,
                sha256_hash=result.sha256_hash,
                state=result.state,
                document_type=result.document_type.value,
                overall_confidence=result.overall_confidence,
                validation_status=result.validation_status.value,
                requires_human_review=result.requires_human_review,
                review_reasons=result.review_reasons,
                processing_time_ms=result.processing_time_total_ms,
            )
            self.db.add(doc)
            self.db.flush()
        else:
            doc.overall_confidence = result.overall_confidence
            doc.validation_status = result.validation_status.value
            doc.requires_human_review = result.requires_human_review
            doc.review_reasons = result.review_reasons
            doc.processing_time_ms = result.processing_time_total_ms

        # Remove existing fields if any (to avoid duplicates on update)
        self.db.query(ExtractedFieldModel).filter(ExtractedFieldModel.document_id == result.document_id).delete()

        # Save all extracted fields
        for field_name, field_obj in result.fields.items():
            field_record = ExtractedFieldModel(
                document_id=result.document_id,
                field_name=field_name,
                raw_value=field_obj.raw_value,
                normalized_value=str(field_obj.normalized_value),
                raw_unit=field_obj.raw_unit,
                normalized_unit=field_obj.normalized_unit,
                confidence=field_obj.confidence,
                page_number=field_obj.page,
                bbox_json=field_obj.bbox.model_dump() if field_obj.bbox else None,
                evidence_json=field_obj.evidence.model_dump() if field_obj.evidence else None,
                extraction_method=field_obj.extraction_method.value,
                validation_status=field_obj.validation_status.value,
                validation_messages=field_obj.validation_messages,
            )
            self.db.add(field_record)

        self.db.commit()
        self.db.refresh(doc)
        return doc

    def get_document(self, document_id: str) -> Optional[DocumentModel]:
        """Retrieves document record by document_id."""
        return self.db.query(DocumentModel).filter(DocumentModel.document_id == document_id).first()

    def get_document_by_hash(self, sha256_hash: str) -> Optional[DocumentModel]:
        """Retrieves document record by SHA-256 hash."""
        return self.db.query(DocumentModel).filter(DocumentModel.sha256_hash == sha256_hash).first()

    def save_correction(
        self,
        document_id: str,
        field_name: str,
        original_prediction: str,
        corrected_value: str,
        page_number: int = 1,
        bbox_json: Optional[Dict[str, Any]] = None,
        model_version: str = "v1.0.0",
        dataset_batch_id: str = "batch_001",
        corrected_by: str = "human_annotator",
    ) -> CorrectionModel:
        """Stores a human correction for Active Learning."""
        correction = CorrectionModel(
            document_id=document_id,
            field_name=field_name,
            original_prediction=original_prediction,
            corrected_value=corrected_value,
            page_number=page_number,
            bbox_json=bbox_json,
            model_version=model_version,
            dataset_batch_id=dataset_batch_id,
            corrected_by=corrected_by,
        )
        self.db.add(correction)
        self.db.commit()
        self.db.refresh(correction)
        logger.info(f"Logged Active Learning correction for doc {document_id}, field {field_name}: '{original_prediction}' -> '{corrected_value}'")
        return correction

    def get_unpromoted_corrections(self) -> List[CorrectionModel]:
        """Returns all corrections ready for active learning fine-tuning batch creation."""
        return self.db.query(CorrectionModel).filter(CorrectionModel.is_promoted == False).all()
