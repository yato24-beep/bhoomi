"""
src/api/router.py
FastAPI REST API router exposing Person C extraction, validation, GIS, duplicate, and active learning endpoints.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from schemas import (
    DocumentOCRResult,
    DuplicateResult,
    FinalDocumentResult,
    GISValidationResult,
    HandwritingResult,
)
from src.confidence.active_learning import ActiveLearningService
from src.database.db_session import get_db
from src.database.duplicates import DuplicateDetector
from src.database.gis import GISValidator
from src.database.repository import DocumentRepository
from src.integration.person_c_service import extract_and_validate
from src.utils.config_loader import ConfigLoader

router = APIRouter(prefix="/api/v1", tags=["Person C Pipeline"])

# Singletons for services
config_loader = ConfigLoader()
gis_validator = GISValidator()
dup_detector = DuplicateDetector()
al_service = ActiveLearningService()


class ProcessDocumentRequest(BaseModel):
    ocr_result: DocumentOCRResult
    handwriting_result: Optional[HandwritingResult] = None
    selected_state: Optional[str] = None
    persist_to_db: bool = True


class GISValidateRequest(BaseModel):
    state: str
    district: str
    tehsil: str
    village: str
    khasra_number: str
    land_area_hectares: Optional[float] = None


class DuplicateCheckRequest(BaseModel):
    document_id: str
    sha256_hash: str
    full_text: str


class CorrectionRequest(BaseModel):
    document_id: str
    field_name: str
    original_prediction: str
    corrected_value: str
    page_number: int = 1
    model_version: str = "v1.0.0"
    corrected_by: str = "human_reviewer"


@router.get("/health")
def health_check():
    """Health check endpoint displaying available states and pipeline configuration."""
    pipeline_cfg = config_loader.get_pipeline_config()
    return {
        "status": "healthy",
        "pipeline_version": pipeline_cfg.get("version", "1.0.0"),
        "supported_states": ["UP", "MP", "MH", "BR", "DEFAULT"],
        "active_services": ["extraction", "normalization", "rule_validation", "gis_validation", "duplicate_detection", "confidence_scoring"],
    }


@router.post("/process", response_model=FinalDocumentResult)
def process_document_endpoint(
    payload: ProcessDocumentRequest,
    db: Session = Depends(get_db),
):
    """
    Main Person C processing endpoint:
    Accepts Person A DocumentOCRResult and Person B HandwritingResult,
    runs extraction, normalization, validation, GIS, duplicates, and confidence scoring.
    """
    try:
        final_res = extract_and_validate(
            ocr_result=payload.ocr_result,
            handwriting_result=payload.handwriting_result,
            selected_state=payload.selected_state,
            gis_validator=gis_validator,
            duplicate_detector=dup_detector,
            config_loader=config_loader,
            db_session=db,  # enables real PostGIS/pgvector queries when DATABASE_URL is Postgres
        )

        if payload.persist_to_db:
            repo = DocumentRepository(db)
            repo.save_final_document(final_res)

        return final_res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing Person C pipeline: {str(e)}",
        )


@router.post("/gis/validate", response_model=GISValidationResult)
def validate_gis_endpoint(payload: GISValidateRequest, db: Session = Depends(get_db)):
    """Direct cadastral GIS spatial verification endpoint."""
    from schemas import ExtractedField
    fields = {
        "district": ExtractedField(field_name="district", raw_value=payload.district, normalized_value=payload.district, confidence=1.0),
        "tehsil": ExtractedField(field_name="tehsil", raw_value=payload.tehsil, normalized_value=payload.tehsil, confidence=1.0),
        "village": ExtractedField(field_name="village", raw_value=payload.village, normalized_value=payload.village, confidence=1.0),
        "khasra_number": ExtractedField(field_name="khasra_number", raw_value=payload.khasra_number, normalized_value=payload.khasra_number, confidence=1.0),
    }
    if payload.land_area_hectares is not None:
        fields["land_area"] = ExtractedField(
            field_name="land_area",
            raw_value=str(payload.land_area_hectares),
            normalized_value=payload.land_area_hectares,
            confidence=1.0,
        )

    return gis_validator.validate_gis(fields, state_code=payload.state, db_session=db)


@router.post("/duplicates/check", response_model=DuplicateResult)
def check_duplicate_endpoint(payload: DuplicateCheckRequest, db: Session = Depends(get_db)):
    """Checks for SHA-256 and vector cosine duplicate records (real pgvector query on Postgres)."""
    return dup_detector.check_duplicate(
        document_id=payload.document_id,
        sha256_hash=payload.sha256_hash,
        full_text=payload.full_text,
        db_session=db,
    )


@router.post("/corrections")
def submit_correction_endpoint(payload: CorrectionRequest, db: Session = Depends(get_db)):
    """Submits a human correction into the Active Learning feedback loop."""
    record = al_service.log_human_correction(
        document_id=payload.document_id,
        field_name=payload.field_name,
        original_prediction=payload.original_prediction,
        corrected_value=payload.corrected_value,
        page_number=payload.page_number,
        model_version=payload.model_version,
        corrected_by=payload.corrected_by,
        db_session=db,
    )
    return {"status": "success", "message": "Correction logged successfully", "record": record}
