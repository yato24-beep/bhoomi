"""FastAPI APIRouter for Land Record Digitization OCR & Handwriting Pipeline.

Exposes RESTful endpoints for:
1. Health Check (GET /health, GET /api/ocr/health)
2. Document & Crop OCR Processing (POST /api/ocr/process)

Reuses the existing production DocumentProcessingPipeline and LanguageScriptRouter
without duplicating any OCR or recognition logic.
"""

from datetime import datetime
import io
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from PIL import Image

from src.integration.schemas import DocumentProcessingResponse

# Configure logging
logger = logging.getLogger("ocr_api")

# Singleton pipeline instance to preserve cache and avoid redundant allocations
_PIPELINE_INSTANCE: Optional[Any] = None


def get_pipeline() -> Any:
    """Returns the shared singleton instance of DocumentProcessingPipeline lazily."""
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        from src.integration.document_pipeline import DocumentProcessingPipeline
        _PIPELINE_INSTANCE = DocumentProcessingPipeline()
    return _PIPELINE_INSTANCE


router = APIRouter(tags=["OCR"])


@router.get("/api/ocr/health", tags=["Health"])
async def ocr_health_check() -> Dict[str, Any]:
    """Health check endpoint exposing supported modalities and pipeline readiness."""
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "land-record-ocr-pipeline",
        "version": "1.0.0",
        "supported_modalities": {
            "printed_kannada": "EasyOCR (easyocr_kannada)",
            "handwritten_kannada": "TrOCR Checkpoint-12000 (models/trocr/checkpoint-12000)",
            "printed_english": "PaddleOCR (ppocr_v4_en)",
            "handwritten_english": "TrOCR (microsoft/trocr-small-handwritten)",
        },
        "supported_languages": pipeline.router.list_supported_languages(),
        "default_language": pipeline.default_language,
        "confidence_threshold": pipeline.confidence_threshold,
    }


@router.post(
    "/api/ocr/process",
    response_model=DocumentProcessingResponse,
    status_code=status.HTTP_200_OK,
    tags=["OCR"],
    summary="Process document image through multimodal OCR pipeline",
)
async def process_document_image(
    file: UploadFile = File(..., description="Document or crop image file (PNG, JPG, JPEG, TIFF, BMP, WEBP)"),
    language: Optional[str] = Form("kannada", description="Target language (e.g. 'kannada', 'english')"),
    is_handwritten: Optional[bool] = Form(None, description="Handwriting flag: True=Handwritten, False=Printed, None=Auto-infer"),
    document_id: Optional[str] = Form(None, description="Optional caller-provided document identifier"),
    page_number: int = Form(1, ge=1, description="1-indexed page number"),
    regions: Optional[str] = Form(None, description="Optional JSON array of candidate bounding box regions from layout analysis"),
    apply_preprocessing: bool = Form(True, description="Whether to apply deskew/denoise/contrast enhancement"),
    apply_normalization: bool = Form(True, description="Whether to apply conservative text normalization"),
    selected_state: Optional[str] = Form(None, description="Target state code (e.g. 'KA', 'UP', 'MP', 'MH') or None for auto-detect"),
) -> DocumentProcessingResponse:
    """Processes an uploaded document image and returns structured OCR results with Person C extraction & validation.

    Accepts an uploaded image and passes it through:
    1. Person A Image Preprocessing (deskew, denoise, enhance)
    2. Person B Multimodal OCR Routing (PaddleOCR printed, TrOCR handwriting)
    3. Person C Extraction & Validation (state rules, normalization, GIS, duplicates, confidence)
    """
    # Step 1: Validate and read uploaded file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    try:
        file_bytes = await file.read()
        if not file_bytes or len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}",
        )

    # Check max file size (50MB)
    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed upload size (50MB). Received {len(file_bytes) / (1024 * 1024):.1f}MB.",
        )

    # Step 2: Validate and load image/document pages via DocumentFormatAdapter
    try:
        from src.preprocessing.format_adapter import DocumentFormatAdapter, DocumentIngestionError
        pages = DocumentFormatAdapter.load_pages(file_bytes, filename=file.filename)
        if not pages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No readable pages found in document '{file.filename}'.",
            )
        target_page_idx = max(0, min(page_number - 1, len(pages) - 1))
        image_obj = pages[target_page_idx]
    except HTTPException:
        raise
    except DocumentIngestionError as die:
        logger.warning(f"Document ingestion error for file '{file.filename}': {die}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or unreadable document format: {die.message}",
        )
    except Exception as e:
        logger.warning(f"Invalid document format for file '{file.filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or unreadable document format for file '{file.filename}'. Error: {str(e)}",
        )


    # Step 3: Parse optional regions JSON if provided
    parsed_regions: Optional[List[Any]] = None
    if regions and regions.strip():
        try:
            parsed = json.loads(regions)
            if not isinstance(parsed, list):
                raise ValueError("The 'regions' field must be a valid JSON array/list.")
            parsed_regions = parsed
        except Exception as e:
            logger.warning(f"Invalid regions JSON parameter: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid 'regions' JSON: {str(e)}",
            )

    # Step 4: Execute DocumentProcessingPipeline (Person A & B)
    pipeline = get_pipeline()
    assigned_doc_id = document_id or f"doc_{int(time.time() * 1000)}"

    try:
        response = pipeline.process_document(
            image=image_obj,
            regions=parsed_regions,
            is_handwritten=is_handwritten,
            language=language or "kannada",
            page_number=page_number,
            document_id=assigned_doc_id,
            image_path=file.filename,
            apply_preprocessing=apply_preprocessing,
            apply_normalization=apply_normalization,
        )
    except Exception as e:
        logger.error(f"Pipeline processing failure for document '{assigned_doc_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed in OCR pipeline: {str(e)}",
        )

    # If document was rejected by the layout classifier gate, return early
    if not response.is_land_record:
        return response

    # Step 5: Execute Person C Structured Extraction, Normalization, GIS & Validation
    try:
        from src.integration.person_c_adapter import PersonCAdapter
        c_result = PersonCAdapter.execute_person_c(
            b_response=response,
            file_bytes=file_bytes,
            selected_state=selected_state,
        )

        # Attach underlying OCR details
        response.ocr = {
            "merged_text": response.merged_text,
            "document_confidence": response.document_confidence,
            "ordered_regions": [r.model_dump() for r in response.ordered_regions],
            "status": response.status,
            "warnings": response.warnings,
            "engine_breakdown": response.engine_breakdown,
            "processing_time_ms": response.processing_time_ms,
            "diagnostics": response.diagnostics,
        }

        # Populate Person C fields on response, preserving high-accuracy canonical semantic fields
        fields_dict: Dict[str, Any] = dict(response.extracted_fields or {})
        for fname, fval in c_result.fields.items():
            if fname not in fields_dict or not fields_dict[fname].get("raw_value"):
                fields_dict[fname] = {
                    "field_name": fval.field_name,
                    "raw_value": fval.raw_value,
                    "normalized_value": fval.normalized_value,
                    "raw_unit": fval.raw_unit,
                    "normalized_unit": fval.normalized_unit,
                    "confidence": round(fval.confidence, 4),
                    "page": fval.page,
                    "bbox": fval.bbox.model_dump() if fval.bbox else None,
                    "validation_status": fval.validation_status.value,
                    "validation_messages": fval.validation_messages,
                    "evidence": fval.evidence.model_dump() if fval.evidence else None,
                }

        # Provide aliases for Karnataka / South Indian and standard frontend field names
        if "khasra_number" in fields_dict and "survey_number" not in fields_dict:
            survey_copy = dict(fields_dict["khasra_number"])
            survey_copy["field_name"] = "survey_number"
            fields_dict["survey_number"] = survey_copy

        if "khatauni_number" in fields_dict and "property_number" not in fields_dict:
            prop_copy = dict(fields_dict["khatauni_number"])
            prop_copy["field_name"] = "property_number"
            fields_dict["property_number"] = prop_copy

        if "tehsil" in fields_dict and "taluk" not in fields_dict:
            taluk_copy = dict(fields_dict["tehsil"])
            taluk_copy["field_name"] = "taluk"
            fields_dict["taluk"] = taluk_copy

        if "document_date" in fields_dict and "date" not in fields_dict:
            date_copy = dict(fields_dict["document_date"])
            date_copy["field_name"] = "date"
            fields_dict["date"] = date_copy

        response.extracted_fields = fields_dict
        response.validation = {
            "status": c_result.validation_status.value,
            "errors": [e.model_dump() for e in c_result.validation_errors],
            "cross_record": c_result.cross_record_validation.model_dump(),
        }
        response.gis_validation = c_result.gis_validation.model_dump()
        response.duplicate_analysis = c_result.duplicate_analysis.model_dump()
        response.overall_confidence = round(float(c_result.overall_confidence), 4)
        response.state = c_result.state
        response.document_type = c_result.document_type.value
        response.tables = [t.model_dump() for t in c_result.tables]
        response.pipeline_stages_completed = c_result.pipeline_stages_completed

        # Merge review flags
        if c_result.requires_human_review:
            response.requires_human_review = True
            for r in c_result.review_reasons:
                if r and r not in response.warnings:
                    response.warnings.append(r)

    except Exception as c_err:
        logger.warning(f"Person C extraction notice for document '{assigned_doc_id}': {c_err}", exc_info=True)

    return response


@router.post(
    "/export/pdf",
    summary="Generate PDF from OCR Result",
    description="Generates a downloadable PDF report directly from digitized OCR payload.",
)
async def export_ocr_pdf(payload: Dict[str, Any]):
    """Generates PDF directly from OCR response body."""
    from backend.app.services.export import build_pdf_export
    doc_id = payload.get("document_id", "doc_1")
    filename = payload.get("image_path", "document.jpeg")
    conf = payload.get("overall_confidence") or payload.get("document_confidence") or 0.85
    status = payload.get("status", "completed")
    raw_fields = payload.get("extracted_fields", {})
    fields: Dict[str, str] = {}
    for k, v in raw_fields.items():
        if isinstance(v, dict):
            fields[k] = v.get("normalized_value") or v.get("raw_value") or ""
        else:
            fields[k] = str(v)
    kn_text = payload.get("original_kannada_text") or payload.get("merged_text") or ""
    en_text = payload.get("translated_text") or payload.get("merged_text") or ""
    req_review = payload.get("requires_human_review", False)
    warnings = payload.get("warnings", [])

    pdf_bytes = build_pdf_export(
        document_id=str(doc_id),
        filename=filename,
        overall_confidence=float(conf),
        status=status,
        fields=fields,
        kannada_text=kn_text,
        english_translation=en_text,
        requires_review=req_review,
        review_reasons=warnings,
    )
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="digitized_land_record.pdf"'},
    )


@router.post(
    "/export/docx",
    summary="Generate Word DOCX from OCR Result",
    description="Generates a downloadable DOCX report directly from digitized OCR payload.",
)
async def export_ocr_docx(payload: Dict[str, Any]):
    """Generates Word DOCX directly from OCR response body."""
    from backend.app.services.export import build_docx_export
    doc_id = payload.get("document_id", "doc_1")
    filename = payload.get("image_path", "document.jpeg")
    conf = payload.get("overall_confidence") or payload.get("document_confidence") or 0.85
    status = payload.get("status", "completed")
    raw_fields = payload.get("extracted_fields", {})
    fields: Dict[str, str] = {}
    for k, v in raw_fields.items():
        if isinstance(v, dict):
            fields[k] = v.get("normalized_value") or v.get("raw_value") or ""
        else:
            fields[k] = str(v)
    kn_text = payload.get("original_kannada_text") or payload.get("merged_text") or ""
    en_text = payload.get("translated_text") or payload.get("merged_text") or ""
    req_review = payload.get("requires_human_review", False)
    warnings = payload.get("warnings", [])

    docx_bytes = build_docx_export(
        document_id=str(doc_id),
        filename=filename,
        overall_confidence=float(conf),
        status=status,
        fields=fields,
        kannada_text=kn_text,
        english_translation=en_text,
        requires_review=req_review,
        review_reasons=warnings,
    )
    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'attachment; filename="digitized_land_record.docx"'},
    )
