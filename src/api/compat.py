"""FastAPI compatibility router for frontend document management.

Provides backwards-compatible `/api/v1/documents/*` endpoints for the Next.js frontend,
delegating all OCR, multimodal routing (PaddleOCR/TrOCR), and Person C extraction directly
to the existing production `process_document_image` pipeline in `src.api.router`.
"""

from datetime import datetime
import hashlib
import io
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.router import process_document_image, export_ocr_pdf, export_ocr_docx
from backend.app.schemas.document import DocumentRead, DocumentUploadResponse
from backend.app.schemas.extraction import ExtractionResultRead
from backend.app.schemas.extracted_field import ExtractedFieldRead, ExtractedFieldsSummary

logger = logging.getLogger("ocr_compat")

router = APIRouter(prefix="/api/v1/documents", tags=["Frontend Compatibility"])

# In-memory document storage for session persistence
_DOCUMENTS: Dict[int, Dict[str, Any]] = {}
_DOCUMENT_IMAGES: Dict[int, bytes] = {}
_DOCUMENT_CONTENT_TYPES: Dict[int, str] = {}
_DOCUMENT_RESULTS: Dict[int, Any] = {}
_NEXT_DOC_ID: int = 1


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Document and Process via Multimodal OCR Pipeline",
)
async def upload_document_compat(
    file: UploadFile = File(..., description="Uploaded document image or PDF"),
) -> DocumentUploadResponse:
    """Accepts document file from frontend, runs TrOCR / OCR pipeline, and stores result."""
    global _NEXT_DOC_ID

    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a valid filename.")

    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Calculate SHA-256
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Check for duplicate document
    for doc_id, doc in _DOCUMENTS.items():
        if doc.get("file_hash") == file_hash:
            doc_read = DocumentRead(
                id=doc_id,
                filename=doc["filename"],
                file_hash=doc["file_hash"],
                status=doc["status"],
                storage_path=doc["storage_path"],
                created_at=doc["created_at"],
            )
            return DocumentUploadResponse(
                message="Document previously ingested (duplicate hash detected)",
                is_duplicate=True,
                document=doc_read,
                task_id=f"local_{doc_id}",
            )

    doc_id = _NEXT_DOC_ID
    _NEXT_DOC_ID += 1

    # Execute production OCR pipeline without duplicating any logic
    upload_file_for_pipeline = UploadFile(
        filename=file.filename,
        file=io.BytesIO(file_bytes),
        headers=file.headers,
    )

    try:
        ocr_response = await process_document_image(
            file=upload_file_for_pipeline,
            language="kannada",
            is_handwritten=None,
            document_id=f"doc_{doc_id}",
            page_number=1,
            regions=None,
            apply_preprocessing=True,
            apply_normalization=True,
            selected_state=None,
        )
    except Exception as e:
        logger.error(f"OCR Pipeline processing error for document {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {e}")

    now = datetime.utcnow()
    doc_record = {
        "id": doc_id,
        "filename": file.filename,
        "file_hash": file_hash,
        "status": "COMPLETED",
        "storage_path": f"local://documents/{doc_id}/{file.filename}",
        "created_at": now,
    }

    _DOCUMENTS[doc_id] = doc_record
    _DOCUMENT_IMAGES[doc_id] = file_bytes
    _DOCUMENT_CONTENT_TYPES[doc_id] = file.content_type or "image/png"
    _DOCUMENT_RESULTS[doc_id] = ocr_response

    doc_read = DocumentRead(
        id=doc_id,
        filename=doc_record["filename"],
        file_hash=doc_record["file_hash"],
        status=doc_record["status"],
        storage_path=doc_record["storage_path"],
        created_at=doc_record["created_at"],
    )

    return DocumentUploadResponse(
        message="Document uploaded and processed successfully",
        is_duplicate=False,
        document=doc_read,
        task_id=f"local_{doc_id}",
    )


@router.get("", response_model=List[DocumentRead], summary="List all documents")
@router.get("/", response_model=List[DocumentRead], include_in_schema=False)
async def list_documents_compat(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
) -> List[DocumentRead]:
    """Returns paginated list of uploaded documents."""
    docs = sorted(_DOCUMENTS.values(), key=lambda d: d["id"], reverse=True)
    sliced = docs[skip : skip + limit]
    return [
        DocumentRead(
            id=d["id"],
            filename=d["filename"],
            file_hash=d["file_hash"],
            status=d["status"],
            storage_path=d["storage_path"],
            created_at=d["created_at"],
        )
        for d in sliced
    ]


@router.get("/{document_id}", response_model=DocumentRead, summary="Get document by ID")
async def get_document_compat(document_id: int) -> DocumentRead:
    """Returns metadata for a specific document."""
    doc = _DOCUMENTS.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document #{document_id} not found.")
    return DocumentRead(
        id=doc["id"],
        filename=doc["filename"],
        file_hash=doc["file_hash"],
        status=doc["status"],
        storage_path=doc["storage_path"],
        created_at=doc["created_at"],
    )


@router.get("/{document_id}/status", summary="Get document processing status")
async def get_document_status_compat(document_id: int) -> Dict[str, Any]:
    """Returns processing status for a specific document."""
    doc = _DOCUMENTS.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document #{document_id} not found.")
    return {
        "id": doc["id"],
        "filename": doc["filename"],
        "status": doc["status"],
        "storage_path": doc["storage_path"],
    }


@router.get("/{document_id}/results", response_model=ExtractionResultRead, summary="Get extraction results")
async def get_document_results_compat(document_id: int) -> ExtractionResultRead:
    """Returns structured extraction and validation results."""
    doc = _DOCUMENTS.get(document_id)
    ocr_res = _DOCUMENT_RESULTS.get(document_id)
    if not doc or not ocr_res:
        raise HTTPException(status_code=404, detail=f"Results for document #{document_id} not found.")

    kannada_txt = getattr(ocr_res, "original_kannada_text", "") or getattr(ocr_res, "merged_text", "") or ""
    english_txt = getattr(ocr_res, "translated_text", "") or getattr(ocr_res, "merged_text", "") or ""

    extracted_data: Dict[str, Any] = {
        "document_id": document_id,
        "filename": doc["filename"],
        "original_kannada_text": kannada_txt,
        "clean_kannada_text": kannada_txt,
        "translated_text": english_txt,
        "merged_text": getattr(ocr_res, "merged_text", ""),
        "extracted_fields": getattr(ocr_res, "extracted_fields", {}),
        "document_type": getattr(ocr_res, "document_type", "Land Record"),
        "document_type_label": getattr(ocr_res, "document_type", "Land Record"),
        "is_land_record": getattr(ocr_res, "is_land_record", True),
        "state": getattr(ocr_res, "state", "KA"),
        "tables": [t.model_dump() if hasattr(t, "model_dump") else t for t in getattr(ocr_res, "tables", [])],
        "regions": [r.model_dump() if hasattr(r, "model_dump") else r for r in getattr(ocr_res, "ordered_regions", [])],
        "review_items": [it.model_dump() if hasattr(it, "model_dump") else it for it in getattr(ocr_res, "review_items", [])],
        "calibrated_confidence": getattr(ocr_res, "calibrated_confidence", None),
        "confidence_state": getattr(getattr(ocr_res, "confidence_state", None), "value", getattr(ocr_res, "confidence_state", "UNCALIBRATED")),
        "recognition_confidence": getattr(ocr_res, "recognition_confidence", None),
        "detection_confidence": getattr(ocr_res, "detection_confidence", None),
        "routing_confidence": getattr(ocr_res, "routing_confidence", None),
        "field_confidence": getattr(ocr_res, "field_confidence", None),
        "verification_status": getattr(ocr_res, "verification_status", "needs_verification" if getattr(ocr_res, "requires_human_review", False) else "accepted"),
        "semantic_data": getattr(ocr_res, "semantic_data", None),
        "gate_classification": getattr(ocr_res, "gate_classification", None),
        "stage_timings": getattr(ocr_res, "stage_timings", {}),
        "bilingual_fields": getattr(ocr_res, "bilingual_fields", {}),
    }

    # Flatten extracted field values for direct lookup while preserving translations
    for fname, fval in getattr(ocr_res, "extracted_fields", {}).items():
        if isinstance(fval, dict):
            extracted_data[fname] = fval.get("normalized_value") or fval.get("raw_value")
            if fval.get("english_value"):
                extracted_data[f"{fname}_english"] = fval.get("english_value")
            if fval.get("raw_value"):
                extracted_data[f"{fname}_kannada"] = fval.get("raw_value")
            if fval.get("translation_status"):
                extracted_data[f"{fname}_translation_status"] = fval.get("translation_status")
        else:
            extracted_data[fname] = str(fval)

    raw_conf = getattr(ocr_res, "overall_confidence", None) or getattr(ocr_res, "document_confidence", None) or 0.85
    conf = float(raw_conf)
    val_dict = getattr(ocr_res, "validation", None)
    if not val_dict or not isinstance(val_dict, dict):
        validation_info = {"status": "valid", "warnings": list(getattr(ocr_res, "warnings", []) or [])}
    else:
        validation_info = dict(val_dict)
    validation_info["requires_human_review"] = getattr(ocr_res, "requires_human_review", False)
    validation_info["warnings"] = list(getattr(ocr_res, "warnings", []) or [])
    is_valid = validation_info.get("status") != "rejected" and not getattr(ocr_res, "requires_human_review", False)

    return ExtractionResultRead(
        id=document_id,
        document_id=document_id,
        extracted_data=extracted_data,
        confidence_score=round(conf, 4),
        is_valid=is_valid,
        validation_info=validation_info if isinstance(validation_info, dict) else {},
        processing_time_ms=int(getattr(ocr_res, "processing_time_ms", 120)),
        created_at=doc["created_at"],
    )


class ReviewCorrectionRequest(BaseModel):
    review_id: str
    decision: str = "CORRECTED"  # ACCEPTED, CORRECTED, REJECTED
    corrected_text: Optional[str] = None
    reviewer_notes: Optional[str] = None
    reviewed_by: Optional[str] = "reviewer_officer"


@router.get("/{document_id}/review/items", summary="Get review items for document")
async def get_document_review_items_compat(document_id: int):
    doc = _DOCUMENTS.get(document_id)
    ocr_res = _DOCUMENT_RESULTS.get(document_id)
    if not doc or not ocr_res:
        raise HTTPException(status_code=404, detail=f"Document #{document_id} not found.")
    raw_items = getattr(ocr_res, "review_items", [])
    return [it.model_dump() if hasattr(it, "model_dump") else it for it in raw_items]


@router.post("/{document_id}/review", summary="Submit human review decision")
async def submit_document_review_compat(
    document_id: int,
    payload: ReviewCorrectionRequest,
):
    doc = _DOCUMENTS.get(document_id)
    ocr_res = _DOCUMENT_RESULTS.get(document_id)
    if not doc or not ocr_res:
        raise HTTPException(status_code=404, detail=f"Document #{document_id} not found.")

    review_items = getattr(ocr_res, "review_items", [])
    target_item = None
    item_idx = -1
    for idx, item in enumerate(review_items):
        r_id = item.get("review_id") if isinstance(item, dict) else getattr(item, "review_id", None)
        if r_id == payload.review_id:
            target_item = item if isinstance(item, dict) else item.model_dump()
            item_idx = idx
            break

    if not target_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review item '{payload.review_id}' not found in document {document_id}",
        )

    raw_ocr = target_item.get("raw_ocr_text", "")
    reviewer = payload.reviewed_by or "reviewer_officer"
    timestamp = datetime.utcnow().isoformat()
    decision_upper = payload.decision.upper()

    if decision_upper in ("ACCEPTED", "APPROVED"):
        target_item["decision"] = "ACCEPTED"
        target_item["status"] = "AUTO_ACCEPT"
        target_item["lifecycle_state"] = "APPROVED"
        target_item["corrected_text"] = raw_ocr
    elif decision_upper in ("CORRECTED", "EDITED"):
        target_item["decision"] = "CORRECTED"
        target_item["status"] = "AUTO_ACCEPT"
        target_item["lifecycle_state"] = "EDITED"
        target_item["corrected_text"] = (payload.corrected_text or "").strip()
        # Update linked field in extracted_fields if applicable
        meta = target_item.get("metadata") or {}
        fname = meta.get("field_name")
        if fname and hasattr(ocr_res, "extracted_fields") and isinstance(ocr_res.extracted_fields, dict):
            if fname in ocr_res.extracted_fields:
                fval = ocr_res.extracted_fields[fname]
                if isinstance(fval, dict):
                    fval["normalized_value"] = target_item["corrected_text"]
                    fval["validation_status"] = "VALID"
                else:
                    ocr_res.extracted_fields[fname] = target_item["corrected_text"]
    elif decision_upper == "REJECTED":
        target_item["decision"] = "REJECTED"
        target_item["status"] = "REJECTED_FAILED"
        target_item["lifecycle_state"] = "REJECTED"
        target_item["corrected_text"] = None
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid decision '{payload.decision}'. Must be ACCEPTED, CORRECTED, or REJECTED.",
        )

    target_item["raw_ocr_text"] = raw_ocr  # Strictly immutable
    target_item["reviewed_by"] = reviewer
    target_item["reviewed_at"] = timestamp
    target_item["reviewer_notes"] = payload.reviewer_notes

    review_items[item_idx] = target_item

    # Check if any remaining pending items require review
    still_pending = any(
        it.get("lifecycle_state") in ("PENDING", "ASSIGNED", "IN_REVIEW")
        or it.get("status") == "REVIEW_REQUIRED"
        for it in (it if isinstance(it, dict) else it.model_dump() for it in review_items)
    )

    if not still_pending:
        ocr_res.requires_human_review = False
        ocr_res.verification_status = "accepted"
        if hasattr(ocr_res, "validation") and isinstance(ocr_res.validation, dict):
            ocr_res.validation["requires_human_review"] = False
            ocr_res.validation["status"] = "valid"

    return {
        "status": "success",
        "review_id": payload.review_id,
        "document_id": document_id,
        "item": target_item,
        "requires_human_review": still_pending,
    }



@router.get("/{document_id}/fields", response_model=ExtractedFieldsSummary, summary="Get document fields")
async def get_document_fields_compat(
    document_id: int,
    min_confidence: Optional[float] = Query(None),
    field_name: Optional[str] = Query(None),
) -> ExtractedFieldsSummary:
    """Returns granular extracted fields summary."""
    doc = _DOCUMENTS.get(document_id)
    ocr_res = _DOCUMENT_RESULTS.get(document_id)
    if not doc or not ocr_res:
        raise HTTPException(status_code=404, detail=f"Fields for document #{document_id} not found.")

    raw_fields = getattr(ocr_res, "extracted_fields", {}) or {}
    field_items: List[ExtractedFieldRead] = []
    fid = 1

    for fname, fval in raw_fields.items():
        if field_name and field_name.lower() not in fname.lower():
            continue

        raw_val = fval.get("raw_value") if isinstance(fval, dict) else str(fval)
        norm_val = fval.get("normalized_value") if isinstance(fval, dict) else str(fval)
        conf = float(fval.get("confidence", 0.85)) if isinstance(fval, dict) else 0.85
        bbox = fval.get("bbox") if isinstance(fval, dict) else None

        if min_confidence is not None and conf < min_confidence:
            continue

        field_items.append(
            ExtractedFieldRead(
                id=fid,
                document_id=document_id,
                field_name=fname,
                original_value=raw_val,
                normalized_value=norm_val,
                confidence_score=round(conf, 4),
                source_page=1,
                bounding_box=bbox,
                english_value=fval.get("english_value") if isinstance(fval, dict) else None,
                translation_status=fval.get("translation_status") if isinstance(fval, dict) else None,
                translation_engine=fval.get("translation_engine") if isinstance(fval, dict) else None,
                created_at=doc["created_at"],
            )
        )
        fid += 1

    avg_conf = (
        sum(f.confidence_score for f in field_items) / len(field_items)
        if field_items
        else 0.85
    )

    return ExtractedFieldsSummary(
        document_id=document_id,
        total_fields=len(field_items),
        average_confidence=round(avg_conf, 4),
        fields=field_items,
    )


@router.get("/{document_id}/download", summary="Download original document file")
async def download_document_compat(document_id: int):
    """Streams original uploaded file bytes."""
    doc = _DOCUMENTS.get(document_id)
    img_bytes = _DOCUMENT_IMAGES.get(document_id)
    content_type = _DOCUMENT_CONTENT_TYPES.get(document_id, "image/png")
    if not doc or not img_bytes:
        raise HTTPException(status_code=404, detail=f"Image for document #{document_id} not found.")

    return StreamingResponse(
        io.BytesIO(img_bytes),
        media_type=content_type,
        headers={"Content-Disposition": f'inline; filename="{doc["filename"]}"'},
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete document")
async def delete_document_compat(document_id: int):
    """Removes a document from memory store."""
    if document_id not in _DOCUMENTS:
        raise HTTPException(status_code=404, detail=f"Document #{document_id} not found.")
    _DOCUMENTS.pop(document_id, None)
    _DOCUMENT_IMAGES.pop(document_id, None)
    _DOCUMENT_CONTENT_TYPES.pop(document_id, None)
    _DOCUMENT_RESULTS.pop(document_id, None)
    return None


@router.get("/{document_id}/export/pdf", summary="Export document PDF")
async def export_pdf_compat(document_id: int):
    """Exports digitized document as PDF."""
    doc = _DOCUMENTS.get(document_id)
    ocr_res = _DOCUMENT_RESULTS.get(document_id)
    if not doc or not ocr_res:
        raise HTTPException(status_code=404, detail=f"Document #{document_id} not found.")

    payload = ocr_res.model_dump() if hasattr(ocr_res, "model_dump") else dict(ocr_res)
    payload["document_id"] = str(document_id)
    payload["image_path"] = doc["filename"]
    return await export_ocr_pdf(payload)


@router.get("/{document_id}/export/docx", summary="Export document DOCX")
async def export_docx_compat(document_id: int):
    """Exports digitized document as DOCX."""
    doc = _DOCUMENTS.get(document_id)
    ocr_res = _DOCUMENT_RESULTS.get(document_id)
    if not doc or not ocr_res:
        raise HTTPException(status_code=404, detail=f"Document #{document_id} not found.")

    payload = ocr_res.model_dump() if hasattr(ocr_res, "model_dump") else dict(ocr_res)
    payload["document_id"] = str(document_id)
    payload["image_path"] = doc["filename"]
    return await export_ocr_docx(payload)


@router.post("/translate", summary="Translate text interactively")
async def translate_compat(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Interactive translation endpoint using field translator."""
    text = payload.get("text", "")
    source_lang = payload.get("source_lang", "auto")
    target_lang = payload.get("target_lang", "en")

    from src.translation.field_translator import transliterate_kannada_phrase, CADASTRAL_TRANSLATIONS
    translated = CADASTRAL_TRANSLATIONS.get(text.strip()) or transliterate_kannada_phrase(text.strip())

    return {
        "original_text": text,
        "translated_text": translated or text,
        "source_lang": source_lang,
        "target_lang": target_lang,
    }
