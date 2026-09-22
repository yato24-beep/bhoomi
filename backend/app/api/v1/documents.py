import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
from fastapi import APIRouter, Depends, Form, HTTPException, status, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import select, or_, and_, delete
from pydantic import BaseModel, Field

from app.api.deps import (
    get_db,
    get_current_active_user,
    require_admin,
    require_officer_or_admin,
    require_viewer_or_above,
)
from app.config import settings
from app.core.hashing import calculate_stream_sha256
from app.services.minio_storage import minio_storage
from app.services.document_service import DocumentService
from app.services.translation_service import TranslationService
from app.workers.tasks import process_document_task
from app.models.document import Document
from app.models.extraction import ExtractionResult
from app.models.extracted_field import ExtractedField
from app.models.user import User
from app.schemas.document import (
    DocumentCreate,
    DocumentRead,
    DocumentUploadResponse,
)
from app.schemas.extraction import ExtractionResultRead
from app.schemas.extracted_field import ExtractedFieldRead, ExtractedFieldsSummary
from app.schemas.search import DocumentSearchItem, DocumentSearchResponse

router = APIRouter()


class PresignedUrlResponse(BaseModel):
    """Schema for returning temporary presigned download URLs."""
    download_url: str = Field(..., description="Temporary presigned MinIO/S3 download URL")
    expires_in_seconds: int = Field(default=3600, description="Expiration time in seconds")


class DocumentStatusResponse(BaseModel):
    """Schema for querying document processing status."""
    id: int
    filename: str
    status: str
    storage_path: Optional[str]


def _is_celery_available() -> bool:
    """Check if the Celery broker (e.g. Redis) is reachable or if running in eager mode."""
    from app.workers.celery_app import celery_app
    is_eager = False
    if isinstance(celery_app.conf, dict):
        is_eager = bool(celery_app.conf.get("task_always_eager"))
    else:
        is_eager = bool(getattr(celery_app.conf, "task_always_eager", False))
    if is_eager:
        return True
    try:
        import socket
        from urllib.parse import urlparse
        broker_url = settings.CELERY_BROKER_URL or f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
        parsed = urlparse(broker_url)
        host = parsed.hostname or settings.REDIS_HOST
        port = parsed.port or settings.REDIS_PORT or 6379
        with socket.create_connection((host, int(port)), timeout=0.2):
            return True
    except Exception:
        return False


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Document & Dispatch Processing Task",
    description="Upload a document to MinIO, record metadata, and dispatch Celery job. Allowed roles: ADMIN, OFFICER.",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (PDF, PNG, JPG, WEBP, TIFF)"),
    is_handwritten: Optional[bool] = Form(None, description="Whether document is handwritten"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_officer_or_admin),
) -> DocumentUploadResponse:
    """Upload a document, calculate SHA-256 hash, detect duplicates, and queue Celery processing task."""
    filename = file.filename or "uploaded_file.bin"
    file_ext = Path(filename).suffix.lower()

    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file format '{file_ext}'. "
                f"Allowed formats are: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
            ),
        )

    file_hash, file_size = calculate_stream_sha256(file.file)


    stmt = select(Document).where(Document.file_hash == file_hash)
    existing_document = db.execute(stmt).scalar_one_or_none()

    if existing_document:
        if existing_document.status not in ("COMPLETED", "PROCESSING"):
            existing_document.status = "UPLOADED"
            db.commit()
            db.refresh(existing_document)
            import threading
            thread = threading.Thread(
                target=process_document_task,
                args=(existing_document.id,),
                kwargs={"is_handwritten": is_handwritten},
                daemon=True,
                name=f"local-task-{existing_document.id}",
            )
            thread.start()
        return DocumentUploadResponse(
            message="Document already exists (duplicate detected via SHA-256 hash).",
            is_duplicate=True,
            document=existing_document,
            task_id="re-queued" if existing_document.status == "UPLOADED" else None,
        )

    content_type = file.content_type or "application/octet-stream"
    try:
        storage_path = minio_storage.upload_file(
            file_stream=file.file,
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            content_type=content_type,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to store file in MinIO object storage: {str(e)}",
        )

    new_document = Document(
        filename=filename,
        file_hash=file_hash,
        status="UPLOADED",
        storage_path=storage_path,
    )
    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    task_id = None
    if _is_celery_available():
        try:
            task = process_document_task.delay(new_document.id, is_handwritten=is_handwritten)
            task_id = getattr(task, "id", None) or "celery-queued-task"
        except Exception:
            task_id = None

    if not task_id:
        # Fallback to local async task execution when Redis/Celery broker is offline
        task_id = "local-sync-worker"
        import threading
        thread = threading.Thread(
            target=process_document_task,
            args=(new_document.id,),
            kwargs={"is_handwritten": is_handwritten},
            daemon=True,
            name=f"local-task-{new_document.id}",
        )
        thread.start()

    return DocumentUploadResponse(
        message="Document uploaded and processing job queued successfully.",
        is_duplicate=False,
        document=new_document,
        task_id=task_id,
    )


@router.get(
    "/search",
    response_model=DocumentSearchResponse,
    summary="Search Documents by Filename & Extracted Fields",
    description="Search documents across filenames and extracted key-value fields. Allowed roles: ALL USERS.",
)
def search_documents(
    q: str = Query(..., min_length=1, max_length=200, description="Search term for filename or extracted text"),
    field_name: Optional[str] = Query(default=None, description="Optional filter by specific field name (e.g. total_amount)"),
    status: Optional[str] = Query(default=None, description="Optional filter by status (UPLOADED, PROCESSING, COMPLETED, FAILED)"),
    skip: int = Query(default=0, ge=0, description="Number of results to skip"),
    limit: int = Query(default=50, ge=1, le=100, description="Max results to return"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
) -> DocumentSearchResponse:
    """Search documents across filenames and extracted fields with snippet provenance."""
    term = f"%{q.strip()}%"

    # Build conditions for extracted fields
    field_conditions = [
        ExtractedField.original_value.ilike(term),
        ExtractedField.normalized_value.ilike(term),
    ]
    if field_name:
        field_match_subquery = (
            select(ExtractedField.document_id)
            .where(
                and_(
                    ExtractedField.field_name == field_name,
                    or_(*field_conditions),
                )
            )
        )
    else:
        field_match_subquery = (
            select(ExtractedField.document_id)
            .where(or_(*field_conditions))
        )

    # Document matches either filename or matching extracted fields
    doc_filter = or_(
        Document.filename.ilike(term),
        Document.id.in_(field_match_subquery),
    )

    query = select(Document).where(doc_filter)
    if status:
        query = query.where(Document.status == status.upper().strip())

    total_docs = db.execute(query).scalars().all()
    total_count = len(total_docs)

    paged_query = query.order_by(Document.created_at.desc()).offset(skip).limit(limit)
    documents = db.execute(paged_query).scalars().all()

    results: List[DocumentSearchItem] = []
    for doc in documents:
        filename_matched = q.lower() in doc.filename.lower()

        f_query = select(ExtractedField).where(
            and_(
                ExtractedField.document_id == doc.id,
                or_(*field_conditions),
            )
        )
        if field_name:
            f_query = f_query.where(ExtractedField.field_name == field_name)

        matching_fields = list(db.execute(f_query).scalars().all())
        match_source = "both" if (filename_matched and matching_fields) else ("filename" if filename_matched else "extracted_fields")

        results.append(
            DocumentSearchItem(
                id=doc.id,
                filename=doc.filename,
                file_hash=doc.file_hash,
                status=doc.status,
                storage_path=doc.storage_path,
                created_at=doc.created_at,
                matched_fields=[ExtractedFieldRead.model_validate(f) for f in matching_fields],
                match_source=match_source,
            )
        )

    return DocumentSearchResponse(
        query=q,
        total_results=total_count,
        skip=skip,
        limit=limit,
        results=results,
    )


@router.get(
    "/{document_id}/fields",
    response_model=ExtractedFieldsSummary,
    summary="Get Extracted Fields",
    description="Retrieve all granular extracted fields with coordinates. Allowed roles: ALL USERS.",
)
def get_extracted_fields(
    document_id: int,
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Filter fields by minimum confidence"),
    field_name: Optional[str] = Query(default=None, description="Filter by specific field name"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
) -> ExtractedFieldsSummary:
    """List all extracted key-value fields with bounding boxes and confidence scores."""
    doc_stmt = select(Document).where(Document.id == document_id)
    document = db.execute(doc_stmt).scalar_one_or_none()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found",
        )

    query = select(ExtractedField).where(ExtractedField.document_id == document_id)
    if min_confidence is not None:
        query = query.where(ExtractedField.confidence_score >= min_confidence)
    if field_name:
        query = query.where(ExtractedField.field_name == field_name)

    query = query.order_by(ExtractedField.id.asc())
    fields = list(db.execute(query).scalars().all())

    avg_conf = (
        round(sum(f.confidence_score for f in fields) / len(fields), 2)
        if fields
        else 0.0
    )

    return ExtractedFieldsSummary(
        document_id=document_id,
        total_fields=len(fields),
        average_confidence=avg_conf,
        fields=fields,
    )


@router.get(
    "/{document_id}/fields/{field_name}",
    response_model=ExtractedFieldRead,
    summary="Get Specific Extracted Field",
    description="Fetch a single field by name. Allowed roles: ALL USERS.",
)
def get_specific_extracted_field(
    document_id: int,
    field_name: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
) -> ExtractedField:
    """Fetch a single field by name for a document."""
    stmt = (
        select(ExtractedField)
        .where(ExtractedField.document_id == document_id)
        .where(ExtractedField.field_name == field_name)
    )
    field = db.execute(stmt).scalar_one_or_none()
    if not field:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Field '{field_name}' not found for document ID {document_id}",
        )
    return field


@router.get(
    "/{document_id}/results",
    response_model=ExtractionResultRead,
    summary="Get Document Extraction Results",
    description="Retrieve aggregate parsed JSON data and validation info. Allowed roles: ALL USERS.",
)
def get_document_results(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
) -> ExtractionResult:
    """Fetch structured extraction summary for a processed document."""
    doc_stmt = select(Document).where(Document.id == document_id)
    document = db.execute(doc_stmt).scalar_one_or_none()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found",
        )

    res_stmt = select(ExtractionResult).where(ExtractionResult.document_id == document_id)
    result = db.execute(res_stmt).scalar_one_or_none()

    if not result:
        if document.status in ("UPLOADED", "PROCESSING"):
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail=f"Document is currently '{document.status}'. Results are not ready yet.",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extraction results for document with ID {document_id} not found",
        )
    ed = result.extracted_data or {}
    logger.info(
        f"[Audit:Stage6-ResultAPI] Serving results for doc_id={document_id} | "
        f"fields_returned={list(ed.keys())} | "
        f"has_original_ocr={bool(ed.get('original_ocr'))} | "
        f"original_ocr_len={len(ed.get('original_ocr', ''))} | "
        f"has_clean_kannada={bool(ed.get('clean_kannada_text'))} | "
        f"has_english={bool(ed.get('english_translation'))} | "
        f"confidence_score={result.confidence_score}"
    )

    return result


@router.get(
    "/{document_id}/reviews",
    summary="Get Document Human Review Items",
    description="Retrieve items flagged for human verification and officer review. Allowed roles: ALL USERS.",
)
def get_document_reviews(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
) -> List[Dict[str, Any]]:
    """Fetch human review queue items for a document."""
    res_stmt = select(ExtractionResult).where(ExtractionResult.document_id == document_id)
    result = db.execute(res_stmt).scalar_one_or_none()
    if not result or not result.extracted_data:
        return []

    extracted_data = result.extracted_data or {}
    review_items = list(extracted_data.get("review_items", []) or [])

    # Supplement with any fields marked with low confidence (< 0.70)
    fields_stmt = select(ExtractedField).where(ExtractedField.document_id == document_id)
    fields = db.execute(fields_stmt).scalars().all()
    for f in fields:
        if f.confidence_score < 0.70 and not any(r.get("field_name") == f.field_name for r in review_items):
            review_items.append({
                "region_id": f"field_{f.id}",
                "field_name": f.field_name,
                "raw_ocr_text": f.original_value or f.normalized_value,
                "calibrated_confidence": f.confidence_score,
                "review_reason": f"Field '{f.field_name}' requires verification (confidence: {f.confidence_score:.2f})",
                "suggested_value": f.normalized_value,
                "needs_review": True,
            })

    return review_items


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get Document Processing Status",
    description="Retrieve live processing status. Allowed roles: ALL USERS.",
)
def get_document_status(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
) -> DocumentStatusResponse:
    """Check processing status of a document."""
    stmt = select(Document).where(Document.id == document_id)
    document = db.execute(stmt).scalar_one_or_none()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found",
        )
    return DocumentStatusResponse(
        id=document.id,
        filename=document.filename,
        status=document.status,
        storage_path=document.storage_path,
    )


@router.post(
    "/",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create Document Record (JSON)",
    description="Manually create a document metadata record. Allowed roles: ADMIN, OFFICER.",
)
def create_document_record(
    doc_in: DocumentCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_officer_or_admin),
) -> Document:
    """Create a new document record from JSON payload."""
    document = Document(
        filename=doc_in.filename,
        file_hash=doc_in.file_hash,
        status=doc_in.status,
        storage_path=doc_in.storage_path,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


class BrowserOCRResultSaveRequest(BaseModel):
    """Schema for saving browser-generated TrOCR results directly."""
    filename: str = Field(..., max_length=255, description="Original filename of the uploaded document")
    file_hash: str = Field(..., description="SHA-256 hash of the document")
    file_size: Optional[int] = Field(default=0, description="Size of file in bytes")
    text: str = Field(..., description="Recognized Kannada text from browser TrOCR")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Confidence score from browser OCR")
    execution_provider: Optional[str] = Field(default="wasm", description="Provider used: webgpu or wasm")
    latency_ms: Optional[int] = Field(default=0, description="Local inference latency in ms")
    storage_path: Optional[str] = Field(default=None, description="Optional storage path")
    file_base64: Optional[str] = Field(default=None, description="Base64-encoded image bytes for persistent storage")


@router.post(
    "/browser-result",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save Browser TrOCR Result",
    description="Stores browser-executed TrOCR handwritten Kannada results directly into the database as completed, bypassing server-side model loading.",
)
def save_browser_result(
    payload: BrowserOCRResultSaveRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
) -> DocumentUploadResponse:
    """Persist browser-generated TrOCR result directly through DocumentService."""
    return DocumentService.process_and_persist_browser_result(
        filename=payload.filename,
        file_hash=payload.file_hash,
        file_size=payload.file_size,
        text=payload.text,
        confidence=payload.confidence,
        execution_provider=payload.execution_provider or "wasm",
        latency_ms=payload.latency_ms or 0,
        storage_path=payload.storage_path,
        file_base64=payload.file_base64,
        db=db,
    )


@router.get(
    "/",
    response_model=List[DocumentRead],
    summary="List Documents",
    description="Retrieve a list of documents. Allowed roles: ALL USERS.",
)
def list_documents(
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=50, ge=1, le=100, description="Max number of records to return"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
) -> List[Document]:
    """List document records from PostgreSQL/SQLite."""
    stmt = select(Document).order_by(Document.created_at.desc()).offset(skip).limit(limit)
    documents = db.execute(stmt).scalars().all()
    return list(documents)


@router.get(
    "/{document_id}",
    response_model=DocumentRead,
    summary="Get Document by ID",
    description="Fetch details of a single document. Allowed roles: ALL USERS.",
)
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
) -> Document:
    """Get document record by primary key."""
    stmt = select(Document).where(Document.id == document_id)
    document = db.execute(stmt).scalar_one_or_none()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found",
        )
    return document


@router.get(
    "/{document_id}/download",
    summary="Download Document from MinIO",
    description="Streams raw file from storage. Allowed roles: ALL USERS.",
)
def download_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
):
    """Download the raw file from storage."""
    stmt = select(Document).where(Document.id == document_id)
    document = db.execute(stmt).scalar_one_or_none()
    if not document or not document.storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found",
        )

    try:
        minio_obj = minio_storage.get_file_object(document.storage_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{document.storage_path}' not found in storage bucket",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving file: {str(e)}",
        )

    # Determine Content-Type from file extension for proper browser rendering
    ext = Path(document.filename).suffix.lower()
    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".pdf": "application/pdf",
        ".bmp": "image/bmp",
    }
    media_type = media_type_map.get(ext, "application/octet-stream")

    # Images should be displayed inline (for <img> tags), other files as attachments
    if media_type.startswith("image/"):
        disposition = f'inline; filename="{document.filename}"'
    else:
        disposition = f'attachment; filename="{document.filename}"'

    return StreamingResponse(
        minio_obj.stream(32 * 1024),
        media_type=media_type,
        headers={"Content-Disposition": disposition},
    )


def _get_document_export_payload(document_id: int, db: Session) -> Dict[str, Any]:
    """Helper to collect extraction data and fields for document export."""
    stmt = select(Document).where(Document.id == document_id)
    doc = db.execute(stmt).scalar_one_or_none()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found",
        )

    ext_stmt = select(ExtractionResult).where(ExtractionResult.document_id == document_id)
    ext = db.execute(ext_stmt).scalar_one_or_none()

    fields_stmt = select(ExtractedField).where(ExtractedField.document_id == document_id)
    field_rows = db.execute(fields_stmt).scalars().all()

    fields_dict: Dict[str, str] = {}
    for f in field_rows:
        val = f.normalized_value or f.original_value or ""
        fields_dict[f.field_name] = val

    extracted_data = ext.extracted_data if ext and ext.extracted_data else {}
    original_kannada = extracted_data.get("original_kannada_text") or extracted_data.get("merged_text") or ""
    translated_text = extracted_data.get("translated_text") or extracted_data.get("merged_text") or ""

    confidence = ext.confidence_score if ext else 0.85
    val_info = ext.validation_info if ext and ext.validation_info else {}
    requires_review = val_info.get("requires_human_review", False) or not (ext.is_valid if ext else True)
    review_reasons = val_info.get("warnings", [])

    return {
        "document_id": str(doc.id),
        "filename": doc.filename,
        "overall_confidence": confidence,
        "status": doc.status,
        "fields": fields_dict,
        "kannada_text": original_kannada,
        "english_translation": translated_text,
        "requires_review": requires_review,
        "review_reasons": review_reasons,
    }


@router.get(
    "/{document_id}/export/pdf",
    summary="Download Digitized Document as PDF",
    description="Generates a downloadable professional PDF report. Allowed roles: ALL USERS.",
)
def export_document_pdf(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
):
    """Download digitized land record report as PDF."""
    try:
        from backend.app.services.export import build_pdf_export
    except ImportError:
        from app.services.export import build_pdf_export
    payload = _get_document_export_payload(document_id, db)
    pdf_bytes = build_pdf_export(**payload)
    safe_name = payload["filename"].rsplit(".", 1)[0] if "." in payload["filename"] else payload["filename"]
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_digitized.pdf"'},
    )


@router.get(
    "/{document_id}/export/docx",
    summary="Download Digitized Document as Word Document",
    description="Generates a downloadable Word DOCX document. Allowed roles: ALL USERS.",
)
def export_document_docx(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
):
    """Download digitized land record report as Word DOCX."""
    try:
        from backend.app.services.export import build_docx_export
    except ImportError:
        from app.services.export import build_docx_export
    payload = _get_document_export_payload(document_id, db)
    docx_bytes = build_docx_export(**payload)
    safe_name = payload["filename"].rsplit(".", 1)[0] if "." in payload["filename"] else payload["filename"]
    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_digitized.docx"'},
    )


@router.get(
    "/{document_id}/export/kannada",
    summary="Download Recognized Kannada Text",
    description="Exports the recognized Kannada text as a text file. Allowed roles: ALL USERS.",
)
def export_document_kannada(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
):
    """Download recognized Kannada transcription text."""
    payload = _get_document_export_payload(document_id, db)
    kn_text = payload["kannada_text"] or "No Kannada script extracted."
    safe_name = payload["filename"].rsplit(".", 1)[0] if "." in payload["filename"] else payload["filename"]
    return StreamingResponse(
        io.BytesIO(kn_text.encode("utf-8")),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_kannada.txt"'},
    )


@router.get(
    "/{document_id}/export/english",
    summary="Download English Translation",
    description="Exports the translated English text as a text file. Allowed roles: ALL USERS.",
)
def export_document_english(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
):
    """Download translated English text."""
    payload = _get_document_export_payload(document_id, db)
    en_text = payload["english_translation"] or "No English translation available."
    safe_name = payload["filename"].rsplit(".", 1)[0] if "." in payload["filename"] else payload["filename"]
    return StreamingResponse(
        io.BytesIO(en_text.encode("utf-8")),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_english.txt"'},
    )


@router.get(
    "/{document_id}/presigned-url",
    response_model=PresignedUrlResponse,
    summary="Get Presigned MinIO URL",
    description="Generates a temporary presigned URL. Allowed roles: ALL USERS.",
)
def get_document_presigned_url(
    document_id: int,
    expires_in_seconds: int = Query(default=3600, ge=60, le=86400, description="Expiration time in seconds"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
) -> PresignedUrlResponse:
    """Generate a temporary presigned MinIO download URL."""
    stmt = select(Document).where(Document.id == document_id)
    document = db.execute(stmt).scalar_one_or_none()
    if not document or not document.storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found",
        )

    try:
        url = minio_storage.get_presigned_download_url(
            object_name=document.storage_path,
            expires_seconds=expires_in_seconds,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating presigned URL: {str(e)}",
        )

    return PresignedUrlResponse(
        download_url=url,
        expires_in_seconds=expires_in_seconds,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Document (ADMIN Only)",
    description="Delete a document record from database and storage. Allowed roles: ADMIN only.",
)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_admin),
):
    """Delete document and its stored object (ADMIN only)."""
    success = DocumentService.delete_document_record(document_id=document_id, db=db)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found",
        )
    return None


class TranslationRequest(BaseModel):
    text: str
    source_lang: Optional[str] = "auto"
    target_lang: Optional[str] = "en"


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str


@router.post(
    "/translate",
    response_model=TranslationResponse,
    summary="Interactive Bidirectional Land Record Translation",
    description="Translates land record text bidirectionally between Kannada and English preserving cadastral tokens.",
)
def translate_land_record_text(
    payload: TranslationRequest,
):
    """Translate arbitrary land record text between English and Kannada."""
    t_res = TranslationService.translate(
        text=payload.text,
        source_lang=payload.source_lang or "auto",
        target_lang=payload.target_lang or "en",
    )
    return TranslationResponse(
        original_text=payload.text,
        translated_text=t_res.translated_text or "",
        source_lang=payload.source_lang or "auto",
        target_lang=payload.target_lang or "en",
    )


class ReviewCorrectionRequest(BaseModel):
    review_id: str
    decision: str = "CORRECTED"  # ACCEPTED, CORRECTED, REJECTED
    corrected_text: Optional[str] = None
    reviewer_notes: Optional[str] = None
    reviewed_by: Optional[str] = "reviewer_officer"


@router.post(
    "/{document_id}/review",
    summary="Submit Human Review Decision",
    description="Updates review queue items, persisting corrected transcription while preserving immutable raw OCR evidence.",
)
def submit_document_review(
    document_id: int,
    payload: ReviewCorrectionRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(require_viewer_or_above),
):
    """Submits human verification/correction for a review item and persists across restarts."""
    from datetime import datetime, timezone

    stmt = select(ExtractionResult).where(ExtractionResult.document_id == document_id)
    ext = db.execute(stmt).scalar_one_or_none()
    if not ext or not ext.extracted_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Extraction data for document {document_id} not found",
        )

    ext_data = dict(ext.extracted_data)
    review_items = ext_data.get("review_items", [])
    target_item = None
    item_idx = -1

    for idx, item in enumerate(review_items):
        if item.get("review_id") == payload.review_id:
            target_item = dict(item)
            item_idx = idx
            break

    if not target_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review item '{payload.review_id}' not found in document {document_id}",
        )

    # Immutable Raw OCR Evidence check: raw_ocr_text must NEVER be mutated
    raw_ocr = target_item.get("raw_ocr_text", "")
    reviewer = payload.reviewed_by or (current_user.email if current_user else "reviewer_officer")
    timestamp = datetime.now(timezone.utc).isoformat()

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

    # Update in review_items list
    review_items[item_idx] = target_item
    ext_data["review_items"] = review_items

    # Check if any remaining pending items require review
    still_pending = any(
        it.get("lifecycle_state") in ("PENDING", "ASSIGNED", "IN_REVIEW")
        or it.get("status") == "REVIEW_REQUIRED"
        for it in review_items
    )
    if not still_pending:
        val_info = dict(ext.validation_info or {})
        val_info["requires_human_review"] = False
        ext.validation_info = val_info
        ext.is_valid = True

    ext.extracted_data = ext_data
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(ext, "extracted_data")
    flag_modified(ext, "validation_info")
    db.commit()
    db.refresh(ext)

    return {
        "status": "success",
        "review_id": payload.review_id,
        "document_id": document_id,
        "item": target_item,
        "requires_human_review": still_pending,
    }


