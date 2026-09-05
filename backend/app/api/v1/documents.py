from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import select, or_, and_
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


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Document & Dispatch Processing Task",
    description="Upload a document to MinIO, record metadata, and dispatch Celery job. Allowed roles: ADMIN, OFFICER.",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (PDF, PNG, JPG, WEBP, TIFF)"),
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
        return DocumentUploadResponse(
            message="Document already exists (duplicate detected via SHA-256 hash).",
            is_duplicate=True,
            document=existing_document,
            task_id=None,
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
    try:
        task = process_document_task.delay(new_document.id)
        task_id = task.id
    except Exception:
        # Fallback to local sync task execution when Redis/Celery broker is offline
        try:
            process_document_task(new_document.id)
            task_id = "local-sync-worker"
        except Exception:
            pass

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
            detail=f"No extraction results found for document ID {document_id} (Status: {document.status})",
        )

    return result


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

    return StreamingResponse(
        minio_obj.stream(32 * 1024),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{document.filename}"'},
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
    stmt = select(Document).where(Document.id == document_id)
    document = db.execute(stmt).scalar_one_or_none()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found",
        )

    if document.storage_path:
        minio_storage.delete_file(document.storage_path)

    db.delete(document)
    db.commit()
    return None
