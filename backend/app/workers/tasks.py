import io
import time
import logging
from sqlalchemy import select, delete

from app.workers.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.document import Document
from app.models.extraction import ExtractionResult
from app.models.extracted_field import ExtractedField
from app.services.minio_storage import minio_storage
from app.pipeline.processor import get_document_processor

logger = logging.getLogger(__name__)


@celery_app.task(name="process_document_task", bind=True, max_retries=2)
def process_document_task(self, document_id: int):
    """Celery background worker task to retrieve document from MinIO, execute extraction,

    and persist both summary results and granular extracted fields in PostgreSQL.
    """
    logger.info(f"📥 [Task {self.request.id}] Starting processing job for Document ID {document_id}")
    
    db = SessionLocal()
    try:
        # 1. Fetch document metadata from PostgreSQL
        stmt = select(Document).where(Document.id == document_id)
        document = db.execute(stmt).scalar_one_or_none()

        if not document:
            logger.error(f"❌ Document ID {document_id} not found in database.")
            return {"status": "error", "message": f"Document ID {document_id} not found"}

        # 2. Update document status to PROCESSING
        logger.info(f"🔄 Updating Document {document_id} status: {document.status} -> PROCESSING")
        document.status = "PROCESSING"
        db.commit()
        db.refresh(document)

        # 3. Retrieve raw file stream from MinIO object storage
        logger.info(f"📦 Retrieving '{document.storage_path}' from MinIO...")
        minio_obj = minio_storage.get_file_object(document.storage_path)

        # 4. Invoke document processing pipeline
        processor = get_document_processor()
        logger.info(f"⚙️ Running extraction pipeline ({processor.__class__.__name__}) on '{document.filename}'...")
        
        pipeline_result = processor.process(
            file_stream=minio_obj,
            filename=document.filename,
        )

        # 5. Persist aggregate ExtractionResult in PostgreSQL
        res_stmt = select(ExtractionResult).where(ExtractionResult.document_id == document_id)
        extraction_record = db.execute(res_stmt).scalar_one_or_none()

        if not extraction_record:
            extraction_record = ExtractionResult(
                document_id=document_id,
                extracted_data=pipeline_result.extracted_data,
                confidence_score=pipeline_result.confidence_score,
                is_valid=pipeline_result.is_valid,
                validation_info=pipeline_result.validation_info,
                processing_time_ms=pipeline_result.processing_time_ms,
            )
            db.add(extraction_record)
        else:
            extraction_record.extracted_data = pipeline_result.extracted_data
            extraction_record.confidence_score = pipeline_result.confidence_score
            extraction_record.is_valid = pipeline_result.is_valid
            extraction_record.validation_info = pipeline_result.validation_info
            extraction_record.processing_time_ms = pipeline_result.processing_time_ms

        # 6. Persist granular ExtractedField rows in PostgreSQL (upsert/replace)
        db.execute(delete(ExtractedField).where(ExtractedField.document_id == document_id))
        
        for f in pipeline_result.fields:
            field_record = ExtractedField(
                document_id=document_id,
                field_name=f.field_name,
                original_value=f.original_value,
                normalized_value=f.normalized_value,
                confidence_score=f.confidence_score,
                source_page=f.source_page,
                bounding_box=f.bounding_box.model_dump() if f.bounding_box else None,
            )
            db.add(field_record)

        # 7. Update document status to COMPLETED
        document.status = "COMPLETED"
        db.commit()
        db.refresh(document)

        logger.info(
            f"✅ Document ID {document_id} processed: saved {len(pipeline_result.fields)} extracted fields "
            f"in {pipeline_result.processing_time_ms}ms (Avg Confidence: {pipeline_result.confidence_score:.2f})."
        )

        return {
            "status": "success",
            "document_id": document_id,
            "filename": document.filename,
            "fields_extracted": len(pipeline_result.fields),
            "confidence_score": pipeline_result.confidence_score,
            "is_valid": pipeline_result.is_valid,
            "final_status": "COMPLETED",
        }

    except Exception as exc:
        logger.error(f"🔥 Error processing Document ID {document_id}: {exc}", exc_info=True)
        try:
            db.rollback()
            stmt = select(Document).where(Document.id == document_id)
            doc = db.execute(stmt).scalar_one_or_none()
            if doc:
                doc.status = "FAILED"
                db.commit()
        except Exception as db_exc:
            logger.error(f"Failed to record FAILED status in DB: {db_exc}")
        
        raise self.retry(exc=exc, countdown=5)
    finally:
        db.close()
