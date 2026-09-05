import io
import hashlib
from app.workers.tasks import process_document_task
from app.models.document import Document
from app.models.extraction import ExtractionResult
import app.services.minio_storage
from sqlalchemy import select


def test_celery_task_status_lifecycle_and_results(client, officer_headers, db_session):
    """Verify Celery task processes document from MinIO and saves extraction results in PostgreSQL."""
    # 1. Upload document through API
    pdf_content = b"%PDF-1.4 celery test document"
    files = {"file": ("vendor_invoice.pdf", io.BytesIO(pdf_content), "application/pdf")}
    upload_res = client.post("/api/v1/documents/upload", files=files, headers=officer_headers)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]
    assert upload_res.json()["task_id"] is not None

    # In eager mode, the task completes synchronously during upload
    # 2. Verify status endpoint returns COMPLETED
    status_res = client.get(f"/api/v1/documents/{doc_id}/status", headers=officer_headers)
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "COMPLETED"

    # 3. Verify results endpoint returns structured extraction data
    results_res = client.get(f"/api/v1/documents/{doc_id}/results", headers=officer_headers)
    assert results_res.status_code == 200
    res_data = results_res.json()
    assert res_data["document_id"] == doc_id
    assert res_data["confidence_score"] >= 0.90
    assert res_data["is_valid"] is True
    assert "extracted_data" in res_data
    assert res_data["extracted_data"]["document_type"] == "Commercial Invoice"
    assert res_data["extracted_data"]["financials"]["total_amount"] == 1350.00
    assert len(res_data["validation_info"]["checks_passed"]) > 0


def test_celery_task_direct_execution(db_session):
    """Directly execute the Celery task function to verify database persistence."""
    # Upload mock file to patched minio storage
    mock_file_hash = "e" * 64
    storage_path = app.services.minio_storage.minio_storage.upload_file(
        file_stream=io.BytesIO(b"%PDF-1.4 direct task test"),
        filename="medical_receipt.png",
        file_hash=mock_file_hash,
        file_size=25,
        content_type="image/png",
    )

    # Create document manually in DB with UPLOADED status
    doc = Document(
        filename="medical_receipt.png",
        file_hash=mock_file_hash,
        status="UPLOADED",
        storage_path=storage_path,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    # Execute task
    result = process_document_task(doc.id)
    assert result["status"] == "success"
    assert result["document_id"] == doc.id
    assert result["final_status"] == "COMPLETED"
    assert result["is_valid"] is True

    # Verify state in DB
    db_session.refresh(doc)
    assert doc.status == "COMPLETED"

    # Verify ExtractionResult in DB
    stmt = select(ExtractionResult).where(ExtractionResult.document_id == doc.id)
    extraction = db_session.execute(stmt).scalar_one()
    assert extraction.document_id == doc.id
    assert extraction.extracted_data["document_type"] == "Receipt"
    assert extraction.confidence_score >= 0.90
