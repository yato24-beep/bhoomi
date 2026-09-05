import io
import hashlib


def test_upload_pdf_to_minio_success(client, officer_headers):
    """Test successful upload of a PDF document to MinIO with background task dispatch."""
    pdf_content = b"%PDF-1.4 sample pdf content for minio testing"
    expected_hash = hashlib.sha256(pdf_content).hexdigest()

    files = {
        "file": ("invoice.pdf", io.BytesIO(pdf_content), "application/pdf")
    }

    response = client.post("/api/v1/documents/upload", files=files, headers=officer_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["message"] == "Document uploaded and processing job queued successfully."
    assert data["is_duplicate"] is False
    assert data["task_id"] is not None
    assert data["document"]["filename"] == "invoice.pdf"
    assert data["document"]["file_hash"] == expected_hash
    assert data["document"]["storage_path"].startswith("uploads/")


def test_upload_image_to_minio_success(client, officer_headers):
    """Test successful upload of a PNG image to MinIO with background task dispatch."""
    png_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR sample minio image data"
    expected_hash = hashlib.sha256(png_content).hexdigest()

    files = {
        "file": ("receipt.png", io.BytesIO(png_content), "image/png")
    }

    response = client.post("/api/v1/documents/upload", files=files, headers=officer_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["is_duplicate"] is False
    assert data["task_id"] is not None
    assert data["document"]["filename"] == "receipt.png"
    assert data["document"]["file_hash"] == expected_hash
    assert data["document"]["storage_path"].startswith("uploads/")


def test_upload_duplicate_detection(client, officer_headers):
    """Test duplicate detection via SHA-256 with MinIO backend."""
    file_content = b"%PDF-1.4 duplicate test content minio"

    # First upload
    files_1 = {"file": ("contract_v1.pdf", io.BytesIO(file_content), "application/pdf")}
    res_1 = client.post("/api/v1/documents/upload", files=files_1, headers=officer_headers)
    assert res_1.status_code == 201
    doc_1 = res_1.json()["document"]
    assert res_1.json()["is_duplicate"] is False
    assert res_1.json()["task_id"] is not None

    # Second upload with identical bytes
    files_2 = {"file": ("contract_copy.pdf", io.BytesIO(file_content), "application/pdf")}
    res_2 = client.post("/api/v1/documents/upload", files=files_2, headers=officer_headers)
    assert res_2.status_code == 201
    res_2_data = res_2.json()
    assert res_2_data["is_duplicate"] is True
    assert res_2_data["task_id"] is None
    assert "duplicate detected" in res_2_data["message"].lower()
    assert res_2_data["document"]["id"] == doc_1["id"]


def test_upload_unsupported_file_type(client, officer_headers):
    """Test that uploading unsupported extensions returns 400 Bad Request."""
    files = {
        "file": ("malicious.exe", io.BytesIO(b"binary data"), "application/x-msdownload")
    }
    response = client.post("/api/v1/documents/upload", files=files, headers=officer_headers)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_download_and_presigned_url_from_minio(client, officer_headers):
    """Test streaming download and generating presigned URLs for MinIO objects."""
    content = b"%PDF-1.4 test minio download content"
    files = {"file": ("download_me.pdf", io.BytesIO(content), "application/pdf")}
    upload_res = client.post("/api/v1/documents/upload", files=files, headers=officer_headers)
    doc_id = upload_res.json()["document"]["id"]

    # 1. Test streaming download
    download_res = client.get(f"/api/v1/documents/{doc_id}/download", headers=officer_headers)
    assert download_res.status_code == 200
    assert download_res.content == content

    # 2. Test presigned URL generation
    presigned_res = client.get(
        f"/api/v1/documents/{doc_id}/presigned-url?expires_in_seconds=1800",
        headers=officer_headers,
    )
    assert presigned_res.status_code == 200
    data = presigned_res.json()
    assert "download_url" in data
    assert "token=" in data["download_url"]
    assert data["expires_in_seconds"] == 1800
