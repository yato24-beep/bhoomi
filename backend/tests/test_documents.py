import hashlib


def test_create_and_get_document(client, admin_headers):
    """Test creating a document in the database and retrieving it by ID."""
    sample_content = b"Invoice #10293 for ACME Corp"
    file_hash = hashlib.sha256(sample_content).hexdigest()

    payload = {
        "filename": "invoice_10293.pdf",
        "file_hash": file_hash,
        "status": "UPLOADED",
        "storage_path": "documents/invoice_10293.pdf",
    }

    # 1. Create document
    create_res = client.post("/api/v1/documents/", json=payload, headers=admin_headers)
    assert create_res.status_code == 201
    created_data = create_res.json()
    assert created_data["id"] is not None
    assert created_data["filename"] == "invoice_10293.pdf"
    assert created_data["file_hash"] == file_hash
    assert created_data["status"] == "UPLOADED"
    assert created_data["storage_path"] == "documents/invoice_10293.pdf"
    assert "created_at" in created_data

    doc_id = created_data["id"]

    # 2. Get document by ID
    get_res = client.get(f"/api/v1/documents/{doc_id}", headers=admin_headers)
    assert get_res.status_code == 200
    doc_data = get_res.json()
    assert doc_data["id"] == doc_id
    assert doc_data["filename"] == "invoice_10293.pdf"


def test_list_documents(client, admin_headers):
    """Test listing documents with pagination."""
    for i in range(3):
        client.post(
            "/api/v1/documents/",
            json={
                "filename": f"doc_{i}.pdf",
                "file_hash": f"{i}" * 64,
                "status": "UPLOADED",
                "storage_path": f"documents/doc_{i}.pdf",
            },
            headers=admin_headers,
        )

    list_res = client.get("/api/v1/documents/?limit=10", headers=admin_headers)
    assert list_res.status_code == 200
    docs = list_res.json()
    assert len(docs) >= 3


def test_get_nonexistent_document(client, admin_headers):
    """Test retrieving a document that does not exist returns 404."""
    response = client.get("/api/v1/documents/999999", headers=admin_headers)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_delete_document(client, admin_headers):
    """Test deleting a document as ADMIN."""
    create_res = client.post(
        "/api/v1/documents/",
        json={
            "filename": "to_delete.pdf",
            "file_hash": "a" * 64,
            "status": "UPLOADED",
            "storage_path": "documents/to_delete.pdf",
        },
        headers=admin_headers,
    )
    doc_id = create_res.json()["id"]

    # Delete
    del_res = client.delete(f"/api/v1/documents/{doc_id}", headers=admin_headers)
    assert del_res.status_code == 204

    # Verify 404 on subsequent get
    get_res = client.get(f"/api/v1/documents/{doc_id}", headers=admin_headers)
    assert get_res.status_code == 404
