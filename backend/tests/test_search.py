import io


def test_search_by_filename(client, officer_headers):
    """Test searching documents by matching filename."""
    files = {"file": ("q4_financial_report.pdf", io.BytesIO(b"%PDF-1.4 financial"), "application/pdf")}
    upload_res = client.post("/api/v1/documents/upload", files=files, headers=officer_headers)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]

    # Search for 'financial'
    search_res = client.get("/api/v1/documents/search?q=financial", headers=officer_headers)
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["query"] == "financial"
    assert data["total_results"] >= 1
    matching_ids = [doc["id"] for doc in data["results"]]
    assert doc_id in matching_ids


def test_search_by_extracted_vendor_name(client, officer_headers):
    """Test searching documents by extracted field vendor name ('Acme')."""
    files = {"file": ("acme_consulting_bill.pdf", io.BytesIO(b"%PDF-1.4 invoice for acme"), "application/pdf")}
    upload_res = client.post("/api/v1/documents/upload", files=files, headers=officer_headers)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]

    # Search for 'Acme'
    search_res = client.get("/api/v1/documents/search?q=Acme", headers=officer_headers)
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["total_results"] >= 1
    
    target_doc = next((d for d in data["results"] if d["id"] == doc_id), None)
    assert target_doc is not None
    assert len(target_doc["matched_fields"]) > 0
    # One of the matched fields should be vendor_name
    field_names = [f["field_name"] for f in target_doc["matched_fields"]]
    assert "vendor_name" in field_names


def test_search_by_extracted_amount(client, officer_headers):
    """Test searching documents by extracted financial total ('1350')."""
    files = {"file": ("service_charge_2026.pdf", io.BytesIO(b"%PDF-1.4 charges"), "application/pdf")}
    upload_res = client.post("/api/v1/documents/upload", files=files, headers=officer_headers)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]

    # Search for '1350'
    search_res = client.get("/api/v1/documents/search?q=1350", headers=officer_headers)
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["total_results"] >= 1
    
    target_doc = next((d for d in data["results"] if d["id"] == doc_id), None)
    assert target_doc is not None
    assert any("total_amount" == f["field_name"] for f in target_doc["matched_fields"])


def test_search_with_field_name_filter(client, officer_headers):
    """Test searching with specific field_name filter (e.g. field_name=invoice_number)."""
    files = {"file": ("supplier_invoice.pdf", io.BytesIO(b"%PDF-1.4 invoice"), "application/pdf")}
    upload_res = client.post("/api/v1/documents/upload", files=files, headers=officer_headers)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]

    # Filter with field_name=invoice_number
    search_res = client.get(
        "/api/v1/documents/search?q=INV-2026-0892&field_name=invoice_number",
        headers=officer_headers,
    )
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["total_results"] >= 1
    target_doc = next((d for d in data["results"] if d["id"] == doc_id), None)
    assert target_doc is not None
    assert all(f["field_name"] == "invoice_number" for f in target_doc["matched_fields"])


def test_search_no_results(client, officer_headers):
    """Test searching with an unmatched term returns total_results: 0."""
    search_res = client.get("/api/v1/documents/search?q=completely_nonexistent_term_xyz_123", headers=officer_headers)
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["total_results"] == 0
    assert len(data["results"]) == 0
