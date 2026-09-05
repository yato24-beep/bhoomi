import io


def test_get_extracted_fields_and_filtering(client, officer_headers):
    """Test retrieving granular extracted fields with coordinates and confidence filtering."""
    # 1. Upload sample invoice document
    pdf_content = b"%PDF-1.4 sample commercial invoice for field extraction test"
    files = {"file": ("enterprise_invoice.pdf", io.BytesIO(pdf_content), "application/pdf")}
    upload_res = client.post("/api/v1/documents/upload", files=files, headers=officer_headers)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]

    # 2. Retrieve all extracted fields
    fields_res = client.get(f"/api/v1/documents/{doc_id}/fields", headers=officer_headers)
    assert fields_res.status_code == 200
    data = fields_res.json()
    assert data["document_id"] == doc_id
    assert data["total_fields"] >= 8
    assert data["average_confidence"] >= 0.90
    assert len(data["fields"]) == data["total_fields"]

    # Check a specific field structure
    first_field = data["fields"][0]
    assert "field_name" in first_field
    assert "original_value" in first_field
    assert "normalized_value" in first_field
    assert "confidence_score" in first_field
    assert "source_page" in first_field
    assert "bounding_box" in first_field
    assert first_field["source_page"] == 1
    if first_field["bounding_box"]:
        bbox = first_field["bounding_box"]
        assert "x_min" in bbox
        assert "y_min" in bbox
        assert "x_max" in bbox
        assert "y_max" in bbox

    # 3. Filter by high confidence (>= 0.98)
    high_conf_res = client.get(f"/api/v1/documents/{doc_id}/fields?min_confidence=0.98", headers=officer_headers)
    assert high_conf_res.status_code == 200
    high_conf_data = high_conf_res.json()
    for field in high_conf_data["fields"]:
        assert field["confidence_score"] >= 0.98

    # 4. Fetch specific single field by name: total_amount
    total_field_res = client.get(f"/api/v1/documents/{doc_id}/fields/total_amount", headers=officer_headers)
    assert total_field_res.status_code == 200
    total_field = total_field_res.json()
    assert total_field["field_name"] == "total_amount"
    assert total_field["original_value"] == "$1,350.00"
    assert total_field["normalized_value"] == "1350.00"
    assert total_field["confidence_score"] >= 0.95
    assert total_field["bounding_box"]["unit"] == "normalized"

    # 5. Non-existent field returns 404
    missing_field_res = client.get(f"/api/v1/documents/{doc_id}/fields/non_existent_key", headers=officer_headers)
    assert missing_field_res.status_code == 404
    assert "not found" in missing_field_res.json()["detail"].lower()
