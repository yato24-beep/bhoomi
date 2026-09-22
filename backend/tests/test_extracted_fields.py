import io


def test_get_extracted_fields_and_filtering(client, officer_headers):
    """Test retrieving granular extracted fields with coordinates and confidence filtering."""
    # 1. Register a verified land record document with canonical fields
    payload = {
        "filename": "sample_bhoomi_rtc.png",
        "file_hash": "ced8979131bdf88f5759bcc62d2cd8047d4bca387da0f61b793dd76f5e6f86df",
        "file_size": 2048,
        "text": "ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಕಂದಾಯ ಇಲಾಖೆ",
        "confidence": 0.91,
        "execution_provider": "webgpu",
        "latency_ms": 320,
    }
    upload_res = client.post("/api/v1/documents/browser-result", json=payload, headers=officer_headers)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]

    # 2. Retrieve all extracted fields
    fields_res = client.get(f"/api/v1/documents/{doc_id}/fields", headers=officer_headers)
    assert fields_res.status_code == 200
    data = fields_res.json()
    assert data["document_id"] == doc_id
    assert data["total_fields"] >= 8
    assert data["average_confidence"] >= 0.85
    assert len(data["fields"]) == data["total_fields"]

    # Check a specific field structure
    first_field = data["fields"][0]
    assert "field_name" in first_field
    assert "original_value" in first_field
    assert "normalized_value" in first_field
    assert "confidence_score" in first_field
    assert "source_page" in first_field
    assert first_field["source_page"] == 1

    # 3. Filter by high confidence (>= 0.90)
    high_conf_res = client.get(f"/api/v1/documents/{doc_id}/fields?min_confidence=0.90", headers=officer_headers)
    assert high_conf_res.status_code == 200
    high_conf_data = high_conf_res.json()
    for field in high_conf_data["fields"]:
        assert field["confidence_score"] >= 0.90

    # 4. Fetch specific single field by name: owner_name
    owner_field_res = client.get(f"/api/v1/documents/{doc_id}/fields/owner_name", headers=officer_headers)
    assert owner_field_res.status_code == 200
    owner_field = owner_field_res.json()
    assert owner_field["field_name"] == "owner_name"
    assert "ನಾಗರಾಜಯ್ಯ" in owner_field["original_value"]
    assert owner_field["confidence_score"] >= 0.80

    # 5. Non-existent field returns 404
    missing_field_res = client.get(f"/api/v1/documents/{doc_id}/fields/non_existent_key", headers=officer_headers)
    assert missing_field_res.status_code == 404
    assert "not found" in missing_field_res.json()["detail"].lower()
