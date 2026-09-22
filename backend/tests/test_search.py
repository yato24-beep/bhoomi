import io


def test_search_by_filename(client, officer_headers):
    """Test searching documents by matching filename."""
    payload = {
        "filename": "karnataka_rtc_mandya_2026.png",
        "file_hash": "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0",
        "file_size": 2048,
        "text": "ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಕಂದಾಯ ಇಲಾಖೆ",
        "confidence": 0.92,
        "execution_provider": "webgpu",
        "latency_ms": 280,
    }
    upload_res = client.post("/api/v1/documents/browser-result", json=payload, headers=officer_headers)
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document"]["id"]

    # Search for 'mandya'
    search_res = client.get("/api/v1/documents/search?q=mandya", headers=officer_headers)
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["query"] == "mandya"
    assert data["total_results"] >= 1
    matching_ids = [doc["id"] for doc in data["results"]]
    assert doc_id in matching_ids


def test_search_by_extracted_owner_name(client, officer_headers):
    """Test searching documents by extracted land record owner name."""
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

    # Search for Kannada owner name 'ನಾಗರಾಜಯ್ಯ'
    search_res = client.get("/api/v1/documents/search?q=ನಾಗರಾಜಯ್ಯ", headers=officer_headers)
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["total_results"] >= 1
    
    target_doc = next((d for d in data["results"] if d["id"] == doc_id), None)
    assert target_doc is not None
    assert len(target_doc["matched_fields"]) > 0
    field_names = [f["field_name"] for f in target_doc["matched_fields"]]
    assert "owner_name" in field_names


def test_search_by_extracted_survey_number(client, officer_headers):
    """Test searching documents by extracted cadastral survey number ('142')."""
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

    # Search for '12/1'
    search_res = client.get("/api/v1/documents/search?q=12/1", headers=officer_headers)
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["total_results"] >= 1
    
    target_doc = next((d for d in data["results"] if d["id"] == doc_id), None)
    assert target_doc is not None
    assert any("survey_number" == f["field_name"] for f in target_doc["matched_fields"])


def test_search_with_field_name_filter(client, officer_headers):
    """Test searching with specific field_name filter (e.g. field_name=survey_number)."""
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

    # Filter with field_name=survey_number
    search_res = client.get(
        "/api/v1/documents/search?q=12/1&field_name=survey_number",
        headers=officer_headers,
    )
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["total_results"] >= 1
    target_doc = next((d for d in data["results"] if d["id"] == doc_id), None)
    assert target_doc is not None
    assert all(f["field_name"] == "survey_number" for f in target_doc["matched_fields"])


def test_search_no_results(client, officer_headers):
    """Test searching with an unmatched term returns total_results: 0."""
    search_res = client.get("/api/v1/documents/search?q=completely_nonexistent_term_xyz_123", headers=officer_headers)
    assert search_res.status_code == 200
    data = search_res.json()
    assert data["total_results"] == 0
    assert len(data["results"]) == 0

