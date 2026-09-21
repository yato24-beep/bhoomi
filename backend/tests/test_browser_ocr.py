from app.models.document import Document
from app.models.extraction import ExtractionResult
from app.models.extracted_field import ExtractedField


def test_browser_ocr_data_contract_flow(client, officer_headers):
    """Verify Stage 4, 5, 6, and 7 data contract for BrowserTrOCR Kannada handwritten results."""
    unique_hash = "f1e2d3c4b5a60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef"
    payload = {
        "filename": "handwritten_kannada_test.png",
        "file_hash": unique_hash,
        "file_size": 1024,
        "text": "ಬೌದ್ಯೆ",
        "confidence": 0.7853,
        "execution_provider": "wasm",
        "latency_ms": 450,
        "tokens": [0, 0, 298, 369, 269, 264, 278, 270],
    }

    # Stage 4 & 5: POST to /api/v1/documents/browser-result
    res = client.post("/api/v1/documents/browser-result", json=payload, headers=officer_headers)
    assert res.status_code == 201
    res_data = res.json()
    assert res_data["document"]["status"] == "COMPLETED"
    doc_id = res_data["document"]["id"]

    # Stage 6: GET /api/v1/documents/{id}/results
    results_res = client.get(f"/api/v1/documents/{doc_id}/results", headers=officer_headers)
    assert results_res.status_code == 200
    results_data = results_res.json()
    extracted_data = results_data["extracted_data"]

    # Stage 7 Verification: Canonical field names contract
    assert extracted_data.get("original_ocr") == "ಬೌದ್ಯೆ"
    assert extracted_data.get("clean_kannada_text") == "ಬೌದ್ಯೆ"
    assert extracted_data.get("recognition_confidence") == 0.7853

    # Frontend display expression simulation
    raw_kannada = (extracted_data.get("original_ocr") or extracted_data.get("original_kannada_text") or "").strip()
    assert raw_kannada == "ಬೌದ್ಯೆ"
    assert raw_kannada != "No raw Kannada OCR output detected."

    clean_kannada = (extracted_data.get("clean_kannada_text") or extracted_data.get("original_kannada_text") or "").strip()
    assert clean_kannada == "ಬೌದ್ಯೆ"
    assert clean_kannada != "No Kannada text available."

    # Extracted fields query
    fields_res = client.get(f"/api/v1/documents/{doc_id}/fields", headers=officer_headers)
    assert fields_res.status_code == 200
    fields_data = fields_res.json()
    assert fields_data["total_fields"] >= 1
    hw_field = next((f for f in fields_data["fields"] if f["field_name"] == "handwritten_kannada_text"), None)
    assert hw_field is not None
    assert hw_field["original_value"] == "ಬೌದ್ಯೆ"
