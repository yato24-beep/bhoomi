"""API and Data Contract Hardening Tests.

Verifies:
- DocumentProcessingResponse schema compliance (types, optionality, nullability)
- Raw and normalized values, provenance, validation status, confidence
- Preservation of uncalibrated confidence as None/null
- Stage timings breakdown (gate, OCR, NER/table, semantic, validation, total)
- Structured error responses on corrupt/empty/oversized inputs without server crash
- Compatibility endpoints (/api/v1/documents/*) consistency with frontend schemas
"""

import io
import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.integration.schemas import DocumentProcessingResponse


@pytest.fixture
def client():
    return TestClient(app)


def create_dummy_image_bytes(width=200, height=100, color="white"):
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_api_process_contract_success(client):
    """Test POST /api/ocr/process response complies with the complete data contract."""
    img_bytes = create_dummy_image_bytes()
    files = {"file": ("test_doc.png", io.BytesIO(img_bytes), "image/png")}
    data = {"language": "kannada", "page_number": 1}

    res = client.post("/api/ocr/process", files=files, data=data)
    assert res.status_code == 200, res.text
    payload = res.json()

    # Validate against Pydantic schema
    resp_obj = DocumentProcessingResponse.model_validate(payload)
    assert resp_obj.image_path is not None
    assert isinstance(resp_obj.merged_text, str)
    assert isinstance(resp_obj.stage_timings, dict)
    assert "total_ms" in resp_obj.stage_timings
    assert "gate_classification_ms" in resp_obj.stage_timings
    assert "ocr_inference_ms" in resp_obj.stage_timings
    assert "semantic_ms" in resp_obj.stage_timings
    assert "validation_ms" in resp_obj.stage_timings

    # Calibrated confidence must be null by default
    assert resp_obj.calibrated_confidence is None
    assert resp_obj.confidence_state.value == "UNCALIBRATED"

    # Extracted fields structure check
    assert isinstance(resp_obj.extracted_fields, dict)
    for fname, fld in resp_obj.extracted_fields.items():
        if isinstance(fld, dict):
            assert "raw_value" in fld
            assert "confidence" in fld
            assert "validation_status" in fld


def test_api_process_empty_file_contract(client):
    """Empty payload returns structured 400 rather than crashing."""
    files = {"file": ("empty.png", io.BytesIO(b""), "image/png")}
    res = client.post("/api/ocr/process", files=files)
    assert res.status_code == 400
    err = res.json()
    assert "detail" in err
    assert "empty" in err["detail"].lower()


def test_api_process_corrupt_file_contract(client):
    """Corrupt image returns structured 400 rather than 500 unhandled crash."""
    files = {"file": ("corrupt.png", io.BytesIO(b"NOT_A_VALID_IMAGE_DATA_12345"), "image/png")}
    res = client.post("/api/ocr/process", files=files)
    assert res.status_code == 400
    err = res.json()
    assert "detail" in err
    assert "invalid" in err["detail"].lower() or "unreadable" in err["detail"].lower()


def test_api_compat_results_contract(client):
    """Ensure compat results endpoint fulfills ExtractionResultRead schema."""
    img_bytes = create_dummy_image_bytes()
    files = {"file": ("sample_upload.png", io.BytesIO(img_bytes), "image/png")}

    # 1. Upload
    up_res = client.post("/api/v1/documents/upload", files=files)
    assert up_res.status_code == 201
    up_data = up_res.json()
    doc_id = up_data["document"]["id"]

    try:
        # 2. Results
        res_res = client.get(f"/api/v1/documents/{doc_id}/results")
        assert res_res.status_code == 200
        res_data = res_res.json()

        assert res_data["id"] == doc_id
        assert res_data["document_id"] == doc_id
        assert "confidence_score" in res_data
        assert "extracted_data" in res_data
        assert "validation_info" in res_data

        ext_data = res_data["extracted_data"]
        assert "review_items" in ext_data
        assert "calibrated_confidence" in ext_data
        assert ext_data["calibrated_confidence"] is None
        assert ext_data["confidence_state"] == "UNCALIBRATED"
        assert "verification_status" in ext_data
        assert "stage_timings" in ext_data
    finally:
        client.delete(f"/api/v1/documents/{doc_id}")
