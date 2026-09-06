"""
Unit tests for Person C FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from src.database.db_session import init_db
from src.main import app

# Ensure tables are created for API tests
init_db()
client = TestClient(app)


def test_api_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "UP" in data["supported_states"]


def test_api_gis_validate():
    payload = {
        "state": "UP",
        "district": "LUCKNOW",
        "tehsil": "MOHANLALGANJ",
        "village": "MAU",
        "khasra_number": "142/1",
        "land_area_hectares": 0.4500,
    }
    response = client.post("/api/v1/gis/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_verified"] is True
    assert data["has_mismatch"] is False


def test_api_duplicate_check():
    payload = {
        "document_id": "DOC_API_01",
        "sha256_hash": "non_existent_hash_123",
        "full_text": "Sample text",
    }
    response = client.post("/api/v1/duplicates/check", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_duplicate"] is False


def test_api_submit_correction():
    payload = {
        "document_id": "DOC_API_02",
        "field_name": "owner_name",
        "original_prediction": "राम",
        "corrected_value": "राम प्रसाद",
        "page_number": 1,
        "model_version": "v1.0.0",
        "corrected_by": "tester",
    }
    response = client.post("/api/v1/corrections", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
