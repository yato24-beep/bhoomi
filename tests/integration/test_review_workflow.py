"""Tests for Human Review Workflow in both Direct/Compat mode and State Machine.

Verifies:
- Retrieval of review items
- Acceptance of OCR text (preserves raw OCR immutability)
- Correction/editing of transcription (updates normalized field while raw OCR remains immutable)
- Rejection of invalid OCR
- Clearing requires_human_review flag once all items are resolved
- Error handling on invalid decision or missing item
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api.main import app
from src.integration.schemas import DocumentProcessingResponse, RecognizedRegionResult
from src.integration.review_state import HumanReviewItem, ReviewLifecycleState, ReviewStatus, ReviewDecision
import src.api.compat as compat_module


@pytest.fixture
def client():
    return TestClient(app)


def test_review_workflow_accept_edit_reject(client):
    """Test complete human review flow: accept, edit, reject, and resolution."""
    # Setup mock document in compat storage
    doc_id = 9999
    now = compat_module.datetime.utcnow()
    compat_module._DOCUMENTS[doc_id] = {
        "id": doc_id,
        "filename": "karnataka_rtc_sample.jpg",
        "file_hash": "hash_9999",
        "status": "COMPLETED",
        "storage_path": "local://documents/9999/karnataka_rtc_sample.jpg",
        "created_at": now,
    }

    # Create mock response with 2 review items
    mock_resp = DocumentProcessingResponse(
        document_id=f"doc_{doc_id}",
        page_number=1,
        image_path="karnataka_rtc_sample.jpg",
        merged_text="Survey 142/A Owner Ramesh",
        requires_human_review=True,
        extracted_fields={
            "survey_number": {
                "raw_value": "142/?",
                "normalized_value": "142/?",
                "confidence": 0.55,
                "validation_status": "INVALID",
            },
            "owner_name": {
                "raw_value": "Ramesh",
                "normalized_value": "Ramesh",
                "confidence": 0.62,
                "validation_status": "WARNING",
            }
        },
        review_items=[
            {
                "review_id": "rev_item_survey",
                "document_id": f"doc_{doc_id}",
                "page_number": 1,
                "region_id": "region_survey",
                "raw_ocr_text": "142/?",
                "recognizer": "paddleocr",
                "review_reason": "Low confidence or invalid survey number format",
                "status": "REVIEW_REQUIRED",
                "lifecycle_state": "PENDING",
                "metadata": {
                    "field_name": "survey_number",
                    "raw_value": "142/?",
                    "normalized_value": "142/?",
                },
            },
            {
                "review_id": "rev_item_owner",
                "document_id": f"doc_{doc_id}",
                "page_number": 1,
                "region_id": "region_owner",
                "raw_ocr_text": "Ramesh",
                "recognizer": "paddleocr",
                "review_reason": "Low confidence name field",
                "status": "REVIEW_REQUIRED",
                "lifecycle_state": "PENDING",
                "metadata": {
                    "field_name": "owner_name",
                    "raw_value": "Ramesh",
                    "normalized_value": "Ramesh",
                },
            }
        ],
    )
    compat_module._DOCUMENT_RESULTS[doc_id] = mock_resp

    try:
        # 1. GET review items
        get_res = client.get(f"/api/v1/documents/{doc_id}/review/items")
        assert get_res.status_code == 200
        items = get_res.json()
        assert len(items) == 2
        assert items[0]["review_id"] == "rev_item_survey"

        # 2. Correct survey_number
        edit_payload = {
            "review_id": "rev_item_survey",
            "decision": "CORRECTED",
            "corrected_text": "142/A",
            "reviewer_notes": "Corrected punctuation from scanned original",
            "reviewed_by": "officer_sharma",
        }
        post_edit = client.post(f"/api/v1/documents/{doc_id}/review", json=edit_payload)
        assert post_edit.status_code == 200
        edit_data = post_edit.json()
        assert edit_data["status"] == "success"
        assert edit_data["requires_human_review"] is True  # Owner is still pending!

        item_survey = edit_data["item"]
        assert item_survey["raw_ocr_text"] == "142/?"  # Raw OCR strictly immutable!
        assert item_survey["corrected_text"] == "142/A"
        assert item_survey["decision"] == "CORRECTED"
        assert item_survey["lifecycle_state"] == "EDITED"
        assert item_survey["reviewed_by"] == "officer_sharma"

        # Verify extracted_fields updated
        assert mock_resp.extracted_fields["survey_number"]["normalized_value"] == "142/A"
        assert mock_resp.extracted_fields["survey_number"]["validation_status"] == "VALID"

        # 3. Accept owner_name
        accept_payload = {
            "review_id": "rev_item_owner",
            "decision": "ACCEPTED",
            "reviewer_notes": "Verified name matches registry",
            "reviewed_by": "officer_sharma",
        }
        post_accept = client.post(f"/api/v1/documents/{doc_id}/review", json=accept_payload)
        assert post_accept.status_code == 200
        accept_data = post_accept.json()
        assert accept_data["status"] == "success"
        # All items are now resolved!
        assert accept_data["requires_human_review"] is False
        assert mock_resp.requires_human_review is False
        assert mock_resp.verification_status == "accepted"

        # 4. Check results endpoint reflects resolved state
        results_res = client.get(f"/api/v1/documents/{doc_id}/results")
        assert results_res.status_code == 200
        res_json = results_res.json()
        assert res_json["extracted_data"]["verification_status"] == "accepted"
        assert len(res_json["extracted_data"]["review_items"]) == 2

    finally:
        # Cleanup
        compat_module._DOCUMENTS.pop(doc_id, None)
        compat_module._DOCUMENT_RESULTS.pop(doc_id, None)


def test_review_workflow_rejection(client):
    """Test rejecting a hallucinated/corrupt region in review."""
    doc_id = 9998
    now = compat_module.datetime.utcnow()
    compat_module._DOCUMENTS[doc_id] = {
        "id": doc_id,
        "filename": "noisy_doc.jpg",
        "file_hash": "hash_9998",
        "status": "COMPLETED",
        "storage_path": "local://documents/9998/noisy_doc.jpg",
        "created_at": now,
    }

    mock_resp = DocumentProcessingResponse(
        document_id=f"doc_{doc_id}",
        page_number=1,
        image_path="noisy_doc.jpg",
        merged_text="Smudge artifact",
        requires_human_review=True,
        review_items=[
            {
                "review_id": "rev_noise",
                "document_id": f"doc_{doc_id}",
                "page_number": 1,
                "region_id": "region_noise",
                "raw_ocr_text": "###@@",
                "recognizer": "paddleocr",
                "review_reason": "Smudge or illegible text",
                "status": "REVIEW_REQUIRED",
                "lifecycle_state": "PENDING",
            }
        ],
    )
    compat_module._DOCUMENT_RESULTS[doc_id] = mock_resp

    try:
        reject_payload = {
            "review_id": "rev_noise",
            "decision": "REJECTED",
            "reviewer_notes": "Ink smudge, not real text",
        }
        post_reject = client.post(f"/api/v1/documents/{doc_id}/review", json=reject_payload)
        assert post_reject.status_code == 200
        data = post_reject.json()
        assert data["item"]["decision"] == "REJECTED"
        assert data["item"]["status"] == "REJECTED_FAILED"
        assert data["item"]["lifecycle_state"] == "REJECTED"
        assert data["item"]["raw_ocr_text"] == "###@@"  # Immutable
        assert data["requires_human_review"] is False

    finally:
        compat_module._DOCUMENTS.pop(doc_id, None)
        compat_module._DOCUMENT_RESULTS.pop(doc_id, None)


def test_review_workflow_invalid_payload(client):
    """Test validation errors for invalid review inputs."""
    doc_id = 9997
    compat_module._DOCUMENTS[doc_id] = {
        "id": doc_id,
        "filename": "sample.jpg",
        "file_hash": "hash_9997",
        "status": "COMPLETED",
        "storage_path": "local://documents/9997/sample.jpg",
        "created_at": compat_module.datetime.utcnow(),
    }
    compat_module._DOCUMENT_RESULTS[doc_id] = DocumentProcessingResponse(
        document_id=f"doc_{doc_id}",
        page_number=1,
        image_path="sample.jpg",
        merged_text="Sample text",
        review_items=[
            {
                "review_id": "rev_item_1",
                "document_id": f"doc_{doc_id}",
                "page_number": 1,
                "region_id": "r1",
                "raw_ocr_text": "text",
                "recognizer": "paddleocr",
                "review_reason": "flag",
                "status": "REVIEW_REQUIRED",
                "lifecycle_state": "PENDING",
            }
        ]
    )

    try:
        # Invalid decision type
        bad_decision = client.post(
            f"/api/v1/documents/{doc_id}/review",
            json={"review_id": "rev_item_1", "decision": "UNKNOWN_ACTION"}
        )
        assert bad_decision.status_code == 400
        assert "Invalid decision" in bad_decision.json()["detail"]

        # Non-existent review ID
        bad_id = client.post(
            f"/api/v1/documents/{doc_id}/review",
            json={"review_id": "non_existent", "decision": "ACCEPTED"}
        )
        assert bad_id.status_code == 404

        # Non-existent doc ID
        bad_doc = client.post(
            "/api/v1/documents/123456/review",
            json={"review_id": "rev_item_1", "decision": "ACCEPTED"}
        )
        assert bad_doc.status_code == 404
    finally:
        compat_module._DOCUMENTS.pop(doc_id, None)
        compat_module._DOCUMENT_RESULTS.pop(doc_id, None)
