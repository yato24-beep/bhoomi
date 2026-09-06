"""
Unit tests for Active Learning Service and Promotion Gate logic.
"""

import shutil
from pathlib import Path
import pytest
from schemas import BoundingBox
from src.confidence.active_learning import ActiveLearningService


@pytest.fixture
def al_service(tmp_path):
    return ActiveLearningService(data_dir=str(tmp_path / "corrections"))


def test_log_correction_and_export(al_service):
    bbox = BoundingBox(x_min=10, y_min=20, x_max=100, y_max=50)
    record = al_service.log_human_correction(
        document_id="DOC_CORR_01",
        field_name="owner_name",
        original_prediction="राम",
        corrected_value="राम प्रसाद",
        page_number=1,
        bbox=bbox,
        model_version="v1.0.0",
        corrected_by="annotator_alice",
    )

    assert record["document_id"] == "DOC_CORR_01"
    assert record["original_prediction"] == "राम"
    assert record["corrected_value"] == "राम प्रसाद"

    batch_path = al_service.export_dataset_batch(batch_id="test_batch_01")
    assert batch_path.is_file()


def test_promotion_gate_decisions(al_service):
    # Case 1: CER decreased and accuracy improved -> Approved
    promoted, msg = al_service.evaluate_promotion_gate(
        baseline_cer=0.1200,
        new_model_cer=0.0800,
        baseline_field_acc=88.0,
        new_model_field_acc=94.5,
    )
    assert promoted is True
    assert "PROMOTION APPROVED" in msg

    # Case 2: CER increased -> Rejected
    promoted, msg = al_service.evaluate_promotion_gate(
        baseline_cer=0.0800,
        new_model_cer=0.0950,
        baseline_field_acc=92.0,
        new_model_field_acc=90.0,
    )
    assert promoted is False
    assert "PROMOTION REJECTED" in msg
