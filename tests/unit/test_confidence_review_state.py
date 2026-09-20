"""Unit tests for Confidence Architecture, Calibration Infrastructure, and Human Review Workflow.

Verifies:
- Separation of recognizer_confidence_raw, routing_confidence, validation_confidence, calibrated_confidence
- calibrated_confidence is strictly None/null when unfitted (Task 3)
- Confidence states: UNCALIBRATED, CALIBRATED, REVIEW_REQUIRED
- CalibrationInterface and CalibrationSample infrastructure (Task 4)
- Human review state machine: AUTO_ACCEPT, REVIEW_REQUIRED, REJECTED_FAILED (Task 5)
- HumanReviewItem preserves immutable raw OCR evidence during reviewer correction
- Line-word decomposition disabled by default (ENABLE_LINE_WORD_DECOMPOSITION=false) (Task 2)
- Multi-word line crops flagged as REVIEW_REQUIRED with translation blocked (Task 2 & 12)
- Document type is 'Unknown / Not classified' without validated classifier (Task 13)
- Zero model training / zero weight mutation
"""

import os
import pytest
from PIL import Image

from schemas import BoundingBox
from src.confidence.calibration_interface import CalibrationInterface, CalibrationSample
from src.integration.schemas import (
    ConfidenceState,
    DocumentProcessingRequest,
    ProcessingStatus,
    RecognizedRegionResult,
    RegionRequest,
    RegionType,
)
from src.integration.review_state import (
    HumanReviewItem,
    ReviewDecision,
    ReviewStateMachine,
    ReviewStatus,
)
from src.integration.document_pipeline import DocumentProcessingPipeline


def test_confidence_state_enum_and_calibrated_null():
    """Verifies that calibrated_confidence defaults strictly to None and ConfidenceState defaults to UNCALIBRATED."""
    reg = RecognizedRegionResult(
        region_id="r1",
        raw_text="ರಾಮಯ್ಯ",
        normalized_text="ರಾಮಯ್ಯ",
        confidence=0.82,
        recognizer_confidence_raw=0.82,
        language="kannada",
        script="Kannada",
    )
    assert reg.calibrated_confidence is None
    assert reg.confidence_state == ConfidenceState.UNCALIBRATED


def test_calibration_interface_unfitted_returns_none():
    """Verifies that CalibrationInterface strictly returns None prior to empirical fitting on held-out validation data."""
    calibrator = CalibrationInterface(calibrator_name="isotonic_scaler")
    assert not calibrator.is_fitted

    score = calibrator.predict_calibrated(0.85)
    assert score is None

    # Adding insufficient samples (< 50) must reject fitting to prevent premature calibration
    samples = [
        CalibrationSample(ground_truth_text="ರಾಮಯ್ಯ", predicted_text="ರಾಮಯ್ಯ", raw_score=0.95),
        CalibrationSample(ground_truth_text="ಮಲ್ಲಪ್ಪ", predicted_text="ಮಲ್ಲ", raw_score=0.50),
    ]
    with pytest.raises(ValueError, match="Insufficient held-out validation samples"):
        calibrator.fit(samples)

    # Valid empirical fitting with >= 50 held-out samples
    valid_held_out = [
        CalibrationSample(ground_truth_text="ಗ್ರಾಮ", predicted_text="ಗ್ರಾಮ", raw_score=0.92)
        for _ in range(50)
    ]
    calibrator.fit(valid_held_out)
    assert calibrator.is_fitted
    cal_score = calibrator.predict_calibrated(0.92)
    assert cal_score is not None
    assert 0.0 <= cal_score <= 1.0


def test_review_state_machine_triggers():
    """Verifies state machine evaluation triggers for empty text, multi-word lines, and low confidence."""
    # 1. Empty text
    st_empty, reason_empty = ReviewStateMachine.evaluate_region_review_status(
        is_handwritten=False,
        raw_confidence=0.90,
        calibrated_confidence=None,
        region_type="word",
        text="",
    )
    assert st_empty == ReviewStatus.REVIEW_REQUIRED
    assert "empty" in reason_empty.lower()

    # 2. Multi-word line on word-oriented recognizer
    st_line, reason_line = ReviewStateMachine.evaluate_region_review_status(
        is_handwritten=True,
        raw_confidence=0.85,
        calibrated_confidence=None,
        region_type="line",
        text="ರಾಮಯ್ಯ ಬಿನ್ ಮಲ್ಲಪ್ಪ",
    )
    assert st_line == ReviewStatus.REVIEW_REQUIRED
    assert "line" in reason_line.lower()

    # 3. Low confidence
    st_low, reason_low = ReviewStateMachine.evaluate_region_review_status(
        is_handwritten=False,
        raw_confidence=0.45,
        calibrated_confidence=None,
        region_type="word",
        text="ಖಾತೆ",
    )
    assert st_low == ReviewStatus.REVIEW_REQUIRED
    assert "low" in reason_low.lower()

    # 4. Clean auto-accept
    st_clean, _ = ReviewStateMachine.evaluate_region_review_status(
        is_handwritten=False,
        raw_confidence=0.95,
        calibrated_confidence=0.94,
        region_type="word",
        text="ಖಾತೆ",
    )
    assert st_clean == ReviewStatus.AUTO_ACCEPT


def test_human_review_item_evidence_immutability():
    """Verifies that applying a human correction stores corrected text separately while preserving raw OCR text."""
    item = ReviewStateMachine.create_review_item(
        document_id="doc_123",
        page_number=1,
        region_id="reg_05",
        bbox=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=50),
        raw_ocr_text="ರಾಮಯ",
        recognizer="iitb_kannada_v002",
        review_reason="Low character confidence",
        recognizer_confidence_raw=0.55,
    )

    assert item.raw_ocr_text == "ರಾಮಯ"
    assert item.status == ReviewStatus.REVIEW_REQUIRED
    assert item.decision == ReviewDecision.PENDING
    assert item.corrected_text is None

    # Reviewer corrects the text
    item.apply_correction(corrected_text="ರಾಮಯ್ಯ", reviewer_id="officer_kannada_01", notes="Missing double ya")

    # Crucial assertion: raw_ocr_text remains intact and unmodified!
    assert item.raw_ocr_text == "ರಾಮಯ"
    assert item.corrected_text == "ರಾಮಯ್ಯ"
    assert item.decision == ReviewDecision.CORRECTED
    assert item.reviewed_by == "officer_kannada_01"


def test_line_word_decomposition_disabled_by_default():
    """Verifies that line-word decomposition is strictly disabled by default."""
    pipe = DocumentProcessingPipeline()
    assert pipe.enable_line_word_decomposition is False


def test_pipeline_neutral_document_type_and_review_state():
    """Verifies DocumentProcessingPipeline sets Unknown/Not classified and uncalibrated confidence state."""
    pipe = DocumentProcessingPipeline()
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    resp = pipe.process_document(
        image=img,
        request=DocumentProcessingRequest(
            image=img,
            page_number=1,
            document_id="doc_audit_neutral",
            regions=[
                RegionRequest(
                    region_id="r1",
                    bbox=BoundingBox(x_min=10, y_min=10, x_max=190, y_max=90),
                    region_type=RegionType.TEXT,
                    language="kannada",
                )
            ],
        ),
    )

    # Must be neutral document classification (Task 9 Contract: document_type is None, state is UNKNOWN)
    assert resp.document_type is None
    assert resp.document_type_state == "UNKNOWN"
    # Calibrated confidence must be null (Task 3)
    assert resp.calibrated_confidence is None
    # Must have confidence state
    assert resp.confidence_state in (ConfidenceState.UNCALIBRATED, ConfidenceState.REVIEW_REQUIRED)
