"""
Unit tests for printed OCR vs handwritten OCR disagreement handling and resolution.
"""

import pytest
from schemas import (
    BoundingBox,
    ExtractedField,
    ExtractionMethod,
    FieldEvidence,
    HandwritingRegionResult,
    HandwritingResult,
    ValidationStatus,
)
from src.extraction.disagreement import DisagreementResolver


def test_handwriting_overrides_when_high_confidence():
    resolver = DisagreementResolver(iou_overlap_threshold=0.20)

    # Printed OCR extracted template header "खसरा संख्या" with low confidence
    bbox = BoundingBox(x_min=100, y_min=100, x_max=300, y_max=150)
    fields = {
        "khasra_number": ExtractedField(
            field_name="khasra_number",
            raw_value="000",
            normalized_value="000",
            confidence=0.55,
            page=1,
            bbox=bbox,
            evidence=FieldEvidence(page_number=1, bbox=bbox, raw_ocr_text="000"),
            extraction_method=ExtractionMethod.REGEX,
        )
    }

    # Handwriting OCR detected handwritten entry "142/1" with high confidence (0.92)
    hw_result = HandwritingResult(
        document_id="DOC_HW_01",
        regions=[
            HandwritingRegionResult(
                region_id="hw_01",
                text="142/1",
                confidence=0.92,
                page_number=1,
                bbox=BoundingBox(x_min=105, y_min=102, x_max=295, y_max=148),
                model_version="trocr-base-v1",
            )
        ]
    )

    resolved, log = resolver.resolve_conflicts(fields, hw_result, {})

    assert len(log) == 1
    assert log[0]["chosen_value"] == "142/1"
    assert log[0]["chosen_source"] == "handwriting_ocr"
    assert resolved["khasra_number"].raw_value == "142/1"
    assert resolved["khasra_number"].extraction_method == ExtractionMethod.HANDWRITING_FUSION
    assert resolved["khasra_number"].evidence.handwriting_text == "142/1"


def test_unresolved_ambiguity_penalizes_and_flags():
    resolver = DisagreementResolver(iou_overlap_threshold=0.20)

    bbox = BoundingBox(x_min=100, y_min=100, x_max=300, y_max=150)
    fields = {
        "owner_name": ExtractedField(
            field_name="owner_name",
            raw_value="राम प्रसाद",
            normalized_value="राम प्रसाद",
            confidence=0.88,
            page=1,
            bbox=bbox,
            evidence=FieldEvidence(page_number=1, bbox=bbox, raw_ocr_text="राम प्रसाद"),
        )
    }

    # Handwriting engine recognized conflicting name "श्याम प्रसाद" with similar confidence (0.85)
    hw_result = HandwritingResult(
        document_id="DOC_HW_02",
        regions=[
            HandwritingRegionResult(
                region_id="hw_02",
                text="श्याम प्रसाद",
                confidence=0.85,
                page_number=1,
                bbox=BoundingBox(x_min=100, y_min=100, x_max=300, y_max=150),
            )
        ]
    )

    resolved, log = resolver.resolve_conflicts(fields, hw_result, {})

    assert len(log) == 1
    assert log[0]["strategy"] == "unresolved_ambiguity"
    assert resolved["owner_name"].validation_status == ValidationStatus.CONFLICT
    assert resolved["owner_name"].confidence < 0.88  # Penalty applied
