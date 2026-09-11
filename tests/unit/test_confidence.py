"""
Unit tests for field-level confidence scoring and human review routing logic.
"""

import pytest
from schemas import (
    CrossRecordValidationResult,
    DuplicateResult,
    ExtractedField,
    ExtractionMethod,
    GISValidationResult,
    RuleValidationItem,
    ValidationStatus,
)
from src.confidence.scorer import ConfidenceScorer


@pytest.fixture
def scorer():
    return ConfidenceScorer(
        high_threshold=0.85,
        medium_threshold=0.65,
    )


def test_high_confidence_field(scorer):
    field = ExtractedField(
        field_name="khasra_number",
        raw_value="142/1",
        normalized_value="142/1",
        confidence=0.98,
        extraction_method=ExtractionMethod.REGEX,
    )

    score, breakdown = scorer.calculate_field_confidence(
        field_obj=field,
        validation_items=[],
        cross_record=CrossRecordValidationResult(passed=True),
        gis_result=GISValidationResult(is_verified=True),
    )

    assert score >= 0.90
    assert breakdown.ocr_confidence == 0.98
    assert breakdown.disagreement_penalty == 0.0


def test_penalized_confidence_on_rule_error(scorer):
    field = ExtractedField(
        field_name="land_area",
        raw_value="-5.0",
        normalized_value=-5.0,
        confidence=0.90,
        extraction_method=ExtractionMethod.REGEX,
    )

    validation_items = [
        RuleValidationItem(
            rule_name="min_range_land_area",
            field_name="land_area",
            passed=False,
            severity="error",
            message="Area cannot be negative",
        )
    ]

    score, breakdown = scorer.calculate_field_confidence(
        field_obj=field,
        validation_items=validation_items,
        cross_record=CrossRecordValidationResult(passed=True),
        gis_result=GISValidationResult(is_verified=True),
    )

    assert score < 0.70
    assert breakdown.rule_validation_score == 0.0


def test_document_review_routing(scorer):
    fields = {
        "khasra_number": ExtractedField(
            field_name="khasra_number",
            raw_value="142/1",
            normalized_value="142/1",
            confidence=0.95,
        ),
    }

    # GIS mismatch triggers human review
    gis_mismatch = GISValidationResult(
        is_verified=False,
        has_mismatch=True,
        flag_reasons=["Cadastral boundary mismatch"],
    )

    overall_conf, requires_review, reasons = scorer.evaluate_document_confidence(
        fields=fields,
        validation_items=[],
        cross_record=CrossRecordValidationResult(passed=True),
        gis_result=gis_mismatch,
        duplicate_result=DuplicateResult(is_duplicate=False),
    )

    assert requires_review is True
    assert any("Cadastral boundary mismatch" in r for r in reasons)
