"""
src/confidence/scorer.py
Multi-factor field-level and document-level confidence calculation engine.
Structure designed for weighted rule-based scoring and future scikit-learn ML calibration.
"""

from typing import Any, Dict, List, Optional, Tuple
from schemas import (
    ConfidenceBreakdown,
    CrossRecordValidationResult,
    DuplicateResult,
    ExtractedField,
    ExtractionMethod,
    GISValidationResult,
    RuleValidationItem,
    ValidationStatus,
)


class ConfidenceScorer:
    """
    Computes field-level confidence scores based on multi-signal evidence,
    and calculates overall document reliability index and review routing flags.
    Supports weighted multi-factor rule scoring and optional scikit-learn ML calibration.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        high_threshold: float = 0.85,
        medium_threshold: float = 0.65,
        ml_calibrator: Optional[Any] = None,
    ):
        self.weights = weights or {
            "ocr_confidence": 0.25,
            "pattern_strength": 0.20,
            "rule_validation": 0.25,
            "cross_record": 0.15,
            "gis_consistency": 0.15,
        }
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.ml_calibrator = ml_calibrator

    def calculate_field_confidence(
        self,
        field_obj: ExtractedField,
        validation_items: List[RuleValidationItem],
        cross_record: CrossRecordValidationResult,
        gis_result: GISValidationResult,
    ) -> Tuple[float, ConfidenceBreakdown]:
        """
        Calculates granular confidence score for a single extracted field.
        """
        # 1. OCR Confidence signal
        ocr_conf = max(0.0, min(1.0, field_obj.confidence))

        # 2. Pattern Strength signal
        if field_obj.extraction_method == ExtractionMethod.REGEX:
            pattern_score = 0.95
        elif field_obj.extraction_method == ExtractionMethod.TABLE_LOOKUP:
            pattern_score = 0.90
        elif field_obj.extraction_method == ExtractionMethod.HANDWRITING_FUSION:
            pattern_score = 0.85
        else:
            pattern_score = 0.70

        # 3. Rule Validation signal
        field_errors = [item for item in validation_items if item.field_name == field_obj.field_name]
        rule_penalty = 0.0
        if not field_errors:
            rule_score = 1.0
        else:
            if any(item.severity == "error" for item in field_errors):
                rule_score = 0.0
                rule_penalty = 0.20
            elif any(item.severity == "warning" for item in field_errors):
                rule_score = 0.60
            else:
                rule_score = 0.85

        # 4. Cross-Record Consistency signal
        if field_obj.field_name == "land_area":
            if cross_record.area_sum_matches is True:
                cross_score = 1.0
            elif cross_record.area_sum_matches is False:
                cross_score = 0.40
            else:
                cross_score = 0.85
        elif field_obj.field_name == "khasra_number":
            cross_score = 1.0 if cross_record.khasra_count_matches else 0.85
        else:
            cross_score = 0.90 if cross_record.passed else 0.75

        # 5. GIS Consistency signal
        if field_obj.field_name in ["khasra_number", "land_area", "village", "tehsil", "district"]:
            if gis_result.is_verified:
                gis_score = 1.0
            elif gis_result.parcel_id_found and not gis_result.has_mismatch:
                gis_score = 0.90
            elif gis_result.has_mismatch:
                gis_score = 0.35
            else:
                gis_score = 0.70
        else:
            gis_score = 0.90 if gis_result.is_verified else 0.80

        # 6. Disagreement penalty
        disagreement_penalty = 0.0
        if field_obj.validation_status == ValidationStatus.CONFLICT:
            disagreement_penalty = 0.25

        # Weighted calculation
        raw_score = (
            self.weights["ocr_confidence"] * ocr_conf
            + self.weights["pattern_strength"] * pattern_score
            + self.weights["rule_validation"] * rule_score
            + self.weights["cross_record"] * cross_score
            + self.weights["gis_consistency"] * gis_score
            - disagreement_penalty
            - rule_penalty
        )

        final_score = max(0.0, min(1.0, round(raw_score, 4)))

        # If scikit-learn ML calibrator is active, blend learned probability
        if self.ml_calibrator:
            ml_pred = self.ml_calibrator.predict_confidence(
                field_obj=field_obj,
                validation_items=validation_items,
                cross_record=cross_record,
                gis_result=gis_result,
            )
            if ml_pred is not None:
                # 60% ML Calibrator + 40% Rule-Based Weighted Score
                final_score = round(0.60 * ml_pred + 0.40 * final_score, 4)

        breakdown = ConfidenceBreakdown(
            ocr_confidence=round(ocr_conf, 3),
            pattern_strength=round(pattern_score, 3),
            rule_validation_score=round(rule_score, 3),
            cross_record_consistency=round(cross_score, 3),
            gis_consistency=round(gis_score, 3),
            disagreement_penalty=round(disagreement_penalty, 3),
            final_score=final_score,
        )

        # Update field object confidence
        field_obj.confidence = final_score
        return final_score, breakdown

    def evaluate_document_confidence(
        self,
        fields: Dict[str, ExtractedField],
        validation_items: List[RuleValidationItem],
        cross_record: CrossRecordValidationResult,
        gis_result: GISValidationResult,
        duplicate_result: DuplicateResult,
    ) -> Tuple[float, bool, List[str]]:
        """
        Calculates document-level confidence and flags if human review is required.
        """
        review_reasons: List[str] = []

        if not fields:
            return 0.0, True, ["No structured fields could be extracted from document"]

        # Calculate field confidence scores
        scores = []
        for field_name, field_obj in fields.items():
            score, _ = self.calculate_field_confidence(
                field_obj=field_obj,
                validation_items=validation_items,
                cross_record=cross_record,
                gis_result=gis_result,
            )
            scores.append(score)

            if score < self.medium_threshold:
                review_reasons.append(f"Low confidence in field '{field_name}' ({score:.2f})")
            if field_obj.validation_status == ValidationStatus.CONFLICT:
                review_reasons.append(f"Unresolved OCR disagreement in field '{field_name}'")

        overall_conf = round(sum(scores) / len(scores), 4)

        # Check critical system flags for human review
        if any(item.severity == "error" for item in validation_items):
            review_reasons.append("Document failed mandatory rule validations")

        if not cross_record.passed:
            review_reasons.extend(cross_record.inconsistencies)

        if gis_result.has_mismatch:
            review_reasons.extend(gis_result.flag_reasons)

        if duplicate_result.is_duplicate:
            review_reasons.append(f"Potential duplicate record ({duplicate_result.evidence})")

        requires_review = (overall_conf < self.high_threshold) or (len(review_reasons) > 0)

        return overall_conf, requires_review, review_reasons
