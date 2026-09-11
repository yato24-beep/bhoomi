"""
Unit tests for Scikit-Learn ML Confidence Calibrator.
"""

import numpy as np
import pytest
from schemas import (
    CrossRecordValidationResult,
    ExtractedField,
    ExtractionMethod,
    GISValidationResult,
    RuleValidationItem,
)
from src.confidence.ml_calibrator import MLConfidenceCalibrator


def test_feature_extraction():
    calibrator = MLConfidenceCalibrator(model_path="models/test_calibrator.joblib")
    
    field = ExtractedField(
        field_name="land_area",
        raw_value="0.4500",
        normalized_value=0.4500,
        confidence=0.95,
        extraction_method=ExtractionMethod.REGEX,
    )

    validation_items = [
        RuleValidationItem(
            rule_name="min_range_land_area",
            field_name="land_area",
            passed=True,
            severity="info",
            message="OK",
        )
    ]

    features = calibrator.extract_features(
        field_obj=field,
        validation_items=validation_items,
        cross_record=CrossRecordValidationResult(passed=True),
        gis_result=GISValidationResult(is_verified=True),
    )

    assert isinstance(features, np.ndarray)
    assert len(features) == len(MLConfidenceCalibrator.FEATURE_NAMES)
    assert features[0] == pytest.approx(0.95, rel=1e-2)
    assert features[4] == 0.0  # error count


def test_train_and_predict():
    calibrator = MLConfidenceCalibrator(model_path="models/test_calibrator.joblib")
    
    # Synthetic feature data: 10 samples
    X = np.array([
        [0.98, 0.95, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.95, 0.95, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.92, 0.90, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.40, 0.70, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [0.30, 0.70, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0],
        [0.99, 0.95, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [0.85, 0.95, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.20, 0.70, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ])
    y = np.array([1, 1, 1, 0, 0, 1, 1, 0])

    calibrator.train_and_save(X, y, save_path="models/test_calibrator.joblib")
    assert calibrator.model is not None

    field_good = ExtractedField(field_name="khasra", raw_value="142", normalized_value="142", confidence=0.98)
    prob_good = calibrator.predict_confidence(
        field_obj=field_good,
        validation_items=[],
        cross_record=CrossRecordValidationResult(passed=True),
        gis_result=GISValidationResult(is_verified=True),
    )
    assert prob_good is not None
    assert prob_good > 0.50
