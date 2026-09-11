"""
src/confidence/ml_calibrator.py
Scikit-learn based Machine Learning Confidence Calibrator.
Predicts calibrated field-level correctness probabilities from multi-signal feature vectors.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from schemas import (
    CrossRecordValidationResult,
    ExtractedField,
    ExtractionMethod,
    GISValidationResult,
    RuleValidationItem,
    ValidationStatus,
)


class MLConfidenceCalibrator:
    """
    Classical Machine Learning Confidence Calibrator using scikit-learn.
    Enhances or replaces the weighted rule scorer with learned correctness probabilities.
    """

    FEATURE_NAMES = [
        "ocr_confidence",
        "pattern_strength",
        "is_regex_match",
        "is_table_match",
        "rule_error_count",
        "rule_warning_count",
        "cross_record_passed",
        "gis_verified",
        "has_conflict",
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "models/confidence_calibrator.joblib"
        self.model: Optional[Any] = None
        self._load_model()

    def _load_model(self):
        """Attempts to load a trained scikit-learn model from disk."""
        if os.path.isfile(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None

    def extract_features(
        self,
        field_obj: ExtractedField,
        validation_items: List[RuleValidationItem],
        cross_record: CrossRecordValidationResult,
        gis_result: GISValidationResult,
    ) -> np.ndarray:
        """
        Extracts 9-dimensional feature vector for a single extracted field.
        """
        field_errors = [item for item in validation_items if item.field_name == field_obj.field_name]
        error_count = sum(1 for item in field_errors if item.severity == "error")
        warning_count = sum(1 for item in field_errors if item.severity == "warning")

        pattern_strength = 0.95 if field_obj.extraction_method == ExtractionMethod.REGEX else 0.85
        is_regex = 1.0 if field_obj.extraction_method == ExtractionMethod.REGEX else 0.0
        is_table = 1.0 if field_obj.extraction_method == ExtractionMethod.TABLE_LOOKUP else 0.0
        cross_passed = 1.0 if cross_record.passed else 0.0
        gis_ok = 1.0 if gis_result.is_verified else (0.5 if not gis_result.has_mismatch else 0.0)
        has_conflict = 1.0 if field_obj.validation_status == ValidationStatus.CONFLICT else 0.0

        features = [
            float(field_obj.confidence),
            float(pattern_strength),
            float(is_regex),
            float(is_table),
            float(error_count),
            float(warning_count),
            float(cross_passed),
            float(gis_ok),
            float(has_conflict),
        ]
        return np.array(features, dtype=np.float32)

    def predict_confidence(
        self,
        field_obj: ExtractedField,
        validation_items: List[RuleValidationItem],
        cross_record: CrossRecordValidationResult,
        gis_result: GISValidationResult,
    ) -> float:
        """
        Predicts calibrated probability of correctness between 0.0 and 1.0.
        If ML model is not trained yet, returns None so caller uses weighted scoring.
        """
        if self.model is None:
            return None

        features = self.extract_features(field_obj, validation_items, cross_record, gis_result).reshape(1, -1)
        prob = float(self.model.predict_proba(features)[0, 1])
        return round(max(0.0, min(1.0, prob)), 4)

    def train_and_save(self, X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None):
        """
        Trains a Logistic Regression classifier with probability calibration and saves to disk.
        """
        base_clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        calibrated = CalibratedClassifierCV(estimator=base_clf, cv=2)
        calibrated.fit(X, y)
        self.model = calibrated

        out_path = Path(save_path or self.model_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(out_path))
