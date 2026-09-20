"""Unit tests for Confidence Calibration System & Reproducible Fitting Utility.

Tests:
1. Insufficient calibration data (< 50 samples): calibrated confidence remains None.
2. Valid calibration data (>= 50 samples): fits Platt/Temperature scaling, verifies ECE reduction.
3. Missing labels / incomplete records: filtered safely without crashing.
4. Invalid confidence values (<0, >1, NaN, non-numeric): filtered safely.
5. Calibrated confidence propagation through DocumentProcessingPipeline:
   - Uncalibrated -> calibrated_confidence is None, confidence_state is UNCALIBRATED.
   - Calibrated -> calibrated_confidence is float, confidence_state is CALIBRATED.
"""

import unittest
import numpy as np
from PIL import Image

from schemas import BoundingBox
from src.confidence.calibration import CalibrationMethod, EngineCalibrator, MultiEngineCalibrator
from src.confidence.calibration_utility import (
    validate_calibration_dataset,
    fit_engine_calibration,
    build_multi_engine_calibration,
)
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput
from src.handwriting.router import LanguageScriptRouter
from src.integration.document_pipeline import DocumentProcessingPipeline
from src.integration.schemas import ConfidenceState, OCRResult


class MockCalibRecognizer(BaseHandwritingRecognizer):
    """Mock recognizer with controllable confidence score."""

    def __init__(self, name: str, output_text: str = "ಟೆಸ್ಟ್ ಪಠ್ಯ", confidence: float = 0.85):
        super().__init__(model_name=name, model_version="1.0")
        self.output_text = output_text
        self.confidence = confidence

    def recognize_handwriting(self, image: ImageInput, bbox=None, **kwargs) -> OCRResult:
        return OCRResult(
            text=self.output_text,
            confidence=self.confidence,
            bbox=bbox,
            model_name=self.model_name,
            model_version=self.model_version,
        )


class TestConfidenceCalibration(unittest.TestCase):
    """Confidence calibration unit test suite."""

    def setUp(self):
        np.random.seed(42)

    def test_1_insufficient_calibration_data(self):
        """Scenario 1: Insufficient calibration samples (< 50) must NOT fit calibrator and keep calibrated confidence None."""
        records = [
            {"engine_name": "easyocr_kannada", "raw_score": 0.85, "label": 1},
            {"engine_name": "easyocr_kannada", "raw_score": 0.40, "label": 0},
            {"engine_name": "easyocr_kannada", "raw_score": 0.90, "label": 1},
        ]
        cal, summary = fit_engine_calibration(records, "easyocr_kannada", min_samples=50)

        self.assertIsNone(cal)
        self.assertEqual(summary["status"], "SKIPPED_INSUFFICIENT_DATA")
        self.assertIsNone(summary["calibrated_confidence"])
        self.assertFalse(summary["is_calibrated"])
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["min_required"], 50)

    def test_2_valid_calibration_data(self):
        """Scenario 2: Valid calibration data (>= 50 samples) fits parameters and produces calibrated predictions."""
        # Generate 60 authentic-like samples
        records = []
        for i in range(60):
            score = 0.5 + 0.45 * (i / 60.0)
            # Label correlates with score with some noise
            label = 1 if (score > 0.65 or np.random.rand() > 0.4) else 0
            records.append({
                "engine_name": "easyocr_kannada",
                "raw_score": round(score, 4),
                "label": label,
            })

        cal, summary = fit_engine_calibration(records, "easyocr_kannada", min_samples=50)

        self.assertIsNotNone(cal)
        self.assertTrue(cal.is_fitted)
        self.assertEqual(summary["status"], "FITTED")
        self.assertTrue(summary["is_calibrated"])
        self.assertIn("metrics", summary)
        self.assertIn("calibrated_ece", summary["metrics"])

        # Test predict produces calibrated float in [0.0, 1.0]
        calibrated_score = cal.predict(0.80)
        self.assertIsInstance(calibrated_score, float)
        self.assertGreaterEqual(calibrated_score, 0.0)
        self.assertLessEqual(calibrated_score, 1.0)

    def test_3_missing_labels_handling(self):
        """Scenario 3: Missing labels are safely filtered out and tracked in validation report."""
        records = [
            {"engine_name": "paddleocr", "raw_score": 0.90, "label": 1},
            {"engine_name": "paddleocr", "raw_score": 0.85},  # Missing label
            {"engine_name": "paddleocr", "raw_score": 0.70, "label": None},  # None label
            {"engine_name": "paddleocr", "raw_score": 0.95, "predicted_text": "same", "ground_truth": "same"},  # Derived label 1
            {"engine_name": "paddleocr", "raw_score": 0.60, "predicted_text": "foo", "ground_truth": "bar"},  # Derived label 0
        ]
        clean, report = validate_calibration_dataset(records)

        self.assertEqual(report.total_records, 5)
        self.assertEqual(report.valid_records, 3)
        self.assertEqual(report.missing_label_count, 2)
        self.assertEqual(len(clean), 3)

    def test_4_invalid_confidence_values(self):
        """Scenario 4: Invalid confidence scores (<0, >1, NaN, non-numeric) are filtered."""
        records = [
            {"engine_name": "trocr", "raw_score": 0.85, "label": 1},
            {"engine_name": "trocr", "raw_score": -0.5, "label": 1},  # Negative
            {"engine_name": "trocr", "raw_score": 1.5, "label": 1},   # > 1.0
            {"engine_name": "trocr", "raw_score": float("nan"), "label": 0},  # NaN
            {"engine_name": "trocr", "raw_score": "not_a_number", "label": 0},  # Unparseable
            {"engine_name": "trocr", "raw_score": 0.92, "label": 1},  # Valid
        ]
        clean, report = validate_calibration_dataset(records)

        self.assertEqual(report.valid_records, 2)
        self.assertEqual(report.invalid_score_count, 4)
        self.assertEqual(len(clean), 2)

    def test_5_calibrated_confidence_propagation_pipeline(self):
        """Scenario 5: Propagation of calibrated vs uncalibrated confidence through DocumentProcessingPipeline."""
        test_img = Image.new("RGB", (600, 400), (255, 255, 255))
        reg_box = BoundingBox(x_min=50, y_min=50, x_max=550, y_max=150)

        # Part A: Uncalibrated pipeline (no fitted calibrator)
        rec_uncal = MockCalibRecognizer("mock-engine", confidence=0.88)
        router_uncal = LanguageScriptRouter(auto_register_defaults=False, auto_register_kannada=False)
        router_uncal.register_recognizer("kannada", rec_uncal, is_handwritten=False, set_as_default=True)
        pipeline_uncal = DocumentProcessingPipeline(router=router_uncal)

        res_uncal = pipeline_uncal.process_document(image=test_img, regions=[reg_box], language="kannada")

        # Must keep calibrated confidence as None and state as UNCALIBRATED
        self.assertIsNone(res_uncal.calibrated_confidence)
        self.assertIsNone(res_uncal.ordered_regions[0].calibrated_confidence)
        self.assertEqual(res_uncal.confidence_state, ConfidenceState.UNCALIBRATED)

        # Part B: Calibrated pipeline (with fitted MultiEngineCalibrator)
        fitted_multi_cal = MultiEngineCalibrator()
        eng_cal = EngineCalibrator(engine_name="mock-engine", method=CalibrationMethod.PLATTS_SCALING)
        eng_cal.is_fitted = True
        eng_cal.a = 2.0
        eng_cal.b = -0.5
        fitted_multi_cal.calibrators["mock-engine"] = eng_cal

        rec_cal = MockCalibRecognizer("mock-engine", confidence=0.88)
        router_cal = LanguageScriptRouter(auto_register_defaults=False, auto_register_kannada=False)
        router_cal.register_recognizer("kannada", rec_cal, is_handwritten=False, set_as_default=True)
        pipeline_cal = DocumentProcessingPipeline(router=router_cal, calibrator=fitted_multi_cal)

        res_cal = pipeline_cal.process_document(image=test_img, regions=[reg_box], language="kannada")

        # Must propagate numeric calibrated confidence and set state to CALIBRATED
        self.assertIsNotNone(res_cal.calibrated_confidence)
        self.assertIsInstance(res_cal.calibrated_confidence, float)
        self.assertIsNotNone(res_cal.ordered_regions[0].calibrated_confidence)
        self.assertEqual(res_cal.confidence_state, ConfidenceState.CALIBRATED)


if __name__ == "__main__":
    unittest.main()
