"""Unit tests for the Person B Handwriting OCR interface and service components."""

import unittest
from typing import Any, Dict, List, Optional

from schemas import BoundingBox, DocumentPage, OCRResult
from src.handwriting.confidence import (
    build_confidence_audit_trail,
    classify_confidence_tier,
    compute_min_token_confidence,
    compute_token_geometric_mean_confidence,
    compute_token_mean_confidence,
)
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput
from src.handwriting.service import HandwritingOCRService


class DummyTestRecognizer(BaseHandwritingRecognizer):
    """Explicitly named test double implementing BaseHandwritingRecognizer for interface verification."""

    def __init__(
        self,
        model_name: str = "dummy-test-ocr",
        model_version: str = "v0.0.1-test",
        fixed_text: str = "Sample Test Text",
        token_probs: Optional[List[float]] = None,
    ):
        super().__init__(model_name=model_name, model_version=model_version)
        self.fixed_text = fixed_text
        self.token_probs = token_probs

    def recognize_handwriting(
        self,
        image: ImageInput,
        bbox: Optional[BoundingBox] = None,
        page_number: Optional[int] = None,
        preprocessing_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> OCRResult:
        confidence = compute_token_mean_confidence(self.token_probs) if self.token_probs else None
        audit = build_confidence_audit_trail(
            raw_token_probabilities=self.token_probs,
            calculation_method="token_mean" if self.token_probs else "unavailable",
            custom_metadata={"preprocessing": preprocessing_info} if preprocessing_info else None,
        )

        return OCRResult(
            text=self.fixed_text,
            confidence=confidence,
            bbox=bbox,
            is_handwritten=True,
            page_number=page_number,
            model_name=self.model_name,
            model_version=self.model_version,
            metadata=audit,
        )


class TestRecognizerInterface(unittest.TestCase):
    """Tests for BaseHandwritingRecognizer abstraction."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Verifies that BaseHandwritingRecognizer cannot be directly instantiated."""
        with self.assertRaises(TypeError):
            BaseHandwritingRecognizer(model_name="test", model_version="1.0")  # type: ignore

    def test_concrete_recognizer_returns_valid_ocr_result(self):
        """Tests that a concrete implementation returns a valid OCRResult with metadata."""
        recognizer = DummyTestRecognizer(token_probs=[0.90, 0.95, 0.92])
        bbox = BoundingBox(x_min=10, y_min=20, x_max=100, y_max=50)

        result = recognizer.recognize_handwriting(
            image="dummy_path.png",
            bbox=bbox,
            page_number=1,
            preprocessing_info={"deskew_angle": 1.2},
        )

        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.text, "Sample Test Text")
        self.assertEqual(result.confidence, 0.9233)
        self.assertTrue(result.is_handwritten)
        self.assertEqual(result.bbox, bbox)
        self.assertEqual(result.page_number, 1)
        self.assertEqual(result.model_name, "dummy-test-ocr")
        self.assertEqual(result.model_version, "v0.0.1-test")
        self.assertEqual(result.metadata["token_count"], 3)
        self.assertEqual(result.metadata["min_token_confidence"], 0.90)

    def test_recognizer_handles_missing_confidence(self):
        """Tests that recognizer supports None for models without probabilistic confidence."""
        recognizer = DummyTestRecognizer(token_probs=None)
        result = recognizer.recognize_handwriting(image="dummy_path.png")

        self.assertIsNone(result.confidence)
        self.assertEqual(result.metadata["calculation_method"], "unavailable")
        self.assertFalse(result.metadata["has_token_probabilities"])

    def test_batch_recognition(self):
        """Tests batch recognition interface."""
        recognizer = DummyTestRecognizer(token_probs=[0.88, 0.90])
        images = ["crop1.png", "crop2.png"]
        bboxes = [
            BoundingBox(x_min=0, y_min=0, x_max=50, y_max=20),
            BoundingBox(x_min=0, y_min=25, x_max=50, y_max=45),
        ]
        pages = [1, 1]

        results = recognizer.recognize_batch(images=images, bboxes=bboxes, page_numbers=pages)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(r, OCRResult) for r in results))
        self.assertEqual(results[0].bbox, bboxes[0])
        self.assertEqual(results[1].bbox, bboxes[1])


class TestHandwritingOCRService(unittest.TestCase):
    """Tests for HandwritingOCRService dependency injection and orchestration."""

    def test_service_initialization_and_type_check(self):
        """Tests dependency injection and type validation on init."""
        recognizer = DummyTestRecognizer()
        service = HandwritingOCRService(recognizer=recognizer)

        self.assertIs(service.recognizer, recognizer)
        self.assertEqual(service.get_model_info()["model_name"], "dummy-test-ocr")

        with self.assertRaises(TypeError):
            HandwritingOCRService(recognizer="not-a-recognizer")  # type: ignore

    def test_service_set_recognizer(self):
        """Tests swapping recognizer implementation dynamically."""
        rec1 = DummyTestRecognizer(model_name="baseline-rec", model_version="1.0")
        rec2 = DummyTestRecognizer(model_name="finetuned-trocr", model_version="2.0")

        service = HandwritingOCRService(recognizer=rec1)
        self.assertEqual(service.get_model_info()["model_name"], "baseline-rec")

        service.set_recognizer(rec2)
        self.assertEqual(service.get_model_info()["model_name"], "finetuned-trocr")
        self.assertEqual(service.get_model_info()["model_version"], "2.0")

    def test_service_process_crop(self):
        """Tests processing an individual crop through the service."""
        recognizer = DummyTestRecognizer(fixed_text="Parcel ID 402/12")
        service = HandwritingOCRService(recognizer=recognizer)
        bbox = BoundingBox(x_min=5, y_min=10, x_max=80, y_max=30)

        result = service.process_crop(image="crop.png", bbox=bbox, page_number=2)

        self.assertEqual(result.text, "Parcel ID 402/12")
        self.assertEqual(result.bbox, bbox)
        self.assertEqual(result.page_number, 2)

    def test_service_process_document_page_crops(self):
        """Tests updating a DocumentPage with recognized crops."""
        recognizer = DummyTestRecognizer(token_probs=[0.95])
        service = HandwritingOCRService(recognizer=recognizer)

        page = DocumentPage(page_number=1, image_path="page_1.png")
        crops = ["crop1.png", "crop2.png"]
        bboxes = [
            BoundingBox(x_min=10, y_min=10, x_max=60, y_max=30),
            BoundingBox(x_min=10, y_min=40, x_max=60, y_max=60),
        ]

        updated_page = service.process_document_page_crops(
            page=page,
            crop_images=crops,
            crop_bboxes=bboxes,
            preprocessing_info={"binarized": True},
        )

        self.assertEqual(len(updated_page.ocr_results), 2)
        self.assertEqual(updated_page.ocr_results[0].page_number, 1)
        self.assertEqual(updated_page.ocr_results[1].page_number, 1)
        self.assertEqual(updated_page.ocr_results[0].bbox, bboxes[0])


class TestConfidenceUtilities(unittest.TestCase):
    """Tests for non-fabricated confidence evaluation and audit functions."""

    def test_compute_token_mean_confidence(self):
        """Tests mean calculation on token probabilities."""
        self.assertEqual(compute_token_mean_confidence([0.8, 0.9, 1.0]), 0.9)
        self.assertIsNone(compute_token_mean_confidence([]))
        self.assertIsNone(compute_token_mean_confidence(None))
        self.assertIsNone(compute_token_mean_confidence([0.5, 1.5]))
        self.assertIsNone(compute_token_mean_confidence([-0.1, 0.8]))

    def test_compute_token_geometric_mean_confidence(self):
        """Tests geometric mean calculation in log-space."""
        self.assertEqual(compute_token_geometric_mean_confidence([1.0, 1.0]), 1.0)
        self.assertEqual(compute_token_geometric_mean_confidence([0.5, 0.5]), 0.5)
        self.assertIsNone(compute_token_geometric_mean_confidence([]))
        self.assertIsNone(compute_token_geometric_mean_confidence([0.0, 0.9]))

    def test_compute_min_token_confidence(self):
        """Tests minimum token confidence extraction."""
        self.assertEqual(compute_min_token_confidence([0.95, 0.42, 0.88]), 0.42)
        self.assertIsNone(compute_min_token_confidence([]))
        self.assertIsNone(compute_min_token_confidence(None))

    def test_classify_confidence_tier(self):
        """Tests tier classification thresholds."""
        self.assertEqual(classify_confidence_tier(0.92), "HIGH")
        self.assertEqual(classify_confidence_tier(0.85), "HIGH")
        self.assertEqual(classify_confidence_tier(0.75), "MEDIUM")
        self.assertEqual(classify_confidence_tier(0.60), "MEDIUM")
        self.assertEqual(classify_confidence_tier(0.40), "LOW")
        self.assertEqual(classify_confidence_tier(None), "UNKNOWN")
        self.assertEqual(classify_confidence_tier(-0.5), "UNKNOWN")

    def test_build_confidence_audit_trail(self):
        """Tests creation of structured audit trails."""
        audit = build_confidence_audit_trail(
            raw_token_probabilities=[0.90, 0.80],
            calculation_method="token_mean",
            custom_metadata={"beam_width": 4},
        )

        self.assertEqual(audit["calculation_method"], "token_mean")
        self.assertEqual(audit["token_count"], 2)
        self.assertTrue(audit["has_token_probabilities"])
        self.assertEqual(audit["min_token_confidence"], 0.80)
        self.assertEqual(audit["mean_token_confidence"], 0.85)
        self.assertEqual(audit["metadata"]["beam_width"], 4)

        empty_audit = build_confidence_audit_trail(raw_token_probabilities=None)
        self.assertFalse(empty_audit["has_token_probabilities"])
        self.assertIsNone(empty_audit["token_count"])
        self.assertNotIn("min_token_confidence", empty_audit)


if __name__ == "__main__":
    unittest.main()
