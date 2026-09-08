"""Unit tests for Multimodal Language & Script Router."""

import unittest
from unittest.mock import MagicMock

from schemas import BoundingBox, OCRResult
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput
from src.handwriting.router import LanguageScriptRouter


class MockRecognizer(BaseHandwritingRecognizer):
    """Mock recognizer for testing router dispatch."""

    def __init__(self, name: str, version: str, output_text: str = "Test Output"):
        super().__init__(model_name=name, model_version=version)
        self.output_text = output_text

    def recognize_handwriting(
        self,
        image: ImageInput,
        bbox=None,
        page_number=None,
        preprocessing_info=None,
        is_handwritten=None,
        **kwargs,
    ) -> OCRResult:
        return OCRResult(
            text=self.output_text,
            confidence=0.95,
            bbox=bbox,
            is_handwritten=is_handwritten,
            page_number=page_number,
            model_name=self.model_name,
            model_version=self.model_version,
            metadata={"metadata": {"engine": self.model_name}},
        )


class TestLanguageScriptRouterMultimodal(unittest.TestCase):
    """Unit tests for multimodal script & handwriting routing."""

    def setUp(self):
        self.printed_kannada = MockRecognizer("paddle-kannada", "v1.0", "ಮುದ್ರಿತ ಕನ್ನಡ")
        self.hw_kannada = MockRecognizer("trocr-kannada-trained", "v2.0", "ಕೈಬರಹದ ಕನ್ನಡ")
        self.printed_english = MockRecognizer("paddle-english", "v1.0", "Printed English")
        self.hw_english = MockRecognizer("trocr-english-base", "v1.0", "Handwritten English")

        self.router = LanguageScriptRouter(auto_register_defaults=False, auto_register_kannada=False)
        self.router.register_recognizer("kannada", self.printed_kannada, is_handwritten=False, set_as_default=True)
        self.router.register_recognizer("kannada", self.hw_kannada, is_handwritten=True)
        self.router.register_recognizer("english", self.printed_english, is_handwritten=False)
        self.router.register_recognizer("english", self.hw_english, is_handwritten=True)

    def test_routes_kannada_handwriting_to_trained_checkpoint(self):
        """Verifies that handwritten Kannada routes to the fine-tuned TrOCR model."""
        res = self.router.route_and_recognize("dummy.png", language="kannada", is_handwritten=True)
        self.assertEqual(res.model_name, "trocr-kannada-trained")
        self.assertEqual(res.text, "ಕೈಬರಹದ ಕನ್ನಡ")

    def test_routes_kannada_printed_to_paddle_backend(self):
        """Verifies that printed Kannada routes to the PaddleOCR backend."""
        res = self.router.route_and_recognize("dummy.png", language="kannada", is_handwritten=False)
        self.assertEqual(res.model_name, "paddle-kannada")
        self.assertEqual(res.text, "ಮುದ್ರಿತ ಕನ್ನಡ")

    def test_routes_english_handwriting_to_trocr_base(self):
        """Verifies that English handwriting routes to pretrained TrOCR."""
        res = self.router.route_and_recognize("dummy.png", language="english", is_handwritten=True)
        self.assertEqual(res.model_name, "trocr-english-base")
        self.assertEqual(res.text, "Handwritten English")

    def test_unsupported_handwriting_language_returns_auditable_review_flag(self):
        """Verifies that unsupported handwritten languages (e.g. Telugu) return an explicit review result without fabrication."""
        res = self.router.route_and_recognize("dummy.png", language="telugu", is_handwritten=True)

        self.assertEqual(res.text, "")
        self.assertIsNone(res.confidence)
        self.assertTrue(res.is_handwritten)

        audit = res.metadata
        self.assertEqual(audit["calculation_method"], "unsupported_handwriting_language")
        self.assertEqual(audit["metadata"]["engine_status"], "unsupported_language_model")
        self.assertEqual(audit["metadata"]["language"], "telugu")
        self.assertTrue(audit["metadata"]["requires_human_review"])


if __name__ == "__main__":
    unittest.main()
