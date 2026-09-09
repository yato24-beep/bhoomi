"""Unit tests for LanguageScriptRouter."""

import unittest
from unittest.mock import MagicMock
from PIL import Image

from schemas import OCRResult
from src.handwriting.recognizer import BaseHandwritingRecognizer
from src.handwriting.router import LanguageScriptRouter


class DummyLanguageRecognizer(BaseHandwritingRecognizer):
    """Test recognizer for testing router dispatch."""

    def __init__(self, model_name: str, detected_text: str):
        super().__init__(model_name=model_name, model_version="1.0.0")
        self.detected_text = detected_text

    def recognize_handwriting(self, image, bbox=None, page_number=None, preprocessing_info=None, **kwargs):
        return OCRResult(
            text=self.detected_text,
            confidence=0.95,
            model_name=self.model_name,
            model_version=self.model_version,
            is_handwritten=True,
        )


class TestLanguageScriptRouter(unittest.TestCase):
    """Test suite for LanguageScriptRouter."""

    def setUp(self):
        self.sample_img = Image.new("RGB", (100, 30), color=(255, 255, 255))
        self.kannada_rec = DummyLanguageRecognizer("kannada-engine", "ಕನ್ನಡ ಪಠ್ಯ")
        self.telugu_rec = DummyLanguageRecognizer("telugu-engine", "తెలుగు వచనం")
        self.tamil_rec = DummyLanguageRecognizer("tamil-engine", "தமிழ் உரை")
        self.hindi_rec = DummyLanguageRecognizer("hindi-engine", "हिंदी पाठ")

    def test_router_initialization_and_default(self):
        """Tests router default language and automatic Kannada setup."""
        router = LanguageScriptRouter(
            default_language="kannada",
            auto_register_kannada=True,
            kannada_recognizer=self.kannada_rec,
        )

        rec = router.get_recognizer()
        self.assertIs(rec, self.kannada_rec)
        self.assertIn("kannada", router.list_supported_languages())

    def test_router_multilingual_registration_and_dispatch(self):
        """Tests registering and routing across Kannada, Telugu, Tamil, and Hindi."""
        router = LanguageScriptRouter(
            default_language="kannada",
            auto_register_kannada=False,
        )

        router.register_recognizer("kannada", self.kannada_rec, set_as_default=True)
        router.register_recognizer("telugu", self.telugu_rec)
        router.register_recognizer("tamil", self.tamil_rec)
        router.register_recognizer("hindi", self.hindi_rec)

        self.assertEqual(
            sorted(router.list_supported_languages()),
            ["hindi", "kannada", "tamil", "telugu"],
        )

        # Test ISO / Alias resolution
        res_kn = router.route_and_recognize(self.sample_img, language="kn")
        self.assertEqual(res_kn.text, "ಕನ್ನಡ ಪಠ್ಯ")
        self.assertEqual(res_kn.model_name, "kannada-engine")

        res_te = router.route_and_recognize(self.sample_img, language="TE")
        self.assertEqual(res_te.text, "తెలుగు వచనం")

        res_ta = router.route_and_recognize(self.sample_img, language="ta")
        self.assertEqual(res_ta.text, "தமிழ் உரை")

        res_hi = router.route_and_recognize(self.sample_img, language="devanagari")
        self.assertEqual(res_hi.text, "हिंदी पाठ")

    def test_router_fallback_on_unregistered_language(self):
        """Tests that unknown language falls back to default recognizer."""
        router = LanguageScriptRouter(
            default_language="kannada",
            auto_register_kannada=True,
            kannada_recognizer=self.kannada_rec,
        )

        # French is not registered, should fallback to default (Kannada)
        rec = router.get_recognizer(language="french")
        self.assertIs(rec, self.kannada_rec)

    def test_router_type_validation(self):
        """Tests rejection of invalid non-recognizer objects."""
        router = LanguageScriptRouter(auto_register_kannada=False)
        with self.assertRaises(TypeError):
            router.register_recognizer("kannada", "invalid_object")  # type: ignore


if __name__ == "__main__":
    unittest.main()
