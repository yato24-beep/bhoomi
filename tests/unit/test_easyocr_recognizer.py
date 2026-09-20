"""Unit tests for EasyOCRKannadaRecognizer and printed engine routing."""

import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from schemas import BoundingBox, OCREngineType, OCRResult
from src.handwriting.easyocr_recognizer import EasyOCRKannadaRecognizer
from src.handwriting.paddle_recognizer import PaddleKannadaRecognizer
from src.handwriting.router import LanguageScriptRouter


class TestEasyOCRRecognizer(unittest.TestCase):
    """Tests for EasyOCRKannadaRecognizer."""

    def test_recognizer_metadata(self):
        rec = EasyOCRKannadaRecognizer(languages=["kn", "en"])
        self.assertEqual(rec.model_name, "easyocr-kannada")
        self.assertEqual(rec.model_version, "1.7.2")
        self.assertEqual(rec.languages, ("kn", "en"))

    def test_reading_order_sorting(self):
        rec = EasyOCRKannadaRecognizer()
        # Create boxes: line 1 (left, right), line 2 (left, right)
        detections = [
            {"text": "Line1_Right", "bbox": BoundingBox(x_min=100, y_min=10, x_max=180, y_max=30), "poly": []},
            {"text": "Line1_Left", "bbox": BoundingBox(x_min=10, y_min=10, x_max=90, y_max=30), "poly": []},
            {"text": "Line2_Right", "bbox": BoundingBox(x_min=100, y_min=50, x_max=180, y_max=70), "poly": []},
            {"text": "Line2_Left", "bbox": BoundingBox(x_min=10, y_min=50, x_max=90, y_max=70), "poly": []},
        ]
        sorted_dets = rec._sort_reading_order(detections)
        ordered_texts = [d["text"] for d in sorted_dets]
        self.assertEqual(ordered_texts, ["Line1_Left", "Line1_Right", "Line2_Left", "Line2_Right"])

    @patch("src.handwriting.easyocr_recognizer.get_easyocr_reader")
    def test_mock_inference(self, mock_get_reader):
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[10, 10], [90, 10], [90, 30], [10, 30]], "ದೊಡ್ಡಬಳ್ಳಾಪುರ", 0.95),
            ([[100, 10], [180, 10], [180, 30], [100, 30]], "ತಾಲೂಕು", 0.92),
        ]
        mock_get_reader.return_value = mock_reader

        rec = EasyOCRKannadaRecognizer(languages=["kn", "en"])
        dummy_img = Image.new("RGB", (200, 50), color=(255, 255, 255))
        res = rec.recognize_handwriting(dummy_img)

        self.assertIsInstance(res, OCRResult)
        self.assertIn("ದೊಡ್ಡಬಳ್ಳಾಪುರ", res.text)
        self.assertIn("ತಾಲೂಕು", res.text)
        self.assertGreater(res.confidence, 0.90)
        self.assertEqual(res.model_name, "easyocr-kannada")
        self.assertEqual(len(res.metadata["metadata"]["line_details"]), 2)


class TestRouterEngineSelection(unittest.TestCase):
    """Tests for dynamic printed OCR engine selection in LanguageScriptRouter."""

    def test_default_router_selects_easyocr(self):
        with patch.dict(os.environ, {"PRINTED_OCR_ENGINE": "easyocr"}):
            router = LanguageScriptRouter(auto_register_kannada=True)
            printed_rec = router.get_recognizer(language="kannada", is_handwritten=False)
            self.assertIsInstance(printed_rec, EasyOCRKannadaRecognizer)

    def test_router_selects_paddleocr_when_configured(self):
        with patch.dict(os.environ, {"PRINTED_OCR_ENGINE": "paddleocr"}):
            router = LanguageScriptRouter(auto_register_kannada=True)
            printed_rec = router.get_recognizer(language="kannada", is_handwritten=False)
            self.assertIsInstance(printed_rec, PaddleKannadaRecognizer)

    def test_router_preserves_handwritten_routing(self):
        router = LanguageScriptRouter(auto_register_kannada=True)
        hw_rec = router.get_recognizer(language="kannada", is_handwritten=True)
        # Should be TrOCR instance
        self.assertNotIsInstance(hw_rec, EasyOCRKannadaRecognizer)
        self.assertNotIsInstance(hw_rec, PaddleKannadaRecognizer)


if __name__ == "__main__":
    unittest.main()
