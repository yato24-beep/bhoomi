"""Unit tests for PaddleKannadaRecognizer with dependency injection and mock engines."""

import unittest
from unittest.mock import MagicMock
from PIL import Image

from schemas import BoundingBox, OCRResult
from src.handwriting.paddle_recognizer import PaddleKannadaRecognizer


class TestPaddleKannadaRecognizer(unittest.TestCase):
    """Test suite for PaddleKannadaRecognizer."""

    def setUp(self):
        """Creates dummy PIL crop image for tests."""
        self.sample_img = Image.new("RGB", (150, 40), color=(255, 255, 255))

    def test_mock_engine_successful_recognition(self):
        """Tests parsing real PaddleOCR format output with mock engine."""
        # Simulated PaddleOCR format: [ [ [polygon, (text, confidence)], ... ] ]
        mock_paddle = MagicMock()
        mock_paddle.ocr.return_value = [
            [
                [[[10, 5], [140, 5], [140, 35], [10, 35]], ("ಖಾತೆ ಸಂಖ್ಯೆ ೧೨೩", 0.945)],
            ]
        ]

        recognizer = PaddleKannadaRecognizer(
            model_name="test-paddle-kannada",
            model_version="v4-test",
            paddle_engine=mock_paddle,
            preprocess_input=False,
        )

        self.assertTrue(recognizer.is_available)
        result = recognizer.recognize_handwriting(
            image=self.sample_img,
            page_number=1,
        )

        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.text, "ಖಾತೆ ಸಂಖ್ಯೆ ೧೨೩")
        self.assertEqual(result.confidence, 0.945)
        self.assertEqual(result.model_name, "test-paddle-kannada")
        self.assertEqual(result.page_number, 1)
        self.assertIsNotNone(result.bbox)
        self.assertEqual(result.bbox.x_min, 10)
        self.assertEqual(result.bbox.y_min, 5)
        self.assertEqual(result.bbox.x_max, 140)
        self.assertEqual(result.bbox.y_max, 35)
        self.assertEqual(result.metadata["detected_lines_count"], 1)
        self.assertEqual(result.metadata["calculation_method"], "paddle_line_mean")

    def test_mock_engine_multi_line_recognition(self):
        """Tests multi-line recognition and mean confidence calculation."""
        mock_paddle = MagicMock()
        mock_paddle.ocr.return_value = [
            [
                [[[10, 5], [100, 5], [100, 20], [10, 20]], ("ಸರ್ವೆ ನಂಬರ್", 0.960)],
                [[[10, 25], [100, 25], [100, 40], [10, 40]], ("೪೫/೨", 0.920)],
            ]
        ]

        recognizer = PaddleKannadaRecognizer(
            paddle_engine=mock_paddle,
            preprocess_input=False,
        )

        result = recognizer.recognize_handwriting(image=self.sample_img)
        self.assertEqual(result.text, "ಸರ್ವೆ ನಂಬರ್ ೪೫/೨")
        self.assertEqual(result.confidence, 0.940)  # (0.960 + 0.920) / 2
        self.assertEqual(result.metadata["detected_lines_count"], 2)

    def test_mock_engine_empty_text_result(self):
        """Tests handling when OCR detects no text regions."""
        mock_paddle = MagicMock()
        mock_paddle.ocr.return_value = [[]]

        recognizer = PaddleKannadaRecognizer(
            paddle_engine=mock_paddle,
            preprocess_input=False,
        )

        result = recognizer.recognize_handwriting(image=self.sample_img)
        self.assertEqual(result.text, "")
        self.assertIsNone(result.confidence)
        self.assertEqual(result.metadata["detected_lines_count"], 0)

    def test_unavailable_engine_graceful_fallback(self):
        """Tests that missing engine returns auditable non-fabricated output with confidence=None."""
        # Force recognizer to have no paddle engine
        recognizer = PaddleKannadaRecognizer(
            paddle_engine=None,
            preprocess_input=False,
        )
        # Explicitly set unavailable state to verify fallback path
        recognizer._engine_available = False
        recognizer._engine = None
        recognizer._init_error = "PaddleOCR not installed in test environment"

        result = recognizer.recognize_handwriting(
            image=self.sample_img,
            bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=20),
        )

        self.assertEqual(result.text, "")
        self.assertIsNone(result.confidence)
        self.assertEqual(result.metadata["engine_status"], "unavailable")
        self.assertIn("PaddleOCR not installed", result.metadata["reason"])
        self.assertEqual(result.metadata["has_token_probabilities"], False)

    def test_engine_inference_error_handling(self):
        """Tests exception safety if inference engine crashes during execution."""
        mock_paddle = MagicMock()
        mock_paddle.ocr.side_effect = RuntimeError("CUDA out of memory or corrupt tensor")

        recognizer = PaddleKannadaRecognizer(
            paddle_engine=mock_paddle,
            preprocess_input=False,
        )

        result = recognizer.recognize_handwriting(image=self.sample_img)
        self.assertEqual(result.text, "")
        self.assertIsNone(result.confidence)
        self.assertEqual(result.metadata["engine_status"], "error")
        self.assertIn("CUDA out of memory", result.metadata["error"])


if __name__ == "__main__":
    unittest.main()
