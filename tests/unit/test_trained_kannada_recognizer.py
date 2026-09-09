"""Unit tests for Fine-Tuned Kannada TrOCR Recognizer, Caching, and Metadata."""

import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from schemas import BoundingBox, OCRResult
from src.handwriting.recognizer import BaseHandwritingRecognizer
from src.handwriting.trocr_recognizer import (
    DEFAULT_ENGLISH_MODEL_PATH,
    DEFAULT_KANNADA_MODEL_PATH,
    TrocrHandwritingRecognizer,
    _GLOBAL_MODEL_CACHE,
    get_english_handwriting_recognizer,
    get_kannada_handwriting_recognizer,
    resolve_model_path,
)


class TestTrainedKannadaRecognizer(unittest.TestCase):
    """Unit tests for the trained Kannada TrOCR recognizer."""

    def setUp(self):
        self.sample_img = Image.new("RGB", (200, 50), color=(255, 255, 255))
        self.sample_bbox = BoundingBox(x_min=10, y_min=10, x_max=190, y_max=40)

    def test_default_kannada_path_resolution(self):
        """Tests that Kannada recognizer resolves to the default production checkpoint."""
        rec = TrocrHandwritingRecognizer(language="kannada", auto_load=False)
        self.assertIn("kannada_full_checkpoints", rec.model_name_or_path)
        self.assertEqual(rec.language, "kannada")
        self.assertEqual(rec.script, "Kannada")

    def test_environment_variable_override(self):
        """Tests overriding model path via KANNADA_HANDWRITING_MODEL_PATH environment variable."""
        with patch.dict(os.environ, {"KANNADA_HANDWRITING_MODEL_PATH": "custom/path/checkpoint"}):
            rec = TrocrHandwritingRecognizer(language="kannada", auto_load=False)
            self.assertEqual(rec.model_name_or_path, "custom/path/checkpoint")

    def test_global_model_caching(self):
        """Tests that global cache prevents redundant reloads for identical checkpoint and device."""
        mock_proc = MagicMock()
        mock_model = MagicMock()
        cache_key = "test/mock/checkpoint::cpu::False"
        _GLOBAL_MODEL_CACHE[cache_key] = (mock_proc, mock_model)

        rec = TrocrHandwritingRecognizer(
            model_name_or_path="test/mock/checkpoint",
            device="cpu",
            fp16=False,
            auto_load=True,
        )

        self.assertTrue(rec.is_loaded)
        self.assertIs(rec._processor, mock_proc)
        self.assertIs(rec._model, mock_model)

    def test_inference_metadata_contract(self):
        """Tests that inference outputs all required contract and audit metadata fields."""
        mock_processor = MagicMock()
        mock_model = MagicMock()

        # Mock processor output
        mock_processor.return_value.pixel_values = torch.zeros((1, 3, 384, 384))
        mock_processor.batch_decode.return_value = ["ಭಾರತ"]

        # Mock model generation output
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[0, 1, 1]])
        step0 = torch.tensor([[-5.0, 5.0, -5.0]])
        step1 = torch.tensor([[-5.0, 5.0, -5.0]])
        mock_outputs.scores = [step0, step1]
        mock_model.generate.return_value = mock_outputs

        recognizer = TrocrHandwritingRecognizer(
            model_name_or_path="models/trocr/kannada_full_checkpoints/best_checkpoint",
            model_version="kannada_full_v1",
            processor=mock_processor,
            model=mock_model,
            language="kannada",
            script="Kannada",
            preprocess_input=False,
        )

        result = recognizer.recognize_handwriting(
            image=self.sample_img,
            bbox=self.sample_bbox,
            page_number=1,
        )

        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.text, "ಭಾರತ")
        self.assertTrue(result.is_handwritten)
        self.assertEqual(result.bbox, self.sample_bbox)
        self.assertEqual(result.page_number, 1)

        # Check metadata dictionary
        meta = result.metadata.get("metadata", {})
        self.assertEqual(meta["language"], "kannada")
        self.assertEqual(meta["script"], "Kannada")
        self.assertEqual(meta["raw_text"], "ಭಾರತ")
        self.assertEqual(meta["normalized_text"], "ಭಾರತ")
        self.assertIn("inference_time_ms", meta)
        self.assertIn("inference_time_sec", meta)
        self.assertFalse(meta["requires_human_review"])  # confidence is high (~0.875)

    def test_low_confidence_flags_human_review(self):
        """Tests that low confidence scores set requires_human_review = True."""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor.return_value.pixel_values = torch.zeros((1, 3, 384, 384))
        mock_processor.batch_decode.return_value = ["ಸುಳ್ಳು"]

        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[0, 5]])
        # Very low confidence (0.20)
        mock_outputs.scores = [torch.tensor([[0.80, 0.20]])]
        mock_model.generate.return_value = mock_outputs

        recognizer = TrocrHandwritingRecognizer(
            processor=mock_processor,
            model=mock_model,
            confidence_threshold=0.60,
        )

        result = recognizer.recognize_handwriting(image=self.sample_img)
        meta = result.metadata.get("metadata", {})
        self.assertTrue(meta["requires_human_review"])

    def test_factory_functions(self):
        """Tests get_kannada_handwriting_recognizer and get_english_handwriting_recognizer helpers."""
        kn_rec = get_kannada_handwriting_recognizer(auto_load=False)
        self.assertEqual(kn_rec.language, "kannada")
        self.assertEqual(kn_rec.script, "Kannada")

        en_rec = get_english_handwriting_recognizer(auto_load=False)
        self.assertEqual(en_rec.language, "english")
        self.assertEqual(en_rec.script, "Latin")


if __name__ == "__main__":
    unittest.main()
