"""Unit tests for Pretrained TrOCR Handwriting Recognition Backend.

Tests TrocrHandwritingRecognizer using mock objects and dependency injection
so that all unit tests execute immediately and deterministically without downloading
heavy checkpoints over the network.
"""

import os
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas import BoundingBox, OCRResult
from src.handwriting.recognizer import BaseHandwritingRecognizer
from src.handwriting.router import LanguageScriptRouter
from src.handwriting.service import HandwritingOCRService
from src.handwriting.trocr_recognizer import TrocrHandwritingRecognizer


class TestTrocrHandwritingRecognizer(TestCase):
    """Unit test suite for TrOCR handwriting recognition backend."""

    def setUp(self):
        """Creates synthetic test sample image for inference testing."""
        self.sample_img = Image.new("RGB", (200, 50), color=(255, 255, 255))
        self.sample_bbox = BoundingBox(x_min=10, y_min=10, x_max=190, y_max=40)

    def test_implements_base_interface(self):
        """Verifies that TrocrHandwritingRecognizer satisfies BaseHandwritingRecognizer."""
        recognizer = TrocrHandwritingRecognizer(
            model_name_or_path="microsoft/trocr-small-handwritten",
            model_version="trocr_small_v1",
            auto_load=False,
        )
        self.assertIsInstance(recognizer, BaseHandwritingRecognizer)
        self.assertEqual(recognizer.model_name, "microsoft/trocr-small-handwritten")
        self.assertEqual(recognizer.model_version, "trocr_small_v1")
        self.assertIn(recognizer.device, ["cpu", "cuda"])

    def test_lazy_loading_behavior(self):
        """Verifies that model weights are not loaded into memory during init when auto_load=False."""
        recognizer = TrocrHandwritingRecognizer(
            model_name_or_path="microsoft/trocr-small-handwritten",
            auto_load=False,
        )
        self.assertFalse(recognizer.is_loaded)
        self.assertTrue(recognizer.is_available)

    def test_token_probability_extraction_math(self):
        """Tests that step-by-step token probabilities are computed with exact softmax without fabrication."""
        recognizer = TrocrHandwritingRecognizer(auto_load=False)

        # Create synthetic logits: 3 steps, vocab size 5
        # Step 0: token 2 has highest logit
        # Step 1: token 4 has highest logit
        # Step 2: token 1 has highest logit
        step0 = torch.tensor([[0.0, 1.0, 5.0, 0.0, 0.0]])
        step1 = torch.tensor([[0.0, 0.0, 1.0, 0.0, 6.0]])
        step2 = torch.tensor([[0.0, 4.0, 1.0, 0.0, 0.0]])
        scores = [step0, step1, step2]

        # Sequences tensor: [decoder_start_token, 2, 4, 1]
        sequences = torch.tensor([[0, 2, 4, 1]])

        probs = recognizer._extract_token_probabilities(scores=scores, sequences=sequences)

        self.assertEqual(len(probs), 3)
        for p in probs:
            self.assertGreater(p, 0.0)
            self.assertLessEqual(p, 1.0)

        # Expected softmax for step 0 at index 2 should be > 0.9
        self.assertGreater(probs[0], 0.9)
        self.assertGreater(probs[1], 0.9)
        self.assertGreater(probs[2], 0.9)

    def test_mock_inference_success_and_metadata(self):
        """Tests full recognize_handwriting workflow using mock model and processor."""
        mock_processor = MagicMock()
        mock_model = MagicMock()

        # Mock processor return
        mock_processor.return_value.pixel_values = torch.zeros((1, 3, 384, 384))
        mock_processor.batch_decode.return_value = ["Sample Handwritten Text"]

        # Mock model.generate output with sequences and scores
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[0, 2, 0, 1]])
        step0 = torch.tensor([[0.1, 0.2, 0.7]])
        step1 = torch.tensor([[0.8, 0.1, 0.1]])
        step2 = torch.tensor([[0.05, 0.9, 0.05]])
        mock_outputs.scores = [step0, step1, step2]
        mock_model.generate.return_value = mock_outputs

        recognizer = TrocrHandwritingRecognizer(
            model_name_or_path="microsoft/trocr-small-handwritten",
            model_version="trocr_small_v1",
            processor=mock_processor,
            model=mock_model,
            preprocess_input=True,
        )
        self.assertTrue(recognizer.is_loaded)

        result = recognizer.recognize_handwriting(
            image=self.sample_img,
            bbox=self.sample_bbox,
            page_number=1,
        )

        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.text, "Sample Handwritten Text")
        self.assertTrue(result.is_handwritten)
        self.assertEqual(result.bbox, self.sample_bbox)
        self.assertEqual(result.page_number, 1)
        self.assertEqual(result.model_name, "microsoft/trocr-small-handwritten")

        # Confidence verification
        self.assertIsNotNone(result.confidence)
        self.assertIsInstance(result.confidence, float)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

        # Audit trail verification
        audit = result.metadata
        self.assertEqual(audit["calculation_method"], "trocr_step_softmax_mean")
        self.assertEqual(audit["metadata"]["engine_status"], "active")
        self.assertIn("language_limitation_notice", audit["metadata"])
        self.assertIn("preprocessing_pipeline", audit["metadata"]["preprocessing"])

    def test_mock_inference_error_handling(self):
        """Verifies that exceptions during TrOCR inference return transparent error metadata."""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor.batch_decode.return_value = [""]
        mock_processor.return_value.pixel_values = torch.zeros((1, 3, 384, 384))
        mock_model.generate.side_effect = RuntimeError("CUDA out of memory simulation")

        recognizer = TrocrHandwritingRecognizer(
            processor=mock_processor,
            model=mock_model,
        )

        result = recognizer.recognize_handwriting(image=self.sample_img)

        self.assertEqual(result.text, "")
        self.assertIsNone(result.confidence)
        self.assertEqual(result.metadata["metadata"]["engine_status"], "error")
        self.assertIn("CUDA out of memory", result.metadata["metadata"]["error"])

    def test_batch_recognition(self):
        """Tests recognize_batch method across multiple crops."""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor.return_value.pixel_values = torch.zeros((1, 3, 384, 384))
        mock_processor.batch_decode.side_effect = [["First Line"], ["Second Line"]]

        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[0, 10]])
        mock_outputs.scores = [torch.tensor([[0.1, 0.9]])]
        mock_model.generate.return_value = mock_outputs

        recognizer = TrocrHandwritingRecognizer(
            processor=mock_processor,
            model=mock_model,
        )

        results = recognizer.recognize_batch(
            images=[self.sample_img, self.sample_img],
            bboxes=[self.sample_bbox, self.sample_bbox],
            page_numbers=[1, 1],
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].text, "First Line")
        self.assertEqual(results[1].text, "Second Line")

    def test_integration_with_handwriting_service(self):
        """Verifies that TrocrHandwritingRecognizer integrates seamlessly with HandwritingOCRService."""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor.return_value.pixel_values = torch.zeros((1, 3, 384, 384))
        mock_processor.batch_decode.return_value = ["Owner Signature"]
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[0, 5]])
        mock_outputs.scores = [torch.tensor([[0.2, 0.8]])]
        mock_model.generate.return_value = mock_outputs

        recognizer = TrocrHandwritingRecognizer(
            processor=mock_processor,
            model=mock_model,
        )

        service = HandwritingOCRService(recognizer=recognizer)
        res = service.process_crop(self.sample_img, bbox=self.sample_bbox, page_number=2)

        self.assertEqual(res.text, "Owner Signature")
        self.assertEqual(res.page_number, 2)
        self.assertTrue(res.is_handwritten)

    def test_integration_with_script_router(self):
        """Verifies that TrocrHandwritingRecognizer can be registered and routed in LanguageScriptRouter."""
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor.return_value.pixel_values = torch.zeros((1, 3, 384, 384))
        mock_processor.batch_decode.return_value = ["Survey 42/A"]
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[0, 5]])
        mock_outputs.scores = [torch.tensor([[0.2, 0.8]])]
        mock_model.generate.return_value = mock_outputs

        trocr = TrocrHandwritingRecognizer(
            processor=mock_processor,
            model=mock_model,
            language="en",
        )

        router = LanguageScriptRouter(auto_register_kannada=False)
        router.register_recognizer("english", trocr, set_as_default=True)

        routed_res = router.route_and_recognize(self.sample_img, language="en")
        self.assertEqual(routed_res.text, "Survey 42/A")
        self.assertEqual(routed_res.model_name, trocr.model_name)


if __name__ == "__main__":
    from unittest import main
    main()
