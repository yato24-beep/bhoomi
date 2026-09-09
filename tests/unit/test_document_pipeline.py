"""Unit tests for Document Processing Pipeline and Orchestration."""

import unittest
from unittest.mock import MagicMock

from PIL import Image

from schemas import BoundingBox, DocumentPage, OCRResult
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput
from src.handwriting.router import LanguageScriptRouter
from src.integration.document_pipeline import (
    DocumentProcessingPipeline,
    DocumentProcessingResult,
    process_document,
)

class MockPipelineRecognizer(BaseHandwritingRecognizer):
    """Mock recognizer for testing pipeline orchestration and confidence computation."""

    def __init__(self, name: str, version: str, text_map=None, default_conf: float = 0.90):
        super().__init__(model_name=name, model_version=version)
        self.text_map = text_map or {}
        self.default_conf = default_conf

    def recognize_handwriting(
        self,
        image: ImageInput,
        bbox=None,
        page_number=None,
        preprocessing_info=None,
        is_handwritten=None,
        **kwargs,
    ) -> OCRResult:
        key = (bbox.x_min, bbox.y_min) if bbox else (0, 0)
        text, conf = self.text_map.get(key, ("Default Text", self.default_conf))

        return OCRResult(
            text=text,
            confidence=conf,
            bbox=bbox,
            is_handwritten=is_handwritten,
            page_number=page_number,
            model_name=self.model_name,
            model_version=self.model_version,
            metadata={"metadata": {"engine": self.model_name, "raw_text": text}},
        )


class TestDocumentProcessingPipeline(unittest.TestCase):
    """Unit tests for DocumentProcessingPipeline orchestration."""

    def setUp(self):
        self.sample_page_img = Image.new("RGB", (400, 600), color=(255, 255, 255))

        # Setup router with mock recognizers
        self.text_map = {
            (20, 20): ("ಕಂದಾಯ ದಾಖಲೆ", 0.92),  # Land record header (top)
            (20, 100): ("ಸರ್ವೆ ನಂಬರ್ 42", 0.88),  # Survey number (middle)
            (20, 300): ("ಮಾಲೀಕರ ಹೆಸರು: ರಮೇಶ್", 0.95),  # Owner name (bottom)
        }

        self.mock_rec = MockPipelineRecognizer(
            name="mock-trocr-kannada",
            version="v1.0",
            text_map=self.text_map,
        )

        self.router = LanguageScriptRouter(auto_register_defaults=False, auto_register_kannada=False)
        self.router.register_recognizer("kannada", self.mock_rec, is_handwritten=True, set_as_default=True)
        self.router.register_recognizer("kannada", self.mock_rec, is_handwritten=False)

        self.pipeline = DocumentProcessingPipeline(
            router=self.router,
            default_language="kannada",
            confidence_threshold=0.60,
            apply_preprocessing=False,
        )

    def test_full_image_mode_without_explicit_regions(self):
        """Tests processing an entire document image as a single region."""
        res = self.pipeline.process_document(
            image=self.sample_page_img,
            is_handwritten=True,
            language="kannada",
            page_number=1,
        )

        self.assertIsInstance(res, DocumentProcessingResult)
        self.assertIsInstance(res.page, DocumentPage)
        self.assertEqual(len(res.page.ocr_results), 1)
        self.assertEqual(res.page.ocr_results[0].model_name, "mock-trocr-kannada")
        self.assertEqual(res.document_confidence, 0.90)
        self.assertFalse(res.requires_human_review)

    def test_multi_region_reading_order_sorting(self):
        """Tests that regions supplied out of order are correctly sorted top-to-bottom in reading order."""
        # Unordered regions: bottom first, then top, then middle
        regions = [
            BoundingBox(x_min=20, y_min=300, x_max=380, y_max=340),  # Line 3
            BoundingBox(x_min=20, y_min=20, x_max=380, y_max=60),    # Line 1
            BoundingBox(x_min=20, y_min=100, x_max=380, y_max=140),  # Line 2
        ]

        res = self.pipeline.process_document(
            image=self.sample_page_img,
            regions=regions,
            is_handwritten=True,
            language="kannada",
        )

        lines = res.full_text.split("\n")
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "ಕಂದಾಯ ದಾಖಲೆ")
        self.assertEqual(lines[1], "ಸರ್ವೆ ನಂಬರ್ 42")
        self.assertEqual(lines[2], "ಮಾಲೀಕರ ಹೆಸರು: ರಮೇಶ್")

    def test_document_confidence_aggregation_weighted_by_length(self):
        """Tests that document confidence is derived accurately as a weighted mean of character lengths."""
        regions = [
            BoundingBox(x_min=20, y_min=20, x_max=380, y_max=60),
            BoundingBox(x_min=20, y_min=100, x_max=380, y_max=140),
        ]

        res = self.pipeline.process_document(
            image=self.sample_page_img,
            regions=regions,
            is_handwritten=True,
            language="kannada",
        )

        # "ಕಂದಾಯ ದಾಖಲೆ" length = 11, conf = 0.92 -> 11 * 0.92 = 10.12
        # "ಸರ್ವೆ ನಂಬರ್ 42" length = 14, conf = 0.88 -> 14 * 0.88 = 12.32
        # Total weight = 25, weighted sum = 22.44 -> 22.44 / 25 = 0.8976
        self.assertAlmostEqual(res.document_confidence, 0.8976, places=3)

    def test_low_confidence_region_triggers_human_review(self):
        """Tests that any low confidence region triggers document-level review flag."""
        low_conf_rec = MockPipelineRecognizer(
            name="mock-low-conf",
            version="v1.0",
            text_map={(20, 20): ("ಅಸ್ಪಷ್ಟ ಅಕ್ಷರ", 0.35)},
        )
        router = LanguageScriptRouter(auto_register_defaults=False, auto_register_kannada=False)
        router.register_recognizer("kannada", low_conf_rec, is_handwritten=True, set_as_default=True)

        pipe = DocumentProcessingPipeline(router=router, confidence_threshold=0.60)
        res = pipe.process_document(
            image=self.sample_page_img,
            regions=[BoundingBox(x_min=20, y_min=20, x_max=100, y_max=40)],
            is_handwritten=True,
        )

        self.assertTrue(res.requires_human_review)
        self.assertTrue(any("Low confidence" in r for r in res.review_reasons))

    def test_process_document_functional_api(self):
        """Tests the public process_document(...) convenience function."""
        res = process_document(
            image=self.sample_page_img,
            pipeline=self.pipeline,
            is_handwritten=True,
            page_number=3,
        )
        self.assertEqual(res.page.page_number, 3)
        self.assertIsInstance(res.to_dict(), dict)


if __name__ == "__main__":
    unittest.main()
