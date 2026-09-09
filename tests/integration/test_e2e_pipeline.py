"""End-to-end integration test verifying all 4 primary OCR cases, fallbacks, and response contract."""

import unittest
from PIL import Image

from schemas import BoundingBox, DocumentPage, OCRResult
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput
from src.handwriting.router import LanguageScriptRouter
from src.integration.document_pipeline import DocumentProcessingPipeline, process_document
from src.integration.schemas import DocumentProcessingResponse, ProcessingStatus


class E2EMockRecognizer(BaseHandwritingRecognizer):
    """Mock recognizer simulating specific engine outputs for E2E testing."""

    def __init__(self, name: str, version: str, output_text: str, confidence: float = 0.95):
        super().__init__(model_name=name, model_version=version)
        self.output_text = output_text
        self.confidence = confidence

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
            confidence=self.confidence,
            bbox=bbox,
            is_handwritten=is_handwritten,
            page_number=page_number or 1,
            model_name=self.model_name,
            model_version=self.model_version,
            metadata={"metadata": {"engine": self.model_name, "raw_text": self.output_text}},
        )


class TestEndToEndPipeline(unittest.TestCase):
    """E2E test suite verifying the 4 primary pipeline scenarios."""

    def setUp(self):
        self.test_img = Image.new("RGB", (800, 1000), color=(255, 255, 255))

        # Setup recognizers for 4 cases
        self.kn_printed_rec = E2EMockRecognizer("paddleocr-kannada", "v4", "ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಕಂದಾಯ ಇಲಾಖೆ", 0.96)
        self.kn_hw_rec = E2EMockRecognizer("trocr-kannada-finetuned", "v1.0", "ಸರ್ವೆ ನಂ 105/2 ವಿಸ್ತೀರ್ಣ 2 ಎಕರೆ", 0.89)
        self.en_printed_rec = E2EMockRecognizer("paddleocr-english", "v4", "Government of Karnataka Land Record", 0.98)
        self.en_hw_rec = E2EMockRecognizer("trocr-english-base", "v1.0", "Owner: Ramesh Kumar S/O Somanna", 0.91)

        self.router = LanguageScriptRouter(auto_register_defaults=False, auto_register_kannada=False)
        self.router.register_recognizer("kannada", self.kn_printed_rec, is_handwritten=False, set_as_default=True)
        self.router.register_recognizer("kannada", self.kn_hw_rec, is_handwritten=True)
        self.router.register_recognizer("english", self.en_printed_rec, is_handwritten=False)
        self.router.register_recognizer("english", self.en_hw_rec, is_handwritten=True)

        self.pipeline = DocumentProcessingPipeline(router=self.router, confidence_threshold=0.60)

    def test_case_1_printed_kannada(self):
        """Case 1: Printed Kannada -> PaddleOCR Kannada -> High Confidence Text."""
        res = self.pipeline.process_document(
            image=self.test_img,
            regions=[BoundingBox(x_min=50, y_min=50, x_max=750, y_max=120)],
            language="kannada",
            is_handwritten=False,
        )
        self.assertEqual(res.ordered_regions[0].model_name, "paddleocr-kannada")
        self.assertIn("ಕರ್ನಾಟಕ ಸರ್ಕಾರ", res.merged_text)
        self.assertFalse(res.requires_human_review)

    def test_case_2_handwritten_kannada(self):
        """Case 2: Handwritten Kannada -> TrOCR Kannada -> High Confidence Text."""
        res = self.pipeline.process_document(
            image=self.test_img,
            regions=[BoundingBox(x_min=50, y_min=200, x_max=750, y_max=280)],
            language="kannada",
            is_handwritten=True,
        )
        self.assertEqual(res.ordered_regions[0].model_name, "trocr-kannada-finetuned")
        self.assertIn("ಸರ್ವೆ ನಂ", res.merged_text)
        self.assertFalse(res.requires_human_review)

    def test_case_3_printed_english(self):
        """Case 3: Printed English -> PaddleOCR English -> High Confidence Text."""
        res = self.pipeline.process_document(
            image=self.test_img,
            regions=[BoundingBox(x_min=50, y_min=50, x_max=750, y_max=120)],
            language="english",
            is_handwritten=False,
        )
        self.assertEqual(res.ordered_regions[0].model_name, "paddleocr-english")
        self.assertIn("Government of Karnataka", res.merged_text)
        self.assertFalse(res.requires_human_review)

    def test_case_4_handwritten_english(self):
        """Case 4: Handwritten English -> TrOCR English -> High Confidence Text."""
        res = self.pipeline.process_document(
            image=self.test_img,
            regions=[BoundingBox(x_min=50, y_min=300, x_max=750, y_max=380)],
            language="english",
            is_handwritten=True,
        )
        self.assertEqual(res.ordered_regions[0].model_name, "trocr-english-base")
        self.assertIn("Owner: Ramesh Kumar", res.merged_text)
        self.assertFalse(res.requires_human_review)

    def test_case_5_unsupported_handwriting_fallback(self):
        """Unsupported handwritten language -> Safe human review trigger without text fabrication."""
        res = self.pipeline.process_document(
            image=self.test_img,
            regions=[BoundingBox(x_min=50, y_min=500, x_max=750, y_max=580)],
            language="telugu",
            is_handwritten=True,
        )
        self.assertTrue(res.requires_human_review)
        self.assertEqual(res.ordered_regions[0].raw_text, "")
        self.assertEqual(res.ordered_regions[0].model_name, "none")
        self.assertTrue(len(res.warnings) > 0)
        self.assertIsNotNone(res.review_reason)
        self.assertTrue(len(res.review_reasons) > 0)

    def test_composite_multi_region_document(self):
        """Multi-region document with mixed scripts, reading order sort, and dictionary serialization."""
        regions = [
            # Intentionally out of vertical order
            {"bbox": (50, 400, 750, 480), "language": "english", "is_handwritten": True},
            {"bbox": (50, 50, 750, 120), "language": "kannada", "is_handwritten": False},
            {"bbox": (50, 200, 750, 280), "language": "kannada", "is_handwritten": True},
        ]
        res = self.pipeline.process_document(
            image=self.test_img,
            regions=regions,
            document_id="doc_test_001",
            page_number=1,
        )

        self.assertEqual(res.document_id, "doc_test_001")
        self.assertEqual(len(res.ordered_regions), 3)
        # Top-to-bottom reading order check
        self.assertIn("ಕರ್ನಾಟಕ ಸರ್ಕಾರ", res.ordered_regions[0].raw_text)
        self.assertIn("ಸರ್ವೆ ನಂ", res.ordered_regions[1].raw_text)
        self.assertIn("Owner: Ramesh Kumar", res.ordered_regions[2].raw_text)

        # Serialization checks
        d = res.to_dict()
        self.assertEqual(d["document_id"], "doc_test_001")
        self.assertEqual(d["page_number"], 1)
        self.assertEqual(res.full_text, res.merged_text)
        self.assertIsInstance(res.document_confidence, float)


if __name__ == "__main__":
    unittest.main()
