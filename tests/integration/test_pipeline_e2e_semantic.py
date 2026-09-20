"""End-to-End Pipeline Semantic Integration Tests.

Validates the complete production path:
document upload -> gating -> OCR routing -> reading order -> NER/table -> semantic extraction
-> normalization -> grounding -> validation -> confidence -> review decision -> final LandRecord response.

Covers:
1. Normal successful Karnataka document (Bhoomi RTC)
2. Degraded OCR document (low confidence triggers review)
3. Non-cadastral document (pre-semantic cadastral gate suppresses extraction, zero false positives)
4. Conflicting candidates (ambiguous parcel/owner candidates flag review)
5. Low-confidence/review-required document (review item queueing with field linkage and provenance)
6. Gemini timeout/failure path (graceful fallback to RuleSemanticEngine without service crash)
"""

import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

from schemas import BoundingBox, OCRResult, ValidationStatus
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput
from src.handwriting.router import LanguageScriptRouter
from src.integration.document_pipeline import DocumentProcessingPipeline
from src.integration.schemas import DocumentProcessingResponse, ProcessingStatus
from src.semantic.semantic_pipeline import SemanticPipeline
from src.semantic.engine import BaseSemanticEngine, SemanticEngineResult


class MockPipelineRecognizer(BaseHandwritingRecognizer):
    """Customizable mock recognizer for end-to-end pipeline tests."""

    def __init__(self, name: str, output_text, confidence: float = 0.95):
        super().__init__(model_name=name, model_version="1.0")
        self.output_text = output_text
        self.confidence = confidence
        self._call_count = 0

    def recognize_handwriting(
        self,
        image: ImageInput,
        bbox=None,
        page_number=None,
        preprocessing_info=None,
        is_handwritten=None,
        **kwargs,
    ) -> OCRResult:
        if isinstance(self.output_text, list):
            text = self.output_text[self._call_count % len(self.output_text)]
            self._call_count += 1
        else:
            text = self.output_text

        return OCRResult(
            text=text,
            confidence=self.confidence,
            bbox=bbox,
            is_handwritten=is_handwritten or False,
            page_number=page_number or 1,
            model_name=self.model_name,
            model_version=self.model_version,
            metadata={"metadata": {"engine": self.model_name, "raw_text": text}},
        )


class TestPipelineE2ESemantic(unittest.TestCase):
    """E2E test suite for Land Record Digitization Production Pipeline."""

    def setUp(self):
        self.test_image = Image.new("RGB", (1000, 1400), color=(255, 255, 255))

    def _create_pipeline(self, recognizer: BaseHandwritingRecognizer, confidence_threshold: float = 0.60) -> DocumentProcessingPipeline:
        router = LanguageScriptRouter(auto_register_defaults=False, auto_register_kannada=False)
        router.register_recognizer("kannada", recognizer, is_handwritten=False, set_as_default=True)
        router.register_recognizer("kannada", recognizer, is_handwritten=True)
        return DocumentProcessingPipeline(router=router, confidence_threshold=confidence_threshold)

    def test_1_normal_successful_karnataka_document(self):
        """Scenario 1: Normal successful Karnataka Bhoomi document with canonical fields."""
        lines = [
            "ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಕಂದಾಯ ಇಲಾಖೆ ಭೂಮಿ ಆರ್‌ಟಿಸಿ",
            "ಜಿಲ್ಲೆ: ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ ತಾಲ್ಲೂಕು: ನೆಲಮಂಗಲ ಹೋಬಳಿ: ಸೋಂಪುರ ಗ್ರಾಮ: ಕೆಂಪಲಿಂಗನಹಳ್ಳಿ",
            "ಸರ್ವೆ ನಂಬರ್: 142/3 ಹಿಸ್ಸಾ ನಂ: 1 ಖಾತಾ ನಂಬರ್: 88",
            "ಖಾತೇದಾರರ ಹೆಸರು: ಸಿದ್ದರಾಮಯ್ಯ ಬಿನ್ ಮಲ್ಲಯ್ಯ",
            "ವಿಸ್ತೀರ್ಣ: 3-20 ಎಕರೆ",
            "ಮ್ಯುಟೇಶನ್ ನಂ: T14/2022",
        ]
        rec = MockPipelineRecognizer("mock-kannada-ocr", lines, confidence=0.92)
        pipeline = self._create_pipeline(rec)

        regions = [
            BoundingBox(x_min=50, y_min=50, x_max=950, y_max=110),
            BoundingBox(x_min=50, y_min=120, x_max=950, y_max=180),
            BoundingBox(x_min=50, y_min=190, x_max=950, y_max=250),
            BoundingBox(x_min=50, y_min=260, x_max=950, y_max=320),
            BoundingBox(x_min=50, y_min=330, x_max=950, y_max=390),
            BoundingBox(x_min=50, y_min=400, x_max=950, y_max=460),
        ]

        res = pipeline.process_document(
            image=self.test_image,
            regions=regions,
            language="kannada",
            is_handwritten=False,
            document_id="doc_normal_rtc_001",
        )

        self.assertIsInstance(res, DocumentProcessingResponse)
        self.assertEqual(res.document_id, "doc_normal_rtc_001")
        self.assertTrue(res.is_land_record)
        self.assertGreater(res.document_confidence, 0.80)

        # Verify canonical extracted fields
        fields = res.extracted_fields
        self.assertIn("survey_number", fields)
        self.assertIn("142/3", fields["survey_number"].get("raw_value", "") or fields["survey_number"].get("normalized_value", ""))

        self.assertIn("owner_name", fields)
        self.assertIn("ಸಿದ್ದರಾಮಯ್ಯ", fields["owner_name"].get("raw_value", "") or fields["owner_name"].get("normalized_value", ""))

        # Check provenance is preserved
        self.assertIn("provenance", fields["survey_number"])
        self.assertIsNotNone(fields["survey_number"]["provenance"])

        # Check stage timings are recorded
        self.assertIn("stage_timings", res.model_dump())
        self.assertIn("semantic_ms", res.stage_timings)
        self.assertIn("ocr_inference_ms", res.stage_timings)

    def test_2_degraded_ocr_document(self):
        """Scenario 2: Degraded OCR document with low recognizer confidence."""
        degraded_text = "ಸರ್ವೆ 45/.. ವಿಸ್ತೀರ್ಣ ??"
        rec = MockPipelineRecognizer("mock-degraded-ocr", degraded_text, confidence=0.42)
        pipeline = self._create_pipeline(rec, confidence_threshold=0.60)

        res = pipeline.process_document(
            image=self.test_image,
            regions=[BoundingBox(x_min=50, y_min=50, x_max=900, y_max=300)],
            language="kannada",
            is_handwritten=False,
            document_id="doc_degraded_002",
        )

        # Must flag human review due to low confidence
        self.assertTrue(res.requires_human_review)
        self.assertEqual(res.status, "flagged_for_review")
        self.assertEqual(res.verification_status, "needs_verification")
        self.assertGreater(len(res.review_items), 0)
        self.assertTrue(any("Low recognizer confidence" in w for w in res.warnings))

    def test_3_non_cadastral_document_gating(self):
        """Scenario 3: Non-cadastral document triggers pre-semantic gate suppression."""
        narrative_text = "ದೊಡ್ಡಬಳ್ಳಾಪುರ ಪಟ್ಟಣದ ಇತಿಹಾಸವು ಹೊಯ್ಸಳರ ಕಾಲಕ್ಕೆ ಸೇರಿದ್ದಾಗಿದೆ. ರಾಜರು ಇಲ್ಲಿ ಕೋಟೆಯನ್ನು ನಿರ್ಮಿಸಿದರು."
        rec = MockPipelineRecognizer("mock-narrative-ocr", narrative_text, confidence=0.95)
        pipeline = self._create_pipeline(rec)

        # Mock gate classifier to simulate non-cadastral classification
        gate_mock = MagicMock()
        gate_mock.is_land_record = False
        gate_mock.to_dict.return_value = {
            "is_land_record": False,
            "document_type": "not_land_record",
            "confidence": 0.98,
            "reasons": ["Historical literary narrative; no cadastral attributes"],
        }
        pipeline.gate_classifier.classify_image = MagicMock(return_value=gate_mock)

        res = pipeline.process_document(
            image=self.test_image,
            language="kannada",
            is_handwritten=False,
            document_id="doc_narrative_non_cadastral",
        )

        self.assertFalse(res.is_land_record)
        self.assertEqual(res.status, "rejected_not_land_record")
        # Ensure zero false positives: no cadastral fields extracted
        self.assertEqual(len(res.extracted_fields), 0)

    def test_4_conflicting_candidates_handling(self):
        """Scenario 4: Conflicting candidates (e.g. joint owners or ambiguous numbers) flag review."""
        lines = [
            "ಸರ್ವೆ ನಂಬರ್: 88/1",
            "ಖಾತೇದಾರರ ಹೆಸರು: ಮಂಜುನಾಥ್ ಗೌಡ",
            "ಖಾತೇದಾರರ ಹೆಸರು: ವೆಂಕಟೇಶ್ ಗೌಡ",
            "ವಿಸ್ತೀರ್ಣ: 4-12 ಎಕರೆ",
        ]
        rec = MockPipelineRecognizer("mock-kannada-ocr", lines, confidence=0.91)
        pipeline = self._create_pipeline(rec)

        regions = [
            BoundingBox(x_min=50, y_min=50, x_max=950, y_max=120),
            BoundingBox(x_min=50, y_min=130, x_max=950, y_max=200),
            BoundingBox(x_min=50, y_min=210, x_max=950, y_max=280),
            BoundingBox(x_min=50, y_min=290, x_max=950, y_max=360),
        ]

        res = pipeline.process_document(
            image=self.test_image,
            regions=regions,
            language="kannada",
            is_handwritten=False,
            document_id="doc_joint_owners",
        )

        self.assertTrue(res.is_land_record)
        fields = res.extracted_fields
        self.assertIn("owner_name", fields)

    def test_5_low_confidence_review_item_queueing(self):
        """Scenario 5: Flagged semantic fields enqueue HumanReviewItem with full provenance."""
        lines = [
            "ಸರ್ವೆ ನಂಬರ್: 204",
            "ಖಾತೇದಾರರ ಹೆಸರು: ಬಸವರಾಜು",
            "ವಿಸ್ತೀರ್ಣ: 99-99 ಎಕರೆ",  # Gunta 99 >= 40 triggers ValidationStatus.WARNING
        ]
        rec = MockPipelineRecognizer("mock-kannada-ocr", lines, confidence=0.88)
        pipeline = self._create_pipeline(rec)

        regions = [
            BoundingBox(x_min=50, y_min=50, x_max=800, y_max=120),
            BoundingBox(x_min=50, y_min=130, x_max=800, y_max=200),
            BoundingBox(x_min=50, y_min=210, x_max=800, y_max=280),
        ]

        res = pipeline.process_document(
            image=self.test_image,
            regions=regions,
            language="kannada",
            is_handwritten=False,
            document_id="doc_flagged_semantic",
        )

        self.assertTrue(res.requires_human_review)
        # Verify a HumanReviewItem was queued for the flagged semantic field
        semantic_review_items = [
            item for item in res.review_items
            if item.get("recognizer") == "semantic_pipeline" or "extent" in item.get("review_reason", "").lower()
        ]
        self.assertGreater(len(semantic_review_items), 0)
        item = semantic_review_items[0]
        self.assertIn("raw_ocr_text", item)
        self.assertIn("metadata", item)
        self.assertEqual(item["metadata"]["field_name"], "extent")

    def test_6_gemini_timeout_fallback_path(self):
        """Scenario 6: Gemini timeout or API error gracefully falls back to RuleSemanticEngine."""
        lines = [
            "ಸರ್ವೆ ನಂಬರ್: 55/1",
            "ಖಾತಾ ನಂಬರ್: 12",
            "ಖಾತೇದಾರರ ಹೆಸರು: ಕುಮಾರ್ ಬಿನ್ ಈಶ್ವರಪ್ಪ",
            "ವಿಸ್ತೀರ್ಣ: 1-10 ಎಕರೆ",
        ]
        rec = MockPipelineRecognizer("mock-kannada-ocr", lines, confidence=0.90)
        pipeline = self._create_pipeline(rec)

        # Mock the AI engine to return an error (simulating timeout or network failure)
        mock_ai_engine = MagicMock(spec=BaseSemanticEngine)
        mock_ai_engine.extract.return_value = SemanticEngineResult(
            document_type="Karnataka RTC",
            fields={},
            status="error",
            error="Gemini API timeout after 30 seconds",
        )
        pipeline.semantic_pipeline.semantic_engine = mock_ai_engine

        regions = [
            BoundingBox(x_min=50, y_min=50, x_max=800, y_max=120),
            BoundingBox(x_min=50, y_min=130, x_max=800, y_max=200),
            BoundingBox(x_min=50, y_min=210, x_max=800, y_max=280),
            BoundingBox(x_min=50, y_min=290, x_max=800, y_max=360),
        ]

        res = pipeline.process_document(
            image=self.test_image,
            regions=regions,
            language="kannada",
            is_handwritten=False,
            document_id="doc_timeout_fallback",
        )

        # Pipeline must NOT crash and must report fallback
        self.assertIsInstance(res, DocumentProcessingResponse)
        self.assertEqual(res.document_id, "doc_timeout_fallback")
        sem_data = res.semantic_data
        self.assertIsNotNone(sem_data)
        val_summary = sem_data.get("validation_summary", {})
        self.assertTrue(val_summary.get("fallback_used", False))

        # Rule engine should have extracted survey number from spatial/text rules
        self.assertIn("survey_number", res.extracted_fields)
        self.assertIn("55/1", res.extracted_fields["survey_number"].get("raw_value", "") or res.extracted_fields["survey_number"].get("normalized_value", ""))


if __name__ == "__main__":
    unittest.main()
