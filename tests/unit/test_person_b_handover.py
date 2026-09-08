"""Unit Tests for Person B Handover, Adapter, Normalizer, and Correction Contracts."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from PIL import Image

from schemas import BoundingBox, DocumentPage, OCRResult
from src.correction.schemas import HumanCorrectionRecord
from src.correction.service import CorrectionService
from src.handwriting.recognizer import BaseHandwritingRecognizer
from src.handwriting.router import LanguageScriptRouter
from src.integration.document_pipeline import DocumentProcessingPipeline, process_document
from src.integration.person_a_adapter import PersonAAdapter
from src.integration.schemas import (
    DocumentProcessingRequest,
    DocumentProcessingResponse,
    ProcessingStatus,
    RecognizedRegionResult,
    RegionRequest,
    RegionType,
)
from src.postprocessing.kannada_normalizer import KannadaNormalizer


class MockHandoverRecognizer(BaseHandwritingRecognizer):
    """Mock recognizer returning deterministic text outputs."""
    def __init__(self, name: str, version: str = "v1.0", text: str = "ಪಠ್ಯ", conf: float = 0.90):
        super().__init__(model_name=name, model_version=version)
        self.text = text
        self.conf = conf

    def recognize_handwriting(self, image, bbox=None, page_number=None, **kwargs):
        return OCRResult(
            text=self.text,
            confidence=self.conf,
            bbox=bbox,
            is_handwritten=kwargs.get("is_handwritten", True),
            page_number=page_number or 1,
            model_name=self.model_name,
            model_version=self.model_version,
            metadata={
                "metadata": {
                    "engine_status": "active",
                    "raw_text": self.text,
                    "normalized_text": self.text,
                    "requires_human_review": self.conf < 0.60,
                }
            },
        )


class TestPersonBHandover(unittest.TestCase):
    """Test suite for Person B integration contracts, adapters, normalizers, and corrections."""

    def setUp(self):
        self.sample_img = Image.new("RGB", (600, 400), color=(255, 255, 255))
        self.normalizer = KannadaNormalizer()

    # -------------------------------------------------------------------------
    # 1. PERSON A ADAPTER TESTS
    # -------------------------------------------------------------------------
    def test_person_a_adapter_bbox_validation_and_clamping(self):
        """Tests that varied bbox formats are correctly validated and clamped to image bounds."""
        # 1. Standard tuple
        box1 = PersonAAdapter.validate_and_clamp_bbox((10, 20, 100, 200), (600, 400))
        self.assertEqual(box1.x_min, 10)
        self.assertEqual(box1.y_min, 20)
        self.assertEqual(box1.x_max, 100)
        self.assertEqual(box1.y_max, 200)

        # 2. Out of bounds clamp
        box2 = PersonAAdapter.validate_and_clamp_bbox((-50, -20, 800, 500), (600, 400))
        self.assertEqual(box2.x_min, 0)
        self.assertEqual(box2.y_min, 0)
        self.assertEqual(box2.x_max, 600)
        self.assertEqual(box2.y_max, 400)

        # 3. Dict with x1, y1
        box3 = PersonAAdapter.validate_and_clamp_bbox({"x1": 50, "y1": 60, "x2": 150, "y2": 160}, (600, 400))
        self.assertEqual(box3.x_min, 50)
        self.assertEqual(box3.y_max, 160)

    def test_person_a_adapter_layout_regions_conversion(self):
        """Tests converting raw layout dictionary outputs into strongly-typed RegionRequest objects."""
        raw_regions = [
            {
                "id": "box_01",
                "bbox": [50, 40, 500, 100],
                "label": "printed_header",
                "language": "kannada",
                "confidence": 0.95,
            },
            {
                "bbox": (50, 120, 400, 180),
                "type": "handwritten_signature",
                "language": "kannada",
            },
        ]

        adapted = PersonAAdapter.adapt_layout_regions(raw_regions, image_size=(600, 400))
        self.assertEqual(len(adapted), 2)

        # First region
        self.assertEqual(adapted[0].region_id, "box_01")
        self.assertEqual(adapted[0].region_type, RegionType.HEADER)
        self.assertFalse(adapted[0].is_handwritten)
        self.assertEqual(adapted[0].layout_confidence, 0.95)

        # Second region
        self.assertEqual(adapted[1].region_id, "region_002")
        self.assertEqual(adapted[1].region_type, RegionType.SIGNATURE)
        self.assertTrue(adapted[1].is_handwritten)

    # -------------------------------------------------------------------------
    # 2. KANNADA CONSERVATIVE NORMALIZER TESTS
    # -------------------------------------------------------------------------
    def test_kannada_normalizer_unicode_nfc_and_whitespace(self):
        """Tests that normalizer standardizes Unicode and whitespace without modifying raw text."""
        raw_input = "  ಖಾತೆ   \t  ದಾಖಲೆ  \u200C "
        res = self.normalizer.normalize(raw_input)

        self.assertEqual(res.raw_text, raw_input)
        self.assertEqual(res.normalized_text, "ಖಾತೆ ದಾಖಲೆ")
        self.assertTrue(res.is_modified)
        self.assertIn("ಖಾತೆ", res.candidate_suggestions)

    def test_kannada_normalizer_never_silently_replaces_words(self):
        """Tests that normalizer does not overwrite ambiguous or unknown words."""
        unknown_text = "ಅಸ್ಪಷ್ಟಶಬ್ದ123"
        res = self.normalizer.normalize(unknown_text)

        self.assertEqual(res.normalized_text, "ಅಸ್ಪಷ್ಟಶಬ್ದ123")
        self.assertEqual(res.raw_text, unknown_text)

    # -------------------------------------------------------------------------
    # 3. HUMAN CORRECTION & ACTIVE LEARNING TESTS
    # -------------------------------------------------------------------------
    def test_correction_service_recording_and_validation(self):
        """Tests recording verified human corrections and preventing empty records."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "corrections.jsonl"
            service = CorrectionService(storage_path=log_path)

            # Valid record
            rec = service.record_correction({
                "document_id": "DOC_001",
                "region_id": "reg_001",
                "image_path": "crops/crop_001.png",
                "raw_prediction": "ಸಚಿವರಿದ್ದರೂ",
                "corrected_text": "ಸಚಿವರಾಗಿದ್ದರೂ",
                "ai_confidence": 0.82,
                "language": "kannada",
                "is_handwritten": True,
            })

            self.assertTrue(rec.correction_id.startswith("corr_"))
            self.assertEqual(rec.corrected_text, "ಸಚಿವರಾಗಿದ್ದರೂ")

            # Empty text validation failure
            with self.assertRaises(ValueError):
                service.record_correction({
                    "region_id": "reg_002",
                    "image_path": "crops/crop_002.png",
                    "raw_prediction": "ಅ",
                    "corrected_text": "   ",
                })

            # Verify persistence and query
            stored = service.list_corrections(document_id="DOC_001")
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0].region_id, "reg_001")

            # Export training manifest
            manifest_path = Path(tmp_dir) / "train_manifest.jsonl"
            exported = service.export_training_manifest(manifest_path)
            self.assertEqual(exported, 1)
            self.assertTrue(manifest_path.exists())

    # -------------------------------------------------------------------------
    # 4. END-TO-END PIPELINE RESPONSE SCHEMA TESTS
    # -------------------------------------------------------------------------
    def test_document_pipeline_produces_person_c_response_contract(self):
        """Tests that document pipeline produces structured DocumentProcessingResponse."""
        mock_rec = MockHandoverRecognizer(name="mock-kannada-engine", text="ಸರ್ವೆ ನಂಬರ್ 42", conf=0.92)
        router = LanguageScriptRouter(
            auto_register_kannada=False,
            auto_register_defaults=False,
        )
        router.register_recognizer("kannada", mock_rec, is_handwritten=True, set_as_default=True)

        pipeline = DocumentProcessingPipeline(router=router)

        # Submit request with Person A region
        req = DocumentProcessingRequest(
            image=self.sample_img,
            document_id="DOC_TEST_007",
            page_number=1,
            regions=[
                RegionRequest(
                    region_id="survey_box_01",
                    bbox=BoundingBox(x_min=20, y_min=20, x_max=300, y_max=80),
                    language="kannada",
                    is_handwritten=True,
                )
            ],
        )

        res: DocumentProcessingResponse = pipeline.process_document(request=req)

        self.assertIsInstance(res, DocumentProcessingResponse)
        self.assertEqual(res.document_id, "DOC_TEST_007")
        self.assertEqual(len(res.ordered_regions), 1)

        region_out = res.ordered_regions[0]
        self.assertEqual(region_out.region_id, "survey_box_01")
        self.assertEqual(region_out.raw_text, "ಸರ್ವೆ ನಂಬರ್ 42")
        self.assertEqual(region_out.normalized_text, "ಸರ್ವೆ ನಂಬರ್ 42")
        self.assertIn("ಸರ್ವೆ ನಂಬರ್", region_out.candidate_suggestions)
        self.assertEqual(region_out.confidence, 0.92)
        self.assertFalse(region_out.requires_human_review)
        self.assertEqual(region_out.status, ProcessingStatus.SUCCESS)

        # Verify backwards compatibility with DocumentPage
        self.assertIsNotNone(res.page)
        self.assertEqual(len(res.page.ocr_results), 1)
        self.assertEqual(res.page.ocr_results[0].text, "ಸರ್ವೆ ನಂಬರ್ 42")


if __name__ == "__main__":
    unittest.main()
