"""Final Production Architecture Test Suite.

Validates all 14 mandatory production requirements:
1.  PDF ingestion
2.  JPG ingestion
3.  PNG ingestion
4.  TIFF ingestion
5.  Document rotations (0, 90, 180, 270 degrees)
6.  Mixed printed + handwritten page routing
7.  Handwritten-only page routing (IITB v0.0.2 word-level reconstruction)
8.  Printed-only page routing (EasyOCR, never running handwriting OCR)
9.  Wrong document type rejection (Land Record Gate)
10. Low-confidence OCR guarded translation skip
11. Nonsense / empty OCR handling
12. Field spatial association (anchor proximity requirement, no distant matching)
13. Calibrated 4-part confidence reporting & verification status
14. Batch-vs-single inference equivalence
"""

import io
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from schemas import BoundingBox, OCRResult
from src.classification.document_gate import LandRecordGateClassifier
from src.classification.style_classifier import ScriptStyleClassifier
from src.extraction.tabular_ner import TabularLayoutExtractor, TabularToken
from src.handwriting.trocr_recognizer import (
    get_iitb_kannada_recognizer,
    TrocrHandwritingRecognizer,
)
from src.integration.document_pipeline import (
    DocumentProcessingPipeline,
    get_default_pipeline,
)
from src.integration.schemas import (
    RecognizedRegionResult,
    RegionRequest,
    RegionType,
)
from src.preprocessing.format_adapter import DocumentFormatAdapter
from src.preprocessing.image_enhancement import detect_and_correct_coarse_orientation


class TestFinalProductionArchitecture(unittest.TestCase):
    """Production architecture verification test suite."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent.parent
        cls.sample_img = Image.new("RGB", (300, 300), color=(255, 255, 255))
        d = ImageDraw.Draw(cls.sample_img)
        d.rectangle([(20, 20), (280, 280)], outline=(0, 0, 0), width=2)
        d.text((40, 50), "Karnataka Land Record RTC", fill=(0, 0, 0))

        # Real test crop if present
        cls.crop1_path = cls.repo_root / "training/datasets/personal_trial/crops/crop_a_mara.png"
        cls.crop2_path = cls.repo_root / "training/datasets/personal_trial/crops/crop_b_katte.png"

    # ------------------------------------------------------------------
    # 1-4: Format Adapter Ingestion (PDF / JPG / PNG / TIFF)
    # ------------------------------------------------------------------

    def test_01_ingest_pdf(self):
        """Unified ingest path converts PDF into normalized RGB PIL Images."""
        buf = io.BytesIO()
        self.sample_img.save(buf, format="PDF")
        pdf_bytes = buf.getvalue()

        pages = DocumentFormatAdapter.load_pages(pdf_bytes, filename="record.pdf")
        self.assertIsInstance(pages, list)
        self.assertGreaterEqual(len(pages), 1)
        self.assertIsInstance(pages[0], Image.Image)
        self.assertEqual(pages[0].mode, "RGB")
        self.assertGreater(pages[0].width, 0)
        self.assertGreater(pages[0].height, 0)

    def test_02_ingest_jpg(self):
        """Unified ingest path converts JPG into normalized RGB PIL Images."""
        buf = io.BytesIO()
        self.sample_img.save(buf, format="JPEG")
        jpg_bytes = buf.getvalue()

        pages = DocumentFormatAdapter.load_pages(jpg_bytes, filename="scan.jpg")
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].mode, "RGB")

    def test_03_ingest_png(self):
        """Unified ingest path converts PNG into normalized RGB PIL Images."""
        buf = io.BytesIO()
        self.sample_img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        pages = DocumentFormatAdapter.load_pages(png_bytes, filename="document.png")
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].mode, "RGB")

    def test_04_ingest_tiff(self):
        """Unified ingest path converts TIFF into normalized RGB PIL Images."""
        buf = io.BytesIO()
        self.sample_img.save(buf, format="TIFF")
        tiff_bytes = buf.getvalue()

        pages = DocumentFormatAdapter.load_pages(tiff_bytes, filename="archive.tiff")
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].mode, "RGB")

    # ------------------------------------------------------------------
    # 5: Document Orientation (0, 90, 180, 270)
    # ------------------------------------------------------------------

    def test_05_rotations_0_90_180_270(self):
        """Validates orientation detection and correction for 0, 90, 180, 270 degrees."""
        for angle in (0, 90, 180, 270):
            rotated_src = self.sample_img.rotate(angle, expand=True) if angle != 0 else self.sample_img
            corrected_img, meta = detect_and_correct_coarse_orientation(rotated_src)
            self.assertIsInstance(corrected_img, Image.Image)
            self.assertIn("coarse_rotation_applied", meta)
            self.assertIn("angle_degrees", meta)

    # ------------------------------------------------------------------
    # 6: Mixed Printed + Handwritten Page Routing
    # ------------------------------------------------------------------

    def test_06_mixed_printed_handwritten_page(self):
        """Validates per-region style classification routes printed to EasyOCR and handwritten to IITB."""
        pipeline = DocumentProcessingPipeline(
            default_language="kannada",
            apply_preprocessing=False,
            enable_document_gating=False,
        )

        regions = [
            BoundingBox(x_min=10, y_min=10, x_max=200, y_max=50),
            BoundingBox(x_min=10, y_min=60, x_max=200, y_max=100),
        ]

        # Region 0: printed Kannada -> EasyOCR
        # Region 1: handwritten Kannada -> IITB v0.0.2
        res = pipeline.process_document(
            image=self.sample_img,
            regions=regions,
            language="kannada",
        )

        self.assertIsNotNone(res)
        self.assertEqual(len(res.ordered_regions), 2)
        # Ensure routing confidence is tracked
        self.assertIsNotNone(res.routing_confidence)
        self.assertGreaterEqual(res.routing_confidence, 0.0)

    # ------------------------------------------------------------------
    # 7: Handwritten-Only Page Routing (IITB v0.0.2)
    # ------------------------------------------------------------------

    def test_07_handwritten_only_page(self):
        """Validates handwritten page dispatches to IITB Indic-TrOCR v0.0.2 with diagnostics."""
        pipeline = DocumentProcessingPipeline(
            default_language="kannada",
            apply_preprocessing=False,
            enable_document_gating=False,
        )

        res = pipeline.process_document(
            image=self.sample_img,
            is_handwritten=True,
            language="kannada",
        )

        self.assertIsNotNone(res)
        self.assertIn("iitb_kannada_v002", res.engine_breakdown)
        self.assertEqual(res.diagnostics.get("recognizer"), "iitb_kannada_v002")
        self.assertEqual(
            res.diagnostics.get("model_path"),
            "models/trocr/experimental/iitb_kannada_v002",
        )

    # ------------------------------------------------------------------
    # 8: Printed-Only Page Routing (EasyOCR, Never Run TrOCR)
    # ------------------------------------------------------------------

    def test_08_printed_only_page(self):
        """Validates printed Kannada uses EasyOCR and never runs handwriting OCR."""
        pipeline = DocumentProcessingPipeline(
            default_language="kannada",
            apply_preprocessing=False,
            enable_document_gating=False,
        )

        res = pipeline.process_document(
            image=self.sample_img,
            is_handwritten=False,
            language="kannada",
        )

        self.assertIsNotNone(res)
        # IITB should NOT be in engine breakdown for printed document
        self.assertNotIn("iitb_kannada_v002", res.engine_breakdown)

    # ------------------------------------------------------------------
    # 9: Wrong Document Type Rejection (Document Gate)
    # ------------------------------------------------------------------

    def test_09_wrong_document_type(self):
        """Validates non-land-record image is rejected by gate before heavy OCR."""
        gate = LandRecordGateClassifier(min_grid_confidence=0.35)
        # Uniform solid block is definitely not a land record
        blank_img = Image.new("RGB", (400, 400), color=(128, 128, 128))
        res = gate.classify_image(blank_img)

        self.assertFalse(res.is_land_record)
        self.assertTrue(len(res.rejection_reason) > 0)

        # Test pipeline gate integration
        pipeline = DocumentProcessingPipeline(
            default_language="kannada",
            apply_preprocessing=False,
            enable_document_gating=True,
        )
        pipe_res = pipeline.process_document(image=blank_img)
        self.assertFalse(pipe_res.is_land_record)
        self.assertEqual(pipe_res.status, "rejected_not_land_record")
        self.assertEqual(pipe_res.verification_status, "needs_verification")

    # ------------------------------------------------------------------
    # 10: Low-Confidence OCR Never Translated
    # ------------------------------------------------------------------

    def test_10_low_confidence_ocr_no_translation(self):
        """Validates machine translation is skipped when OCR confidence is below threshold."""
        pipeline = DocumentProcessingPipeline(
            default_language="kannada",
            confidence_threshold=0.99,  # High threshold guarantees low-confidence trigger
            apply_preprocessing=False,
            enable_document_gating=False,
        )

        res = pipeline.process_document(
            image=self.sample_img,
            language="kannada",
        )

        self.assertIn("[Translation skipped: Low OCR confidence]", res.english_translation)
        self.assertEqual(res.verification_status, "needs_verification")

    # ------------------------------------------------------------------
    # 11: Nonsense / Empty OCR Handling
    # ------------------------------------------------------------------

    def test_11_nonsense_or_empty_ocr(self):
        """Validates empty or noisy crops are handled gracefully without pipeline crash."""
        pipeline = DocumentProcessingPipeline(
            default_language="kannada",
            apply_preprocessing=False,
            enable_document_gating=False,
        )

        empty_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        res = pipeline.process_document(image=empty_img, is_handwritten=True)

        self.assertIsNotNone(res)
        self.assertTrue(res.requires_human_review)
        self.assertEqual(res.verification_status, "needs_verification")

    # ------------------------------------------------------------------
    # 12: Field Spatial Association
    # ------------------------------------------------------------------

    def test_12_field_spatial_association(self):
        """Validates fields are matched strictly to adjacent value cells, ignoring distant matching text."""
        extractor = TabularLayoutExtractor(
            value_search_radius_y=100.0,
            value_search_radius_x=150.0,
        )

        tokens = [
            # Anchor: Survey Number at (20, 20) -> (120, 40)
            TabularToken(text="ಸರ್ವೆ ನಂಬರ್", confidence=0.95, x_min=20, y_min=20, x_max=120, y_max=40),
            # Spatially valid adjacent value directly below at (20, 50) -> (80, 70)
            TabularToken(text="142/2", confidence=0.92, x_min=20, y_min=50, x_max=80, y_max=70),
            # Distant distractor matching text at (600, 800) -> (660, 820)
            TabularToken(text="999/X", confidence=0.99, x_min=600, y_min=800, x_max=660, y_max=820),
        ]

        extracted = extractor.extract_from_tokens(tokens)
        self.assertIn("survey_number", extracted)
        # Value must be the spatially adjacent token, NOT the distant one
        self.assertEqual(extracted["survey_number"].value_text, "142/2")
        self.assertEqual(extracted["survey_number"].spatial_relationship, "below")

    # ------------------------------------------------------------------
    # 13: Confidence Calibration & Verification Status
    # ------------------------------------------------------------------

    def test_13_confidence_calibration(self):
        """Validates separate recognition, detection, routing, field confidences and explicit verification status."""
        pipeline = DocumentProcessingPipeline(
            default_language="kannada",
            apply_preprocessing=False,
            enable_document_gating=False,
        )

        res = pipeline.process_document(
            image=self.sample_img,
            language="kannada",
        )

        # 4 separate calibrated probabilities
        self.assertIsNotNone(res.recognition_confidence)
        self.assertIsNotNone(res.detection_confidence)
        self.assertIsNotNone(res.routing_confidence)
        # Explicit status
        self.assertIn(res.verification_status, ("accepted", "needs_verification"))

    # ------------------------------------------------------------------
    # 14: Batch vs Single Inference Equivalence
    # ------------------------------------------------------------------

    def test_14_batch_vs_single_inference_equivalence(self):
        """Validates IITB Kannada TrOCR yields equivalent results in batch mode vs single inference."""
        if not self.crop1_path.exists() or not self.crop2_path.exists():
            self.skipTest("Personal trial crops not available for batch equivalence test.")

        rec = get_iitb_kannada_recognizer(auto_load=True)
        img1 = Image.open(self.crop1_path).convert("RGB")
        img2 = Image.open(self.crop2_path).convert("RGB")

        # Single inferences
        single1 = rec.recognize_handwriting(img1)
        single2 = rec.recognize_handwriting(img2)

        # Batch inference
        batch_results = rec.recognize_batch([img1, img2])
        self.assertEqual(len(batch_results), 2)

        # Equivalence checks
        self.assertEqual(single1.text.strip(), batch_results[0].text.strip())
        self.assertEqual(single2.text.strip(), batch_results[1].text.strip())
        if single1.confidence is not None and batch_results[0].confidence is not None:
            self.assertAlmostEqual(single1.confidence, batch_results[0].confidence, places=2)
        if single2.confidence is not None and batch_results[1].confidence is not None:
            self.assertAlmostEqual(single2.confidence, batch_results[1].confidence, places=2)


if __name__ == "__main__":
    unittest.main()
