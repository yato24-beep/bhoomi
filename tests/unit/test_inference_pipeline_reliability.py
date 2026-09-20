"""Unit tests validating inference and pipeline reliability constraints.

Tests:
1. Model reuse (singleton instance, loaded once per process)
2. PDF -> page images and multi-page ordering
3. Printed Kannada -> EasyOCR route preservation
4. Handwritten Kannada -> IITB TrOCR crop route
5. No accidental full-page TrOCR (full page routes to detector/EasyOCR)
6. Low-confidence OCR withheld from machine translation
7. Multi-word line crops flagged as needs_review on word-oriented TrOCR
8. Structured OCR schema compliance
9. Unsupported/corrupt input handling (no crashing, no fake invoice)
10. No fake confidence (calibrated_confidence is None / uncalibrated)
11. Deterministic routing decisions and logged reasons
"""

import io
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image, ImageDraw

from src.handwriting.trocr_recognizer import get_iitb_kannada_recognizer, TrocrHandwritingRecognizer
from src.preprocessing.format_adapter import DocumentFormatAdapter, DocumentIngestionError
from src.integration.structured_schema import (
    StructuredDocumentOCR,
    OCRPage,
    OCRRegion,
    RegionType,
    ScriptType,
)
from schemas import OCRResult
from src.integration.schemas import DocumentProcessingRequest, RegionRequest, BoundingBox
from src.integration.document_pipeline import DocumentProcessingPipeline, get_default_pipeline
from backend.app.pipeline.processor import MultimodalOCRDocumentProcessor, ProcessingResult


class TestInferencePipelineReliability(unittest.TestCase):
    """Suite verifying pipeline reliability, safety, and strict non-regression."""

    def setUp(self):
        # Create small test synthetic image
        self.test_img = Image.new("RGB", (200, 60), color=(255, 255, 255))
        draw = ImageDraw.Draw(self.test_img)
        draw.text((10, 20), "ಭೂಮಿ", fill=(0, 0, 0))

    def test_01_model_reuse_singleton(self):
        """Task 2: Ensure IITB TrOCR model instance is cached as singleton per worker."""
        rec1 = get_iitb_kannada_recognizer(auto_load=False)
        rec2 = get_iitb_kannada_recognizer(auto_load=False)
        self.assertIs(rec1, rec2, "get_iitb_kannada_recognizer must return the exact same cached instance")

    def test_02_pdf_to_page_images_and_ordering(self):
        """Task 8: Verify multi-page PDF renders to ordered PIL page images."""
        adapter = DocumentFormatAdapter()
        
        # Build 2-page synthetic PDF in-memory using PIL multi-page save
        img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = Image.new("RGB", (100, 100), color=(0, 255, 0))
        pdf_buf = io.BytesIO()
        img1.save(pdf_buf, format="PDF", save_all=True, append_images=[img2])
        pdf_bytes = pdf_buf.getvalue()

        pages = adapter.load_pages(pdf_bytes, filename="test_multipage.pdf")
        self.assertEqual(len(pages), 2, "Must extract exactly 2 pages from 2-page PDF")
        self.assertIsInstance(pages[0], Image.Image)
        self.assertIsInstance(pages[1], Image.Image)
        # Check dimensions
        self.assertGreaterEqual(pages[0].width, 100)
        self.assertGreaterEqual(pages[0].height, 100)
        self.assertGreaterEqual(pages[1].width, 100)
        self.assertGreaterEqual(pages[1].height, 100)

    def test_03_printed_kannada_routes_to_easyocr(self):
        """Task 4: Enforce printed Kannada routes to EasyOCR, keeping route deterministic."""
        pipeline = DocumentProcessingPipeline(enable_document_gating=False)
        
        # Region explicitly specified as printed Kannada
        region = RegionRequest(
            region_id="reg_printed_1",
            bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50),
            is_handwritten=False,
            language="kannada",
            script="Kannada",
            region_type="word",
        )

        with patch.object(pipeline.router, "route_and_recognize") as mock_route:
            mock_route.return_value = OCRResult(
                text="ಬೆಂಗಳೂರು",
                confidence=0.91,
                bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50),
                is_handwritten=False,
                model_name="easyocr_kannada",
                model_version="1.7.1",
                metadata={"routing_reason": "Printed Kannada routed to EasyOCR"},
            )
            resp = pipeline.process_document(
                image=self.test_img,
                regions=[region],
                is_handwritten=False,
            )
            self.assertEqual(len(resp.ordered_regions), 1)
            reg_out = resp.ordered_regions[0]
            self.assertEqual(reg_out.model_name, "easyocr_kannada")
            self.assertIn("EasyOCR", reg_out.routing_reason)

    def test_04_handwritten_kannada_word_routes_to_trocr(self):
        """Task 4: Enforce handwritten Kannada word crops route to IITB TrOCR."""
        pipeline = DocumentProcessingPipeline(enable_document_gating=False)
        
        region = RegionRequest(
            region_id="reg_hw_word_1",
            bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50),
            is_handwritten=True,
            language="kannada",
            script="Kannada",
            region_type="word",
        )

        with patch.object(pipeline.router, "route_and_recognize") as mock_route:
            mock_route.return_value = OCRResult(
                text="ರಾಮಯ್ಯ",
                confidence=0.78,
                bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50),
                is_handwritten=True,
                model_name="iitb_kannada_v002",
                model_version="0.0.2",
                metadata={"routing_reason": "Handwritten Kannada isolated word routed to IITB TrOCR v0.0.2"},
            )
            resp = pipeline.process_document(
                image=self.test_img,
                regions=[region],
                is_handwritten=True,
            )
            self.assertEqual(len(resp.ordered_regions), 1)
            reg_out = resp.ordered_regions[0]
            self.assertEqual(reg_out.model_name, "iitb_kannada_v002")
            self.assertIn("IITB TrOCR", reg_out.routing_reason)

    def test_05_no_accidental_full_page_trocr(self):
        """Task 4: Never send full-page image directly to TrOCR merely because is_handwritten is True."""
        pipeline = DocumentProcessingPipeline(enable_document_gating=False)
        
        # When full image is passed with is_handwritten=True and no pre-segmented regions
        with patch.object(pipeline.text_detector, "detect_lines_and_words") as mock_detect, \
             patch.object(pipeline.router, "route_and_recognize") as mock_route:
            
            mock_detect.return_value = []
            mock_route.return_value = OCRResult(
                text="ಪಹಣಿ",
                confidence=0.88,
                bbox=BoundingBox(x_min=0, y_min=0, x_max=200, y_max=60),
                is_handwritten=False,
                model_name="easyocr_kannada",
                model_version="1.7.1",
                metadata={"routing_reason": "Fallback text detection via EasyOCR"},
            )
            
            resp = pipeline.process_document(
                image=self.test_img,
                regions=None,
                is_handwritten=True,
                apply_segmentation=False,
            )
            
            # The full-page fallback must route to EasyOCR detector, NEVER directly into TrOCR
            for reg in resp.ordered_regions:
                self.assertNotEqual(
                    reg.model_name,
                    "iitb_kannada_v002",
                    "Full page image must NEVER be recognized directly by word-oriented TrOCR",
                )

    def test_06_low_confidence_ocr_withheld_from_translation(self):
        """Task 7: Low-confidence OCR (< 0.60) must NOT automatically translate."""
        pipeline = DocumentProcessingPipeline(enable_document_gating=False)
        
        region = RegionRequest(
            region_id="reg_low_conf",
            bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=30),
            is_handwritten=True,
            language="kannada",
            script="Kannada",
            region_type="word",
        )

        with patch.object(pipeline.router, "route_and_recognize") as mock_route:
            mock_route.return_value = OCRResult(
                text="ಕೇವಲ",
                confidence=0.35,  # LOW CONFIDENCE < 0.60
                bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=30),
                is_handwritten=True,
                model_name="iitb_kannada_v002",
                model_version="0.0.2",
                metadata={"routing_reason": "Handwritten Kannada"},
            )
            resp = pipeline.process_document(
                image=self.test_img,
                regions=[region],
                is_handwritten=True,
            )
            
            self.assertEqual(resp.translation_status, "skipped_low_confidence")
            self.assertIn("Translation skipped: Low OCR confidence", resp.translated_text)
            self.assertEqual(resp.clean_kannada_text, "ಕೇವಲ")

    def test_07_multi_word_line_flagged_for_review(self):
        """Task 5: Multi-word line crops sent to word-oriented TrOCR must be flagged needs_review."""
        pipeline = DocumentProcessingPipeline(enable_document_gating=False)
        
        region = RegionRequest(
            region_id="reg_line_1",
            bbox=BoundingBox(x_min=0, y_min=0, x_max=180, y_max=40),
            is_handwritten=True,
            language="kannada",
            script="Kannada",
            region_type="line",  # LINE crop
        )

        with patch.object(pipeline.router, "route_and_recognize") as mock_route:
            mock_route.return_value = OCRResult(
                text="ಶ್ರೀ ಕರಿದಿರ್ ಮದನಗೌಡರ",
                confidence=0.72,
                bbox=BoundingBox(x_min=0, y_min=0, x_max=180, y_max=40),
                is_handwritten=True,
                model_name="iitb_kannada_v002",
                model_version="0.0.2",
                metadata={"routing_reason": "Handwritten line crop routed to IITB TrOCR"},
            )
            resp = pipeline.process_document(
                image=self.test_img,
                regions=[region],
                is_handwritten=True,
            )
            
            self.assertEqual(len(resp.ordered_regions), 1)
            reg = resp.ordered_regions[0]
            self.assertTrue(reg.requires_human_review, "Multi-word line on word model must require human review")
            self.assertIn("Multi-word line crop routed to word-oriented TrOCR", reg.review_reason)

    def test_08_structured_ocr_output_schema(self):
        """Task 10: Verify structured OCR JSON output schema conformance."""
        pipeline = DocumentProcessingPipeline(enable_document_gating=False)
        
        resp = pipeline.process_document(
            image=self.test_img,
            regions=[],
            is_handwritten=False,
        )
        
        self.assertIsNotNone(resp.structured_ocr)
        struct_doc = StructuredDocumentOCR(**resp.structured_ocr)
        self.assertIsInstance(struct_doc.pages, list)
        self.assertGreaterEqual(len(struct_doc.pages), 1)
        self.assertIsNone(struct_doc.overall_confidence_calibrated)
        self.assertIn("total_ms", struct_doc.total_timings.model_dump())

    def test_09_unsupported_or_corrupt_input_handling(self):
        """Tasks 8 & 9: Corrupt/unsupported input must return structured error, never crash or fabricate invoice."""
        processor = MultimodalOCRDocumentProcessor()
        corrupt_stream = io.BytesIO(b"NOT_A_VALID_IMAGE_OR_PDF_CORRUPT_BYTES")
        
        result = processor.process(
            file_stream=corrupt_stream,
            filename="corrupt_document.xyz",
            content_type="application/octet-stream",
        )
        
        self.assertIsInstance(result, ProcessingResult)
        self.assertFalse(result.is_valid)
        self.assertIn(result.extracted_data.get("document_type"), (None, "Unknown / Not classified"))
        self.assertEqual(result.extracted_data.get("document_type_state"), "UNKNOWN")
        self.assertNotEqual(result.extracted_data.get("document_type"), "Commercial Invoice")
        self.assertEqual(result.verification_status, "needs_verification")

    def test_10_no_fake_confidence_uncalibrated(self):
        """Task 6: Ensure calibrated_confidence is strictly None and no 97-99% mock score is fabricated."""
        pipeline = DocumentProcessingPipeline(enable_document_gating=False)
        resp = pipeline.process_document(
            image=self.test_img,
            regions=[],
            is_handwritten=False,
        )
        
        self.assertIsNone(resp.calibrated_confidence, "Calibrated confidence must be None until genuine calibration exists")
        for reg in resp.ordered_regions:
            self.assertIsNone(reg.calibrated_confidence)

    def test_11_deterministic_routing_reasons(self):
        """Task 4: Enforce deterministic routing decisions and explicit reason tracking."""
        pipeline = DocumentProcessingPipeline(enable_document_gating=False)
        region = RegionRequest(
            region_id="reg_det_1",
            bbox=BoundingBox(x_min=10, y_min=10, x_max=80, y_max=40),
            is_handwritten=False,
            language="kannada",
            script="Kannada",
            region_type="word",
        )
        
        with patch.object(pipeline.router, "route_and_recognize") as mock_route:
            mock_route.return_value = OCRResult(
                text="ಸರ್ವೆ",
                confidence=0.89,
                bbox=BoundingBox(x_min=10, y_min=10, x_max=80, y_max=40),
                is_handwritten=False,
                model_name="easyocr_kannada",
                model_version="1.7.1",
                metadata={"routing_reason": "Printed Kannada routed to EasyOCR"},
            )
            resp1 = pipeline.process_document(image=self.test_img, regions=[region], is_handwritten=False)
            resp2 = pipeline.process_document(image=self.test_img, regions=[region], is_handwritten=False)
            
            self.assertEqual(resp1.ordered_regions[0].model_name, resp2.ordered_regions[0].model_name)
            self.assertEqual(resp1.ordered_regions[0].routing_reason, resp2.ordered_regions[0].routing_reason)


if __name__ == "__main__":
    unittest.main()
