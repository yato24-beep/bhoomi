"""Comprehensive End-to-End Integration Tests for Person A + B + C Pipeline.

Verifies the full pipeline flow:
    Document / Image -> Person A -> Person B -> Person C -> FinalDocumentResult / Unified Response

Covers the 10 mandatory pipeline verification scenarios:
1. Test 1: Printed Kannada document
2. Test 2: Handwritten Kannada document
3. Test 3: Printed English document
4. Test 4: Handwritten English document
5. Test 5: Multi-region document
6. Test 6: Low-confidence OCR
7. Test 7: Human-review case
8. Test 8: Person C extraction from OCR output
9. Test 9: Validation failure / warning
10. Test 10: Final response serialization
"""

import json
import unittest
from PIL import Image, ImageDraw

from schemas import (
    BoundingBox,
    DocumentType,
    ExtractedField,
    FinalDocumentResult,
    ValidationStatus,
)
from src.integration.document_pipeline import DocumentProcessingPipeline
from src.integration.person_c_adapter import PersonCAdapter
from src.integration.schemas import (
    DocumentProcessingResponse,
    RecognizedRegionResult,
    RegionRequest,
)
from src.handwriting.recognizer import BaseHandwritingRecognizer, OCRResult


class _MockPredictableRecognizer(BaseHandwritingRecognizer):
    """Predictable mock recognizer for deterministic integration testing."""

    def __init__(self, output_text: str, confidence: float = 0.95, is_handwritten: bool = False):
        super().__init__(model_name="mock-pipeline-engine", model_version="v1")
        self.output_text = output_text
        self.output_conf = confidence
        self.is_hw = is_handwritten

    def recognize_handwriting(self, image, bbox=None, page_number=None, **kwargs):
        return OCRResult(
            text=self.output_text,
            confidence=self.output_conf,
            bbox=bbox,
            is_handwritten=self.is_hw,
            page_number=page_number,
            model_name=self.model_name,
            model_version=self.model_version,
            metadata={"mock_execution": True},
        )


class TestFullIntegratedPipeline(unittest.TestCase):
    """Test suite exercising the entire Person A -> Person B -> Person C pipeline."""

    def setUp(self):
        # Create a basic synthetic document image
        self.dummy_image = Image.new("RGB", (600, 400), color=(250, 250, 248))
        draw = ImageDraw.Draw(self.dummy_image)
        draw.rectangle([(20, 20), (580, 380)], outline=(180, 180, 180), width=2)

    def test_01_printed_kannada_document(self):
        """Test 1: Printed Kannada document produces Bhoomi RTC extraction."""
        rec = _MockPredictableRecognizer("ಸರ್ವೆ ಸಂಖ್ಯೆ: 142/2  ವಿಸ್ತೀರ್ಣ: 1.50 ಹೆಕ್ಟೇರ್", confidence=0.96, is_handwritten=False)
        pipeline = DocumentProcessingPipeline()
        pipeline.router.register_recognizer("kannada", rec, is_handwritten=False, set_as_default=True)

        # Execute A + B
        b_resp = pipeline.process_document(
            image=self.dummy_image,
            language="kannada",
            is_handwritten=False,
            document_id="TEST_01_KN_PRINTED",
        )

        # Execute C via Adapter
        c_res = PersonCAdapter.execute_person_c(b_resp, selected_state="KA")

        self.assertIsInstance(c_res, FinalDocumentResult)
        self.assertEqual(c_res.document_id, "TEST_01_KN_PRINTED")
        self.assertEqual(c_res.state, "KA")
        self.assertEqual(c_res.document_type, DocumentType.BHOOMI_RTC)
        self.assertIn("khasra_number", c_res.fields)
        self.assertEqual(c_res.fields["khasra_number"].normalized_value, "142/2")

    def test_02_handwritten_kannada_document(self):
        """Test 2: Handwritten Kannada document routed through handwriting pipeline."""
        rec = _MockPredictableRecognizer("ಖಾತೆದಾರರ ಹೆಸರು: ಬಸವರಾಜ್ ಪಾಟೀಲ್", confidence=0.91, is_handwritten=True)
        pipeline = DocumentProcessingPipeline()
        pipeline.router.register_recognizer("kannada", rec, is_handwritten=True)

        b_resp = pipeline.process_document(
            image=self.dummy_image,
            language="kannada",
            is_handwritten=True,
            document_id="TEST_02_KN_HANDWRITTEN",
        )

        c_res = PersonCAdapter.execute_person_c(b_resp, selected_state="KA")

        self.assertIsInstance(c_res, FinalDocumentResult)
        self.assertIn("owner_name", c_res.fields)
        self.assertEqual(c_res.fields["owner_name"].raw_value, "ಬಸವರಾಜ್ ಪಾಟೀಲ್")

    def test_03_printed_english_document(self):
        """Test 3: Printed English document extraction."""
        rec = _MockPredictableRecognizer("SURVEY NUMBER: 88/1  TOTAL EXTENT: 2.45 acre", confidence=0.97, is_handwritten=False)
        pipeline = DocumentProcessingPipeline()
        pipeline.router.register_recognizer("english", rec, is_handwritten=False)

        b_resp = pipeline.process_document(
            image=self.dummy_image,
            language="english",
            is_handwritten=False,
            document_id="TEST_03_EN_PRINTED",
        )

        c_res = PersonCAdapter.execute_person_c(b_resp, selected_state="KA")

        self.assertIsInstance(c_res, FinalDocumentResult)
        self.assertIn("khasra_number", c_res.fields)
        self.assertEqual(c_res.fields["khasra_number"].normalized_value, "88/1")
        self.assertIn("land_area", c_res.fields)
        self.assertAlmostEqual(c_res.fields["land_area"].normalized_value, 2.45 * 0.404686, places=3)

    def test_04_handwritten_english_document(self):
        """Test 4: Handwritten English document extraction."""
        rec = _MockPredictableRecognizer("Owner: Johnathan Doe\nFather: William Doe", confidence=0.92, is_handwritten=True)
        pipeline = DocumentProcessingPipeline()
        pipeline.router.register_recognizer("english", rec, is_handwritten=True)

        b_resp = pipeline.process_document(
            image=self.dummy_image,
            language="english",
            is_handwritten=True,
            document_id="TEST_04_EN_HANDWRITTEN",
        )

        c_res = PersonCAdapter.execute_person_c(b_resp, selected_state="DEFAULT")

        self.assertIsInstance(c_res, FinalDocumentResult)
        self.assertIn("owner_name", c_res.fields)
        self.assertEqual(c_res.fields["owner_name"].raw_value, "Johnathan Doe")

    def test_05_multi_region_document(self):
        """Test 5: Multi-region layout with discrete bounding boxes."""
        pipeline = DocumentProcessingPipeline()
        pipeline.router.register_recognizer(
            "kannada",
            _MockPredictableRecognizer("ಸರ್ವೆ ನಂ: 105", confidence=0.95),
            is_handwritten=False,
            set_as_default=True,
        )

        regions = [
            RegionRequest(region_id="reg_1", bbox=BoundingBox(x_min=20, y_min=20, x_max=300, y_max=60), language="kannada", is_handwritten=False),
            RegionRequest(region_id="reg_2", bbox=BoundingBox(x_min=20, y_min=80, x_max=300, y_max=120), language="kannada", is_handwritten=False),
            RegionRequest(region_id="reg_3", bbox=BoundingBox(x_min=20, y_min=140, x_max=300, y_max=180), language="kannada", is_handwritten=False),
        ]

        b_resp = pipeline.process_document(
            image=self.dummy_image,
            regions=regions,
            document_id="TEST_05_MULTI_REGION",
        )

        self.assertEqual(len(b_resp.ordered_regions), 3)
        c_res = PersonCAdapter.execute_person_c(b_resp, selected_state="KA")
        self.assertIn("khasra_number", c_res.fields)

    def test_06_low_confidence_ocr(self):
        """Test 6: Low-confidence OCR triggers lower overall confidence."""
        rec = _MockPredictableRecognizer("ಗಾತಾ ಸಂ: 12", confidence=0.35, is_handwritten=False)
        pipeline = DocumentProcessingPipeline()
        pipeline.router.register_recognizer("kannada", rec, is_handwritten=False, set_as_default=True)

        b_resp = pipeline.process_document(
            image=self.dummy_image,
            document_id="TEST_06_LOW_CONF",
        )

        c_res = PersonCAdapter.execute_person_c(b_resp, selected_state="UP")
        self.assertLess(c_res.overall_confidence, 0.70)

    def test_07_human_review_case(self):
        """Test 7: Ambiguity or review trigger sets requires_human_review to True."""
        regions = [
            RecognizedRegionResult(
                region_id="r1",
                raw_text="Unknown script content ???",
                normalized_text="Unknown script content ???",
                bbox=BoundingBox(x_min=10, y_min=10, x_max=200, y_max=50),
                language="kannada",
                script="Kannada",
                confidence=0.30,
                requires_human_review=True,
            )
        ]
        b_resp = DocumentProcessingResponse(
            document_id="TEST_07_REVIEW",
            page_number=1,
            image_path="test.png",
            ordered_regions=regions,
            merged_text="Unknown script content ???",
            document_confidence=0.30,
            requires_human_review=True,
            warnings=["Low confidence OCR output"],
        )

        c_res = PersonCAdapter.execute_person_c(b_resp, selected_state="DEFAULT")
        unified = PersonCAdapter.to_unified_response(b_resp, c_res)

        self.assertTrue(unified["requires_human_review"])
        self.assertGreater(len(unified["review_reasons"]), 0)

    def test_08_person_c_extraction_from_ocr_output(self):
        """Test 8: Full Person C extraction cycle from structured OCR output."""
        regions = [
            RecognizedRegionResult(
                region_id="r1",
                raw_text="ग्राम: मऊ  तहसील: मोहनलालगंज  जनपद: लखनऊ",
                normalized_text="ग्राम: मऊ  तहसील: मोहनलालगंज  जनपद: लखनऊ",
                bbox=BoundingBox(x_min=10, y_min=10, x_max=400, y_max=40),
                language="hindi",
                script="Devanagari",
                confidence=0.98,
            ),
            RecognizedRegionResult(
                region_id="r2",
                raw_text="खाता संख्या: 00124  गाटा संख्या: 142/1  क्षेत्रफल: 0.4500 हेक्टेयर",
                normalized_text="खाता संख्या: 00124  गाटा संख्या: 142/1  क्षेत्रफल: 0.4500 हेक्टेयर",
                bbox=BoundingBox(x_min=10, y_min=50, x_max=500, y_max=90),
                language="hindi",
                script="Devanagari",
                confidence=0.97,
            ),
            RecognizedRegionResult(
                region_id="r3",
                raw_text="खातेदार: श्री राम प्रसाद  पिता: श्याम लाल",
                normalized_text="खातेदार: श्री राम प्रसाद  पिता: श्याम लाल",
                bbox=BoundingBox(x_min=10, y_min=100, x_max=400, y_max=140),
                language="hindi",
                script="Devanagari",
                confidence=0.96,
            ),
        ]
        b_resp = DocumentProcessingResponse(
            document_id="TEST_08_UP_FULL",
            page_number=1,
            image_path="test_up.png",
            ordered_regions=regions,
            merged_text="\n".join(r.raw_text for r in regions),
            document_confidence=0.97,
            requires_human_review=False,
        )

        c_res = PersonCAdapter.execute_person_c(b_resp, selected_state="UP")

        self.assertEqual(c_res.state, "UP")
        self.assertEqual(c_res.fields["khasra_number"].normalized_value, "142/1")
        self.assertEqual(c_res.fields["owner_name"].normalized_value, "राम प्रसाद")
        self.assertEqual(c_res.fields["land_area"].normalized_value, 0.4500)
        self.assertTrue(c_res.gis_validation.is_verified)

    def test_09_validation_failure_warning(self):
        """Test 9: Invalid area value triggers validation warning/error."""
        regions = [
            RecognizedRegionResult(
                region_id="r1",
                raw_text="गाटा संख्या: 142/1  क्षेत्रफल: 999999 हेक्टेयर",
                normalized_text="गाटा संख्या: 142/1  क्षेत्रफल: 999999 हेक्टेयर",
                bbox=BoundingBox(x_min=10, y_min=10, x_max=300, y_max=50),
                language="hindi",
                script="Devanagari",
                confidence=0.95,
            )
        ]
        b_resp = DocumentProcessingResponse(
            document_id="TEST_09_AREA_FAIL",
            page_number=1,
            image_path="test_fail.png",
            ordered_regions=regions,
            merged_text=regions[0].raw_text,
            document_confidence=0.95,
        )

        c_res = PersonCAdapter.execute_person_c(b_resp, selected_state="UP")
        self.assertIn(c_res.validation_status, (ValidationStatus.INVALID, ValidationStatus.WARNING))
        self.assertTrue(c_res.requires_human_review)

    def test_10_final_response_serialization(self):
        """Test 10: Unified dictionary serializes cleanly to standard JSON."""
        regions = [
            RecognizedRegionResult(
                region_id="r1",
                raw_text="ಸರ್ವೆ ಸಂಖ್ಯೆ: 55/1",
                normalized_text="ಸರ್ವೆ ಸಂಖ್ಯೆ: 55/1",
                bbox=BoundingBox(x_min=10, y_min=10, x_max=200, y_max=40),
                language="kannada",
                script="Kannada",
                confidence=0.96,
            )
        ]
        b_resp = DocumentProcessingResponse(
            document_id="TEST_10_SERIALIZE",
            page_number=1,
            image_path="test_ser.png",
            ordered_regions=regions,
            merged_text="ಸರ್ವೆ ಸಂಖ್ಯೆ: 55/1",
            document_confidence=0.96,
        )

        c_res = PersonCAdapter.execute_person_c(b_resp, selected_state="KA")
        unified = PersonCAdapter.to_unified_response(b_resp, c_res)

        # Must serialize to string without TypeError
        json_str = json.dumps(unified, ensure_ascii=False, indent=2)
        self.assertIsInstance(json_str, str)
        self.assertIn("khasra_number", json_str)
        self.assertIn("ocr", json_str)


if __name__ == "__main__":
    unittest.main()
