"""Unit tests for Person C Adapter layer.

Validates:
1. Converting Person B DocumentProcessingResponse into Person C DocumentOCRResult & HandwritingResult
2. Preserving text, confidence, bounding boxes, language, and review flags
3. Executing Person C extraction pipeline on Person B output
4. Formatting unified response for frontend API consumption
"""

import unittest
from schemas import (
    BoundingBox,
    DocumentOCRResult,
    DocumentType,
    ExtractedField,
    FinalDocumentResult,
    HandwritingResult,
    ValidationStatus,
)
from src.integration.person_c_adapter import PersonCAdapter
from src.integration.schemas import (
    DocumentProcessingResponse,
    ProcessingStatus,
    RecognizedRegionResult,
)


class TestPersonCAdapter(unittest.TestCase):
    """Test suite for PersonCAdapter bridging Person B and Person C."""

    def setUp(self):
        # Sample Person B DocumentProcessingResponse for Karnataka Bhoomi RTC
        self.karnataka_regions = [
            RecognizedRegionResult(
                region_id="reg_001",
                raw_text="ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಕಂದಾಯ ಇಲಾಖೆ",
                normalized_text="ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಕಂದಾಯ ಇಲಾಖೆ",
                bbox=BoundingBox(x_min=50.0, y_min=20.0, x_max=500.0, y_max=60.0),
                page_number=1,
                language="kannada",
                script="Kannada",
                is_handwritten=False,
                confidence=0.96,
                model_name="paddleocr-kannada",
            ),
            RecognizedRegionResult(
                region_id="reg_002",
                raw_text="ಸರ್ವೆ ನಂ: 42/1",
                normalized_text="ಸರ್ವೆ ನಂ: 42/1",
                bbox=BoundingBox(x_min=50.0, y_min=80.0, x_max=300.0, y_max=120.0),
                page_number=1,
                language="kannada",
                script="Kannada",
                is_handwritten=False,
                confidence=0.98,
                model_name="paddleocr-kannada",
            ),
            RecognizedRegionResult(
                region_id="reg_003",
                raw_text="ಖಾತೆದಾರರ ಹೆಸರು: ರಮೇಶ್ ಕುಮಾರ್",
                normalized_text="ಖಾತೆದಾರರ ಹೆಸರು: ರಮೇಶ್ ಕುಮಾರ್",
                bbox=BoundingBox(x_min=50.0, y_min=140.0, x_max=400.0, y_max=180.0),
                page_number=1,
                language="kannada",
                script="Kannada",
                is_handwritten=True,
                confidence=0.92,
                model_name="trocr-kannada-handwritten",
            ),
            RecognizedRegionResult(
                region_id="reg_004",
                raw_text="ವಿಸ್ತೀರ್ಣ: 2 ಎಕರೆ 10 ಗುಂಟೆ",
                normalized_text="ವಿಸ್ತೀರ್ಣ: 2 ಎಕರೆ 10 ಗುಂಟೆ",
                bbox=BoundingBox(x_min=50.0, y_min=200.0, x_max=380.0, y_max=240.0),
                page_number=1,
                language="kannada",
                script="Kannada",
                is_handwritten=False,
                confidence=0.94,
                model_name="paddleocr-kannada",
            ),
        ]

        self.karnataka_b_response = DocumentProcessingResponse(
            document_id="DOC_B_KA_001",
            page_number=1,
            image_path="karnataka_rtc_sample.png",
            ordered_regions=self.karnataka_regions,
            merged_text="\n".join(r.normalized_text for r in self.karnataka_regions),
            document_confidence=0.95,
            status="completed",
            requires_human_review=False,
            warnings=[],
            engine_breakdown={"paddleocr-kannada": 3, "trocr-kannada-handwritten": 1},
            processing_time_ms=150.0,
        )

    def test_to_person_c_input_mapping(self):
        """Validates that Person B response maps cleanly to DocumentOCRResult and HandwritingResult."""
        ocr_result, hw_result = PersonCAdapter.to_person_c_input(
            b_response=self.karnataka_b_response,
            selected_state="KA",
        )

        self.assertIsInstance(ocr_result, DocumentOCRResult)
        self.assertEqual(ocr_result.document_id, "DOC_B_KA_001")
        self.assertEqual(ocr_result.classification.state, "KA")
        self.assertEqual(ocr_result.classification.document_type, DocumentType.BHOOMI_RTC)
        self.assertEqual(len(ocr_result.text_lines), 4)

        # HandwritingResult check
        self.assertIsNotNone(hw_result)
        self.assertIsInstance(hw_result, HandwritingResult)
        self.assertEqual(len(hw_result.regions), 1)
        self.assertEqual(hw_result.regions[0].region_id, "reg_003")
        self.assertEqual(hw_result.regions[0].text, "ಖಾತೆದಾರರ ಹೆಸರು: ರಮೇಶ್ ಕುಮಾರ್")
        self.assertEqual(hw_result.regions[0].confidence, 0.92)

    def test_execute_person_c_extraction_and_validation(self):
        """Validates that Person C extraction executes on Person B output and extracts fields."""
        final_doc = PersonCAdapter.execute_person_c(
            b_response=self.karnataka_b_response,
            selected_state="KA",
        )

        self.assertIsInstance(final_doc, FinalDocumentResult)
        self.assertEqual(final_doc.document_id, "DOC_B_KA_001")
        self.assertEqual(final_doc.state, "KA")

        # Verify extracted fields
        self.assertIn("khasra_number", final_doc.fields)
        self.assertEqual(final_doc.fields["khasra_number"].raw_value, "42/1")
        self.assertEqual(final_doc.fields["khasra_number"].normalized_value, "42/1")

        # Verify confidence & audit trail
        self.assertGreater(final_doc.overall_confidence, 0.70)
        self.assertIn("extraction", final_doc.pipeline_stages_completed)
        self.assertIn("normalization", final_doc.pipeline_stages_completed)
        self.assertIn("rule_validation", final_doc.pipeline_stages_completed)

    def test_to_unified_response_structure(self):
        """Validates the structure of the final unified dictionary."""
        final_doc = PersonCAdapter.execute_person_c(
            b_response=self.karnataka_b_response,
            selected_state="KA",
        )
        unified = PersonCAdapter.to_unified_response(self.karnataka_b_response, final_doc)

        # Check top-level contract keys
        self.assertEqual(unified["document_id"], "DOC_B_KA_001")
        self.assertIn("ocr", unified)
        self.assertIn("extracted_fields", unified)
        self.assertIn("validation", unified)
        self.assertIn("gis_validation", unified)
        self.assertIn("duplicate_analysis", unified)
        self.assertIn("overall_confidence", unified)
        self.assertIn("requires_human_review", unified)

        # Check preserved Person B OCR data
        self.assertEqual(unified["ocr"]["merged_text"], self.karnataka_b_response.merged_text)
        self.assertEqual(len(unified["ocr"]["ordered_regions"]), 4)

        # Check Person C extracted field details
        self.assertIn("survey_number", unified["extracted_fields"])
        field = unified["extracted_fields"]["survey_number"]
        self.assertEqual(field["raw_value"], "42/1")
        self.assertIn("confidence", field)
        self.assertIn("validation_status", field)


if __name__ == "__main__":
    unittest.main()
