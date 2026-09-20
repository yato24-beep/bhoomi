"""Comprehensive Unit and Integration Tests for Field Translation Pipeline.

Verifies:
1. Kannada owner name -> English transliteration (e.g. ಸಿದ್ದರಾಮಯ್ಯ -> Siddaramaiah)
2. Kannada village/taluk/district (e.g. ಬೆಂಗಳೂರು -> Bengaluru, ಯಲಹಂಕ -> Yelahanka)
3. Already-English value passthrough with ALREADY_ENGLISH status
4. Numeric survey number non-translation with NOT_APPLICABLE status
5. Date non-translation with NOT_APPLICABLE status
6. Translation failure non-fabrication guarantee (returns None, never invents an English name)
7. Translation service timeout fallback without crash or fabrication
8. Provenance preservation intact across translation
9. API serialization contract with bilingual fields
10. Kannada text permanent retention guarantee
"""

import unittest
from unittest.mock import patch, MagicMock
from src.translation.field_translator import (
    FieldAlignedTranslator,
    transliterate_kannada_phrase,
    transliterate_kannada_word,
    has_kannada_script,
)
from src.semantic.schema import SemanticFieldItem, ValidationStatus, FieldProvenance, BoundingBox


class TestFieldTranslationPipeline(unittest.TestCase):
    def setUp(self):
        self.translator = FieldAlignedTranslator(enable_online=True, timeout=2.0)

    def test_kannada_owner_name_transliteration(self):
        """Owner name in Kannada transliterates phonetically to English."""
        # Test known leader / common proper name
        res = self.translator.translate_field(
            field_name="owner_name",
            value="ಸಿದ್ದರಾಮಯ್ಯ",
            raw_value="ಸಿದ್ದರಾಮಯ್ಯ",
            provenance={"region_id": "reg_001"},
        )
        self.assertEqual(res["raw_value"], "ಸಿದ್ದರಾಮಯ್ಯ")
        self.assertEqual(res["normalized_value"], "ಸಿದ್ದರಾಮಯ್ಯ")
        self.assertEqual(res["translation_status"], "TRANSLATED")
        self.assertIn(res["english_value"], ["Siddaramaiah", "Siddaramayya"])
        self.assertIsNotNone(res["translation_engine"])
        self.assertEqual(res["provenance"], {"region_id": "reg_001"})

        # Another common patronymic name
        res2 = self.translator.translate_field(
            field_name="owner_name",
            value="ರಮೇಶ್ ಕುಮಾರ್",
            raw_value="ರಮೇಶ್ ಕುಮಾರ್",
        )
        self.assertEqual(res2["translation_status"], "TRANSLATED")
        self.assertIn("Ramesh", res2["english_value"])
        self.assertIn("Kumar", res2["english_value"])

    def test_kannada_geography_fields(self):
        """District, Taluk, and Village translate to canonical administrative English."""
        # Bengaluru District
        res_dist = self.translator.translate_field(
            field_name="district",
            value="ಬೆಂಗಳೂರು",
            raw_value="ಬೆಂಗಳೂರು",
            provenance={"region_id": "reg_dist"},
        )
        self.assertEqual(res_dist["raw_value"], "ಬೆಂಗಳೂರು")
        self.assertEqual(res_dist["english_value"], "Bengaluru")
        self.assertEqual(res_dist["translation_status"], "TRANSLATED")
        self.assertEqual(res_dist["provenance"], {"region_id": "reg_dist"})

        # Yelahanka Taluk
        res_taluk = self.translator.translate_field(
            field_name="taluk",
            value="ಯಲಹಂಕ",
            raw_value="ಯಲಹಂಕ",
        )
        self.assertEqual(res_taluk["english_value"], "Yelahanka")
        self.assertEqual(res_taluk["translation_status"], "TRANSLATED")

        # Kasaba Hobli
        res_hobli = self.translator.translate_field(
            field_name="hobli",
            value="ಕಸಬಾ",
            raw_value="ಕಸಬಾ",
        )
        self.assertEqual(res_hobli["english_value"], "Kasaba")
        self.assertEqual(res_hobli["translation_status"], "TRANSLATED")

    def test_already_english_value(self):
        """Values already written in Latin script pass through without modification."""
        res = self.translator.translate_field(
            field_name="owner_name",
            value="John Doe",
            raw_value="John Doe",
        )
        self.assertEqual(res["raw_value"], "John Doe")
        self.assertEqual(res["normalized_value"], "John Doe")
        self.assertEqual(res["english_value"], "John Doe")
        self.assertEqual(res["translation_status"], "ALREADY_ENGLISH")
        self.assertEqual(res["translation_engine"], "passthrough")

    def test_survey_number_remains_unchanged(self):
        """Structured survey identifier must NOT be translated or converted."""
        res = self.translator.translate_field(
            field_name="survey_number",
            value="124/2A",
            raw_value="124/2A",
            provenance={"region_id": "reg_survey"},
        )
        self.assertEqual(res["raw_value"], "124/2A")
        self.assertEqual(res["normalized_value"], "124/2A")
        self.assertIsNone(res["english_value"])
        self.assertEqual(res["translation_status"], "NOT_APPLICABLE")
        self.assertEqual(res["translation_engine"], "passthrough")
        self.assertEqual(res["provenance"], {"region_id": "reg_survey"})

    def test_date_remains_unchanged(self):
        """Dates are structured temporal values and must NOT be translated."""
        for date_val in ["15/08/2023", "2023-04-12"]:
            res = self.translator.translate_field(
                field_name="document_date",
                value=date_val,
                raw_value=date_val,
            )
            self.assertEqual(res["raw_value"], date_val)
            self.assertIsNone(res["english_value"])
            self.assertEqual(res["translation_status"], "NOT_APPLICABLE")
            self.assertEqual(res["translation_engine"], "passthrough")

    def test_translation_failure_no_fabrication(self):
        """When all translation sources fail, return None without inventing a name."""
        offline_trans = FieldAlignedTranslator(enable_online=False)
        # Mock transliterator to simulate an internal exception
        with patch("src.translation.field_translator.transliterate_kannada_phrase", side_effect=RuntimeError("Engine crash")):
            res = offline_trans.translate_field(
                field_name="owner_name",
                value="ವಿಶೇಷಪದ",  # Word not in domain glossary
                raw_value="ವಿಶೇಷಪದ",
                provenance={"region_id": "reg_fail"},
            )
            self.assertIsNone(res["english_value"])
            self.assertEqual(res["translation_status"], "FAILED")
            self.assertEqual(res["translation_engine"], "unavailable")
            self.assertEqual(res["raw_value"], "ವಿಶೇಷಪದ")
            self.assertEqual(res["provenance"], {"region_id": "reg_fail"})

    def test_translation_timeout_handling(self):
        """When external service times out, system falls back gracefully to phonetic transliteration."""
        with patch("src.translation.translator.translate_kannada_to_english_online", side_effect=TimeoutError("Online service timeout")):
            res = self.translator.translate_field(
                field_name="village",
                value="ಅರಳೀಮರದಹಳ್ಳಿ",
                raw_value="ಅರಳೀಮರದಹಳ್ಳಿ",
            )
            # Should not raise exception and should fallback gracefully
            self.assertIsNotNone(res["english_value"])
            self.assertEqual(res["translation_status"], "TRANSLATED")
            self.assertEqual(res["translation_engine"], "aksharamukha_phonetic")
            self.assertFalse(has_kannada_script(res["english_value"]))

    def test_provenance_remains_intact(self):
        """Every field translation retains exact source region bounding boxes and provenance."""
        mock_prov = {
            "page_number": 1,
            "region_id": "reg_header_12",
            "bbox": {"x_min": 0.12, "y_min": 0.34, "x_max": 0.56, "y_max": 0.40},
            "raw_ocr_text": "ಬೆಂಗಳೂರು",
        }
        res = self.translator.translate_field(
            field_name="district",
            value="ಬೆಂಗಳೂರು",
            raw_value="ಬೆಂಗಳೂರು",
            provenance=mock_prov,
        )
        self.assertEqual(res["provenance"], mock_prov)
        self.assertEqual(res["provenance"]["region_id"], "reg_header_12")
        self.assertEqual(res["raw_value"], "ಬೆಂಗಳೂರು")

    def test_semantic_field_item_model_integration(self):
        """Verify SemanticFieldItem Pydantic model can carry translation attributes."""
        fitem = SemanticFieldItem(
            field_name="owner_name",
            value="ಸಿದ್ದರಾಮಯ್ಯ",
            raw_value="ಸಿದ್ದರಾಮಯ್ಯ",
            validation_status=ValidationStatus.VALID,
            provenance=FieldProvenance(
                page_number=1,
                region_id="reg_01",
                bbox=BoundingBox(x_min=10, y_min=20, x_max=100, y_max=50),
                raw_ocr_text="ಸಿದ್ದರಾಮಯ್ಯ",
                normalized_value="ಸಿದ್ದರಾಮಯ್ಯ",
            ),
            english_value="Siddaramaiah",
            translation_status="TRANSLATED",
            translation_engine="google_neural",
        )
        d = fitem.model_dump()
        self.assertEqual(d["english_value"], "Siddaramaiah")
        self.assertEqual(d["translation_status"], "TRANSLATED")
        self.assertEqual(d["translation_engine"], "google_neural")
        self.assertEqual(d["raw_value"], "ಸಿದ್ದರಾಮಯ್ಯ")
        self.assertEqual(d["provenance"]["region_id"], "reg_01")


if __name__ == "__main__":
    unittest.main()
