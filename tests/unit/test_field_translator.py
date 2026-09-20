"""Unit tests for Word-Aligned Translation & Phonetic Transliteration."""

import unittest
from src.translation.field_translator import (
    FieldAlignedTranslator,
    transliterate_kannada_phrase,
    transliterate_kannada_word,
)


class TestFieldTranslator(unittest.TestCase):
    def setUp(self):
        self.translator = FieldAlignedTranslator()

    def test_proper_noun_transliteration(self):
        name = transliterate_kannada_phrase("ರಮೇಶ್ ಕುಮಾರ್ ಬಿನ್ ಮಂಜುನಾಥ್")
        self.assertIn("Ramesh", name)
        self.assertIn("Kumar", name)
        self.assertIn("Manjunath", name)

        place = transliterate_kannada_phrase("ಬೆಂಗಳೂರು ಉತ್ತರ")
        self.assertIn("Bengaluru", place)

    def test_structured_field_alignment(self):
        fields = {
            "owner_name": type("F", (), {"normalized_value": "ರಮೇಶ್ ಕುಮಾರ್", "confidence": 0.95})(),
            "survey_number": type("F", (), {"normalized_value": "142/3", "confidence": 0.94})(),
            "taluk": type("F", (), {"normalized_value": "ಬೆಂಗಳೂರು", "confidence": 0.92})(),
            "extent_area": type("F", (), {"normalized_value": "2 ಎಕರೆ 14 ಗುಂಟೆ", "confidence": 0.90})(),
        }
        bilingual = self.translator.translate_structured_fields(fields)

        self.assertEqual(bilingual["owner_name"].english_value, "Ramesh Kumar")
        self.assertIn(bilingual["owner_name"].method, ("transliteration", "domain_glossary", "google_neural", "aksharamukha_phonetic"))

        self.assertIsNone(bilingual["survey_number"].english_value)
        self.assertEqual(bilingual["survey_number"].method, "passthrough")

        self.assertIn("Acres", bilingual["extent_area"].english_value)
        self.assertIn("Guntas", bilingual["extent_area"].english_value)


if __name__ == "__main__":
    unittest.main()
