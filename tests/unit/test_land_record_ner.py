"""Unit tests for Gazetteer and Label-Anchor Land Record NER Extractor."""

import unittest
from src.extraction.land_record_ner import LandRecordFieldExtractor


class TestLandRecordNER(unittest.TestCase):
    def setUp(self):
        self.extractor = LandRecordFieldExtractor(confidence_threshold=0.60)

    def test_extract_all_canonical_rtc_fields(self):
        lines = [
            "ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಕಂದಾಯ ಇಲಾಖೆ",
            "ಮಾಲೀಕರ ಹೆಸರು: ರಮೇಶ್ ಕುಮಾರ್ ಬಿನ್ ಮಂಜುನಾಥ್",
            "ಸರ್ವೆ ನಂಬರ್: ೧೪೨/೩",
            "ಖಾತಾ ಸಂಖ್ಯೆ: ೪೮೯",
            "ತಾಲೂಕು: ಬೆಂಗಳೂರು ಉತ್ತರ ಗ್ರಾಮ: ಕೆಂಗೇರಿ",
            "ವಿಸ್ತೀರ್ಣ: ೨ ಎಕರೆ ೧೪ ಗುಂಟೆ",
        ]
        fields = self.extractor.extract_fields(lines)

        self.assertIn("owner_name", fields)
        self.assertEqual(fields["owner_name"].normalized_value, "ರಮೇಶ್ ಕುಮಾರ್ ಬಿನ್ ಮಂಜುನಾಥ್")
        self.assertFalse(fields["owner_name"].requires_human_review)

        self.assertIn("survey_number", fields)
        self.assertEqual(fields["survey_number"].normalized_value, "142/3")

        self.assertIn("khata_number", fields)
        self.assertEqual(fields["khata_number"].normalized_value, "489")

        self.assertIn("taluk", fields)
        self.assertIn("ಬೆಂಗಳೂರು", fields["taluk"].normalized_value)

        self.assertIn("village", fields)
        self.assertEqual(fields["village"].normalized_value, "ಕೆಂಗೇರಿ")

        self.assertIn("extent_area", fields)
        self.assertIn("2", fields["extent_area"].normalized_value)


if __name__ == "__main__":
    unittest.main()
