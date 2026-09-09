"""Unit Tests for Handwriting OCR Evaluation Metrics (CER and WER)."""

import unittest

from src.training.evaluate import (
    compute_cer,
    compute_wer,
    evaluate_predictions,
    levenshtein_distance,
)


class TestTrainingEvaluationMetrics(unittest.TestCase):
    """Test suite for CER, WER, and per-language evaluation aggregation."""

    def test_levenshtein_distance_exactness(self):
        """Tests Levenshtein edit distance calculations."""
        # Identical sequences
        self.assertEqual(levenshtein_distance("kannada", "kannada"), 0)

        # 1 substitution: 'k' -> 'c'
        self.assertEqual(levenshtein_distance("kannada", "cannada"), 1)

        # 1 insertion: 'kannada' -> 'kannadaa'
        self.assertEqual(levenshtein_distance("kannada", "kannadaa"), 1)

        # 1 deletion: 'kannada' -> 'kannad'
        self.assertEqual(levenshtein_distance("kannada", "kannad"), 1)

        # Unicode Indic character edit distance
        # "ಸರ್ವೆ" (s-a-r-v-e) -> "ಸರ್ವ" (1 deletion)
        self.assertEqual(levenshtein_distance("ಸರ್ವೆ", "ಸರ್ವ"), 1)

    def test_compute_cer(self):
        """Tests Character Error Rate (CER) calculation."""
        # Perfect match
        self.assertAlmostEqual(compute_cer("Survey 45", "Survey 45"), 0.0)

        # 1 edit on 10 characters = 0.1
        self.assertAlmostEqual(compute_cer("1234567890", "123456789X"), 0.1)

        # Empty reference vs non-empty hypothesis
        self.assertAlmostEqual(compute_cer("", "text"), 1.0)

        # Both empty
        self.assertAlmostEqual(compute_cer("", ""), 0.0)

    def test_compute_wer(self):
        """Tests Word Error Rate (WER) calculation."""
        # Perfect match
        self.assertAlmostEqual(compute_wer("Survey No 45", "Survey No 45"), 0.0)

        # 1 word wrong out of 3 words = 1/3 = 0.3333...
        self.assertAlmostEqual(compute_wer("Survey No 45", "Survey Number 45"), 1.0 / 3.0)

        # Empty reference
        self.assertAlmostEqual(compute_wer("", "word"), 1.0)
        self.assertAlmostEqual(compute_wer("", ""), 0.0)

    def test_evaluate_predictions_multilingual_aggregation(self):
        """Tests global and per-language metrics aggregation."""
        references = [
            "ಕರ್ನಾಟಕ ಕಂದಾಯ",     # Kannada: 13 chars
            "ಸರ್ವೆ ನಂ ೧೨",        # Kannada: 11 chars
            "ఆంధ్రప్రదేశ్ రెవెన్యూ", # Telugu: 21 chars
            "खसरा संख्या",        # Hindi: 11 chars
        ]

        hypotheses = [
            "ಕರ್ನಾಟಕ ಕಂದಾಯ",     # Exact match (0 edits)
            "ಸರ್ವೆ ನಂ ೧೦",        # 1 char edit ('೨' -> '೦')
            "ఆంధ్రప్రదేశ్ రికార్డు", # 5 char edits ('రెవెన్యూ' vs 'రికార్డు')
            "खसरा संख्या",        # Exact match (0 edits)
        ]

        languages = ["kannada", "kannada", "telugu", "hindi"]

        report = evaluate_predictions(
            references=references,
            hypotheses=hypotheses,
            languages=languages,
        )

        self.assertEqual(report.total_samples, 4)
        self.assertIn("kannada", report.per_language)
        self.assertIn("telugu", report.per_language)
        self.assertIn("hindi", report.per_language)

        # Kannada metrics: 1 edit over 24 total chars = 1/24 ≈ 0.0417
        self.assertEqual(report.per_language["kannada"]["sample_count"], 2)
        self.assertAlmostEqual(report.per_language["kannada"]["cer"], round(1.0 / 24.0, 4), places=4)

        # Hindi metrics: 0 edits -> CER = 0.0
        self.assertEqual(report.per_language["hindi"]["sample_count"], 1)
        self.assertEqual(report.per_language["hindi"]["cer"], 0.0)

        # Overall CER is calculated and > 0
        self.assertGreater(report.overall_cer, 0.0)

    def test_evaluate_predictions_mismatched_lengths_raise_error(self):
        """Tests that mismatched reference and hypothesis list lengths raise ValueError."""
        with self.assertRaises(ValueError):
            evaluate_predictions(
                references=["a", "b"],
                hypotheses=["a"],
            )

        with self.assertRaises(ValueError):
            evaluate_predictions(
                references=["a", "b"],
                hypotheses=["a", "b"],
                languages=["kannada"],
            )


if __name__ == "__main__":
    unittest.main()
