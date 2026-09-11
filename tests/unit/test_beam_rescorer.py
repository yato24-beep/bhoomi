import unittest
from src.postprocessing.beam_rescorer import KannadaBeamRescorer, RescoringResult


class TestKannadaBeamRescorer(unittest.TestCase):
    """Unit tests for generic multi-signal beam search candidate rescorer."""

    def setUp(self):
        self.rescorer = KannadaBeamRescorer(custom_lexicon=["ಕೋತಿ", "ಮರ", "ಹಣ್ಣು", "ಸರ್ವೆ", "ಖಾತೆ"])

    def test_orthographic_validity(self):
        # Valid Kannada words
        self.assertAlmostEqual(self.rescorer.check_orthographic_validity("ಕೋತಿ"), 1.0)
        self.assertAlmostEqual(self.rescorer.check_orthographic_validity("ಮರ"), 1.0)
        
        # Invalid floating matra at start
        self.assertLess(self.rescorer.check_orthographic_validity("\u0CBFಕೋತಿ"), 1.0)

    def test_lexicon_similarity(self):
        # Exact match
        sim_exact, match_exact = self.rescorer.evaluate_lexicon_similarity("ಕೋತಿ")
        self.assertAlmostEqual(sim_exact, 1.0)
        self.assertEqual(match_exact, "ಕೋತಿ")

        # Close edit match
        sim_close, match_close = self.rescorer.evaluate_lexicon_similarity("ಕೋಟಿ")
        self.assertGreaterEqual(sim_close, 0.70)
        self.assertIsNotNone(match_close)

    def test_rescoring_switch_when_alternative_is_exact_word(self):
        # Simulated candidates where rank 1 is 'ಕೋಟಿ' and rank 4 is exact word 'ಕೋತಿ'
        candidates = [
            {"text": "ಕೋಟಿ", "score": -1.2, "token_ids": [33694, 12463]},
            {"text": "ಕೋಡಿ", "score": -1.5, "token_ids": [33694, 6540]},
            {"text": "ಕೇತಿ", "score": -1.6, "token_ids": [65303, 9128]},
            {"text": "ಕೋತಿ", "score": -1.7, "token_ids": [33694, 9128]},
            {"text": "ಕೊಳ", "score": -2.0, "token_ids": [55460, 22953]},
        ]
        result: RescoringResult = self.rescorer.rescore(candidates)
        self.assertEqual(result.original_top_candidate, "ಕೋಟಿ")
        self.assertEqual(result.rescored_candidate, "ಕೋತಿ")
        self.assertEqual(result.final_prediction, "ಕೋತಿ")
        self.assertTrue(result.is_switched)

    def test_rescoring_preserves_top_when_no_strong_evidence(self):
        # When all candidates are unknown or rank 1 is already best
        candidates = [
            {"text": "ಅನನ್ಯ", "score": -0.5, "token_ids": [100, 200]},
            {"text": "ಅನನ್ಯಾ", "score": -1.8, "token_ids": [100, 201]},
        ]
        result: RescoringResult = self.rescorer.rescore(candidates)
        self.assertEqual(result.final_prediction, "ಅನನ್ಯ")
        self.assertFalse(result.is_switched)


if __name__ == "__main__":
    unittest.main()
