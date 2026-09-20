"""Unit tests for Visual Land Record Layout Gating Classifier."""

import unittest
from PIL import Image, ImageDraw
from src.classification.document_gate import LandRecordGateClassifier, GateClassificationResult


class TestDocumentGate(unittest.TestCase):
    def setUp(self):
        self.gate = LandRecordGateClassifier()

    def test_reject_tiny_or_blank_image(self):
        img = Image.new("RGB", (40, 40), color=(255, 255, 255))
        res = self.gate.classify_image(img)
        self.assertFalse(res.is_land_record)
        self.assertEqual(res.document_layout_type, "not_land_record")
        self.assertIn("too small", res.rejection_reason)

    def test_reject_extreme_aspect_ratio(self):
        img = Image.new("RGB", (1000, 50), color=(255, 255, 255))
        res = self.gate.classify_image(img)
        self.assertFalse(res.is_land_record)
        self.assertEqual(res.document_layout_type, "not_land_record")

    def test_accept_tabular_land_record_grid(self):
        img = Image.new("RGB", (600, 800), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        for y in [100, 200, 300, 400, 500]:
            draw.line([(50, y), (550, y)], fill=(0, 0, 0), width=3)
        for x in [50, 200, 400, 550]:
            draw.line([(x, 100), (x, 500)], fill=(0, 0, 0), width=3)

        res = self.gate.classify_image(img)
        self.assertTrue(res.is_land_record)
        self.assertIn("grid", res.document_layout_type)
        self.assertGreater(res.confidence, 0.70)


if __name__ == "__main__":
    unittest.main()
