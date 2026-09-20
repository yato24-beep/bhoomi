"""Unit tests for Text Detection & Geometric Spacing Reconstruction."""

import unittest
from PIL import Image
from schemas import BoundingBox
from src.detection.text_detector import (
    DetectedLineGroup,
    DetectedWordBox,
    DocumentTextDetector,
)


class TestTextDetectorSpacing(unittest.TestCase):
    def test_spacing_reconstruction_from_gaps(self):
        line_bbox = BoundingBox(x_min=10, y_min=20, x_max=300, y_max=60)
        # Line height = 40. space threshold = 0.22 * 40 = 8.8 px
        # Box 1: [10, 50]
        # Box 2: [75, 120] -> gap = 25 (> 8.8) -> inserts space
        # Box 3: [122, 170] -> gap = 2 (< 8.8) -> no space
        b1 = DetectedWordBox(bbox=BoundingBox(x_min=10, y_min=20, x_max=50, y_max=60))
        b2 = DetectedWordBox(bbox=BoundingBox(x_min=75, y_min=20, x_max=120, y_max=60))
        b3 = DetectedWordBox(bbox=BoundingBox(x_min=122, y_min=20, x_max=170, y_max=60))

        lg = DetectedLineGroup(line_index=1, line_bbox=line_bbox, word_boxes=[b1, b2, b3])
        reconstructed = lg.reconstruct_line_text(["word1", "word2:", "suffix"])
        self.assertEqual(reconstructed, "word1 word2:suffix")

    def test_detector_clustering_mock(self):
        img = Image.new("RGB", (400, 300), color=(255, 255, 255))
        detector = DocumentTextDetector()
        # Line 1 boxes
        b1 = BoundingBox(x_min=10, y_min=20, x_max=80, y_max=50)
        b2 = BoundingBox(x_min=100, y_min=22, x_max=160, y_max=52)
        # Line 2 boxes
        b3 = BoundingBox(x_min=10, y_min=80, x_max=90, y_max=110)
        b4 = BoundingBox(x_min=120, y_min=82, x_max=190, y_max=112)

        clusters = detector._cluster_into_lines(img, [b4, b1, b3, b2])
        self.assertEqual(len(clusters), 2)
        self.assertEqual(len(clusters[0].word_boxes), 2)
        self.assertEqual(len(clusters[1].word_boxes), 2)


if __name__ == "__main__":
    unittest.main()
