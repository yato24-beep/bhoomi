"""Unit Tests for Dataset Readiness Audit Utilities."""

import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.training.dataset_audit import (
    classify_sample_text_type,
    run_dataset_audit,
)


class TestDatasetAudit(unittest.TestCase):
    """Test suite for dataset readiness audit scanning and reporting."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

        self.data_dir = self.base_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.manifest_dir = self.base_dir / "training" / "datasets"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        # Create physical image files
        self.img1 = self.raw_dir / "kannada_crop_01.png"
        self.img2 = self.raw_dir / "english_hw_01.png"
        Image.new("RGB", (100, 30), color=255).save(self.img1)
        Image.new("RGB", (100, 30), color=255).save(self.img2)

        # Create sample manifest
        self.manifest = self.manifest_dir / "train.jsonl"
        with open(self.manifest, "w", encoding="utf-8") as f:
            f.write(
                json.dumps({
                    "image": str(self.img1),
                    "text": "ಕರ್ನಾಟಕ",
                    "language": "kannada",
                    "metadata": {"text_type": "printed"},
                }) + "\n"
            )
            f.write(
                json.dumps({
                    "image": str(self.img2),
                    "text": "Survey 45",
                    "language": "english",
                    "metadata": {"text_type": "handwriting"},
                }) + "\n"
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_classify_sample_text_type(self):
        """Tests text type classification heuristics (real handwriting vs printed vs synthetic)."""
        self.assertEqual(classify_sample_text_type({"text_type": "handwriting"}, "crop.png"), "real_handwriting")
        self.assertEqual(classify_sample_text_type({"text_type": "printed"}, "crop.png"), "printed")
        self.assertEqual(classify_sample_text_type({"text_type": "synthetic"}, "crop.png"), "synthetic")
        self.assertEqual(classify_sample_text_type({}, "sample_handwritten_crop.png"), "synthetic")
        self.assertEqual(classify_sample_text_type({}, "kannada_crop.png"), "printed")

    def test_run_dataset_audit_accuracy(self):
        """Tests audit scan results on test environment."""
        report = run_dataset_audit(
            root_dir=self.base_dir,
            data_dir=self.data_dir,
            manifest_dir=self.manifest_dir,
        )

        self.assertEqual(report.total_physical_images, 2)
        self.assertEqual(report.total_manifest_records, 2)
        self.assertEqual(report.valid_manifest_records, 2)
        self.assertEqual(report.missing_manifest_records, 0)

        # Language counts
        self.assertEqual(report.samples_by_language.get("kannada"), 1)
        self.assertEqual(report.samples_by_language.get("english"), 1)

        # Text type counts
        self.assertEqual(report.samples_by_type.get("printed"), 1)
        self.assertEqual(report.samples_by_type.get("real_handwriting"), 1)

        # Sufficiency evaluation
        self.assertTrue(report.sufficiency["smoke_testing"]["is_sufficient"])
        self.assertFalse(report.sufficiency["small_fine_tuning"]["is_sufficient"])
        self.assertFalse(report.sufficiency["production_training"]["is_sufficient"])

        # Markdown report generation
        md = report.to_markdown()
        self.assertIn("# Dataset Readiness Audit Report", md)
        self.assertIn("Kannada", md)


if __name__ == "__main__":
    unittest.main()
