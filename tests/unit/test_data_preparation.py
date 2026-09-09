"""Unit Tests for Dataset Preparation and Validation Pipeline."""

import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.training.data_preparation import (
    ValidationIssue,
    ValidationReport,
    clean_and_save_manifest,
    normalize_unicode_text,
    validate_image_file,
    validate_manifest,
)


class TestDataPreparation(unittest.TestCase):
    """Test suite for data preparation and validation utilities."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

        # Create valid dummy images
        self.valid_img1 = self.base_dir / "valid_01.png"
        self.valid_img2 = self.base_dir / "valid_02.png"
        self.corrupt_img = self.base_dir / "corrupt.png"
        self.empty_img = self.base_dir / "empty.png"

        img = Image.new("RGB", (100, 40), color=(255, 255, 255))
        img.save(self.valid_img1)
        img.save(self.valid_img2)

        # Corrupt file (random invalid bytes)
        with open(self.corrupt_img, "wb") as f:
            f.write(b"NOT_A_VALID_PNG_IMAGE_DATA_BYTES")

        # 0-byte empty file
        with open(self.empty_img, "wb") as f:
            pass

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validate_image_file(self):
        """Tests image file existence, dimension, and decoder validation."""
        # 1. Valid image
        is_val, err, size = validate_image_file(self.valid_img1)
        self.assertTrue(is_val)
        self.assertIsNone(err)
        self.assertEqual(size, (100, 40))

        # 2. Missing image
        is_val, err, _ = validate_image_file(self.base_dir / "non_existent.png")
        self.assertFalse(is_val)
        self.assertIn("not found", err.lower())

        # 3. Empty (0-byte) file
        is_val, err, _ = validate_image_file(self.empty_img)
        self.assertFalse(is_val)
        self.assertIn("empty", err.lower())

        # 4. Corrupt file
        is_val, err, _ = validate_image_file(self.corrupt_img)
        self.assertFalse(is_val)
        self.assertIn("cannot decode", err.lower())

    def test_normalize_unicode_text(self):
        """Tests Unicode NFC normalization on Indic text."""
        raw_text = "  ಕರ್ನಾಟಕ ಕಂದಾಯ   "
        normalized = normalize_unicode_text(raw_text)
        self.assertEqual(normalized, "ಕರ್ನಾಟಕ ಕಂದಾಯ")
        self.assertEqual(normalize_unicode_text(""), "")

    def test_validate_manifest_detects_all_issues(self):
        """Tests comprehensive detection of missing files, empty text, duplicates, and malformed JSON."""
        manifest_file = self.base_dir / "mixed_manifest.jsonl"
        with open(manifest_file, "w", encoding="utf-8") as f:
            # 1. Valid Kannada record
            f.write(json.dumps({"image": str(self.valid_img1), "text": "ಸರ್ವೆ ನಂ ೧೨", "language": "kannada"}) + "\n")
            # 2. Missing image file
            f.write(json.dumps({"image": "missing_crop.png", "text": "missing file record"}) + "\n")
            # 3. Corrupt image file
            f.write(json.dumps({"image": str(self.corrupt_img), "text": "corrupt image record"}) + "\n")
            # 4. Empty text
            f.write(json.dumps({"image": str(self.valid_img2), "text": "   "}) + "\n")
            # 5. Duplicate image path (refers to self.valid_img1 again)
            f.write(json.dumps({"image": str(self.valid_img1), "text": "duplicate path record"}) + "\n")
            # 6. Malformed JSON line
            f.write("{invalid_json_line\n")

        valid_samples, report = validate_manifest(
            manifest_path=manifest_file,
            root_dir=self.base_dir,
            verify_images=True,
            allow_duplicates=False,
        )

        self.assertEqual(report.total_records, 6)
        self.assertEqual(len(valid_samples), 1)
        self.assertEqual(report.valid_count, 1)
        self.assertFalse(report.is_valid)
        self.assertGreater(report.error_count, 0)
        self.assertEqual(report.duplicate_count, 1)

        issue_types = [issue.issue_type for issue in report.issues]
        self.assertIn("missing_image", issue_types)
        self.assertIn("corrupt_image", issue_types)
        self.assertIn("empty_text", issue_types)
        self.assertIn("duplicate_path", issue_types)
        self.assertIn("invalid_json", issue_types)

    def test_clean_and_save_manifest(self):
        """Tests that clean_and_save_manifest purges bad records and writes clean JSONL."""
        input_manifest = self.base_dir / "dirty_manifest.jsonl"
        output_manifest = self.base_dir / "clean_manifest.jsonl"

        with open(input_manifest, "w", encoding="utf-8") as f:
            f.write(json.dumps({"image": str(self.valid_img1), "text": "valid 1", "language": "kannada"}) + "\n")
            f.write(json.dumps({"image": "missing.png", "text": "bad 1"}) + "\n")
            f.write(json.dumps({"image": str(self.valid_img2), "text": "valid 2", "language": "telugu"}) + "\n")

        saved_count, report = clean_and_save_manifest(
            input_manifest=input_manifest,
            output_manifest=output_manifest,
            root_dir=self.base_dir,
            verify_images=True,
        )

        self.assertEqual(saved_count, 2)
        self.assertTrue(output_manifest.exists())

        # Verify output JSONL content
        with open(output_manifest, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["text"], "valid 1")
        self.assertEqual(lines[1]["text"], "valid 2")


if __name__ == "__main__":
    unittest.main()
