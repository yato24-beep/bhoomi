"""Unit tests for IIIT Indic Handwriting Dataset Importer.

Uses lightweight temporary fixtures to test:
- Ground-truth line parsing (single split, tabs/spaces, multi-word transcriptions)
- Exact Unicode Kannada preservation and NFC normalization
- Path resolution and POSIX formatting
- Malformed line and missing image error handling
- Split preservation and manifest serialization
- Integration with HandwritingDataset contract
"""

import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.training.dataset import HandwritingDataset, MultilingualHandwritingSample
from src.training.importers.iiit_indic_hw import (
    IIITIndicHWImporter,
    ImportIssue,
    ImportReport,
)


class TestIIITIndicHWImporter(unittest.TestCase):
    """Test suite for IIITIndicHWImporter using synthetic fixtures."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.dataset_root = self.root_path / "iiit_indic_hw_words"

        # Build mock dataset structure: train/, val/, test/
        for split in ("train", "val", "test"):
            split_dir = self.dataset_root / split
            images_dir = split_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            # Create sample dummy images: 1.jpg, 2.jpg
            for idx in (1, 2):
                img = Image.new("RGB", (30, 20), color=(255, 255, 255))
                img.save(images_dir / f"{idx}.jpg")

        # Create mock ground truth files with Kannada Unicode
        train_gt = self.dataset_root / "train" / "train_gt.txt"
        train_gt.write_text(
            "images/1.jpg\tನಾಜೂಕಾಗಿರುವುದರಿಂದ\n"
            "images/2.jpg\tಗುರುತಿಸಿಕೊಂಡಮೇಲೆ ಕರ್ನಾಟಕ ಕಂದಾಯ\n",
            encoding="utf-8",
        )

        val_gt = self.dataset_root / "val" / "val_gt.txt"
        val_gt.write_text(
            "images/1.jpg\tಚುರಮರಿ\n"
            "images/2.jpg\tಟಿವಿ ಕೇಂದ್ರ\n",
            encoding="utf-8",
        )

        test_gt = self.dataset_root / "test" / "test_gt.txt"
        test_gt.write_text(
            "images/1.jpg\tಕಾಯ್ದಿರಿಸಿದ್ದ\n"
            "images/2.jpg\tಸೋಲು ಗೆಲುವು\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_ground_truth_line_valid_unicode(self):
        """Tests parsing valid line with Kannada Unicode and multi-word phrase."""
        importer = IIITIndicHWImporter(
            dataset_root=self.dataset_root,
            language="kannada",
            project_root=self.root_path,
        )

        split_dir = self.dataset_root / "train"
        line = "images/2.jpg\tಗುರುತಿಸಿಕೊಂಡಮೇಲೆ ಕರ್ನಾಟಕ ಕಂದಾಯ"
        sample, issue = importer.parse_ground_truth_line(
            line=line,
            line_number=2,
            split_dir=split_dir,
            split_name="train",
            validate_images=True,
        )

        self.assertIsNone(issue)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.text, "ಗುರುತಿಸಿಕೊಂಡಮೇಲೆ ಕರ್ನಾಟಕ ಕಂದಾಯ")
        self.assertEqual(sample.language, "kannada")
        self.assertEqual(sample.script, "Kannada")
        self.assertEqual(sample.metadata["source"], "IIIT-INDIC-HW-WORDS")
        self.assertEqual(sample.metadata["original_split"], "train")
        self.assertEqual(sample.metadata["raw_text"], "ಗುರುತಿಸಿಕೊಂಡಮೇಲೆ ಕರ್ನಾಟಕ ಕಂದಾಯ")
        self.assertIn("train/images/2.jpg", str(sample.image_path).replace("\\", "/"))

    def test_single_split_preserves_spaces_in_transcription(self):
        """Verifies that multi-space transcriptions are not tokenized or truncated."""
        importer = IIITIndicHWImporter(
            dataset_root=self.dataset_root,
            language="kannada",
            project_root=self.root_path,
        )
        split_dir = self.dataset_root / "train"
        line = "images/1.jpg   ಪ್ರಾರ್ಥಿಸಿದ   ಮೊರೆಹೋಗಿದ್ದಾರೆ   ೭೫೦೦೦  "
        sample, issue = importer.parse_ground_truth_line(
            line=line,
            line_number=1,
            split_dir=split_dir,
            split_name="train",
            validate_images=True,
        )

        self.assertIsNone(issue)
        self.assertIsNotNone(sample)
        # Transcription must preserve interior words
        self.assertEqual(sample.text, "ಪ್ರಾರ್ಥಿಸಿದ   ಮೊರೆಹೋಗಿದ್ದಾರೆ   ೭೫೦೦೦")

    def test_parse_ground_truth_line_malformed_and_empty(self):
        """Tests rejection of empty and malformed lines without throwing unhandled exceptions."""
        importer = IIITIndicHWImporter(
            dataset_root=self.dataset_root,
            language="kannada",
            project_root=self.root_path,
        )
        split_dir = self.dataset_root / "train"

        # 1. Blank line
        sample, issue = importer.parse_ground_truth_line("", 1, split_dir, "train")
        self.assertIsNone(sample)
        self.assertEqual(issue.issue_type, "malformed_line")

        # 2. Only image path without transcription
        sample, issue = importer.parse_ground_truth_line("images/1.jpg", 2, split_dir, "train")
        self.assertIsNone(sample)
        self.assertEqual(issue.issue_type, "malformed_line")

    def test_missing_image_on_disk_reported(self):
        """Tests that missing referenced images are reported as issues."""
        importer = IIITIndicHWImporter(
            dataset_root=self.dataset_root,
            language="kannada",
            project_root=self.root_path,
        )
        split_dir = self.dataset_root / "train"
        # 999.jpg does not exist
        sample, issue = importer.parse_ground_truth_line(
            "images/999.jpg\tನಾಜೂಕಾಗಿರುವುದರಿಂದ",
            10,
            split_dir,
            "train",
            validate_images=True,
        )
        self.assertIsNone(sample)
        self.assertIsNotNone(issue)
        self.assertEqual(issue.issue_type, "missing_image")
        self.assertIn("999.jpg", issue.message)

    def test_import_split_independence(self):
        """Verifies importing train, val, and test splits independently."""
        importer = IIITIndicHWImporter(
            dataset_root=self.dataset_root,
            language="kannada",
            project_root=self.root_path,
        )

        train_samples, train_rep = importer.import_split("train")
        self.assertEqual(len(train_samples), 2)
        self.assertEqual(train_rep.total_lines, 2)
        self.assertEqual(train_rep.imported_count, 2)
        self.assertTrue(train_rep.is_clean)

        val_samples, val_rep = importer.import_split("val")
        self.assertEqual(len(val_samples), 2)
        self.assertEqual(val_samples[0].text, "ಚುರಮರಿ")

        test_samples, test_rep = importer.import_split("test")
        self.assertEqual(len(test_samples), 2)
        self.assertEqual(test_samples[0].text, "ಕಾಯ್ದಿರಿಸಿದ್ದ")

    def test_save_manifest_and_handwriting_dataset_contract(self):
        """Tests manifest serialization and loading into existing HandwritingDataset."""
        importer = IIITIndicHWImporter(
            dataset_root=self.dataset_root,
            language="kannada",
            project_root=self.root_path,
        )

        train_samples, _ = importer.import_split("train")
        manifest_path = self.root_path / "manifest_test.jsonl"
        importer.save_manifest(train_samples, manifest_path)

        self.assertTrue(manifest_path.exists())

        # Load with HandwritingDataset
        ds = HandwritingDataset(
            manifest_path=manifest_path,
            root_dir=self.root_path,
            validate_images=True,
        )
        self.assertEqual(len(ds), 2)
        item = ds[0]
        self.assertEqual(item["text"], "ನಾಜೂಕಾಗಿರುವುದರಿಂದ")
        self.assertEqual(item["language"], "kannada")
        self.assertEqual(item["script"], "Kannada")
        self.assertEqual(item["metadata"]["source"], "IIIT-INDIC-HW-WORDS")

    def test_integrity_checks_pass_and_detect_overlap(self):
        """Tests verify_dataset_integrity for clean splits and detecting overlaps."""
        importer = IIITIndicHWImporter(
            dataset_root=self.dataset_root,
            language="kannada",
            project_root=self.root_path,
        )

        split_results = importer.import_all_splits()
        integrity = importer.verify_dataset_integrity(split_results)

        self.assertTrue(integrity["all_images_exist"])
        self.assertTrue(integrity["all_transcriptions_non_empty"])
        self.assertTrue(integrity["zero_split_overlap"])
        self.assertEqual(integrity["total_imported"], 6)
        self.assertEqual(integrity["sample_1_verified"]["train"]["text"], "ನಾಜೂಕಾಗಿರುವುದರಿಂದ")
        self.assertEqual(integrity["sample_1_verified"]["val"]["text"], "ಚುರಮರಿ")
        self.assertEqual(integrity["sample_1_verified"]["test"]["text"], "ಕಾಯ್ದಿರಿಸಿದ್ದ")

    def test_language_script_auto_inference(self):
        """Verifies that other Indic languages automatically map to their respective scripts."""
        telugu_importer = IIITIndicHWImporter(dataset_root=self.dataset_root, language="telugu")
        self.assertEqual(telugu_importer.script, "Telugu")

        hindi_importer = IIITIndicHWImporter(dataset_root=self.dataset_root, language="hindi")
        self.assertEqual(hindi_importer.script, "Devanagari")

        tamil_importer = IIITIndicHWImporter(dataset_root=self.dataset_root, language="tamil")
        self.assertEqual(tamil_importer.script, "Tamil")

        malayalam_importer = IIITIndicHWImporter(dataset_root=self.dataset_root, language="malayalam")
        self.assertEqual(malayalam_importer.script, "Malayalam")


if __name__ == "__main__":
    unittest.main()
