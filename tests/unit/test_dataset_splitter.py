"""Unit Tests for Dataset Splitting and Stratification."""

import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.training.dataset import MultilingualHandwritingSample
from src.training.splitter import (
    DatasetSplitResult,
    split_and_save_manifests,
    split_dataset_samples,
)


class TestDatasetSplitter(unittest.TestCase):
    """Test suite for deterministic, stratified dataset splitting."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

        # Create dummy samples across Kannada, Telugu, and English
        self.samples = []
        for i in range(10):
            img_p = self.base_dir / f"kannada_{i}.png"
            Image.new("RGB", (50, 20), color=255).save(img_p)
            self.samples.append(
                MultilingualHandwritingSample(
                    image_path=img_p,
                    text=f"ಕನ್ನಡ {i}",
                    language="kannada",
                )
            )

        for i in range(10):
            img_p = self.base_dir / f"telugu_{i}.png"
            Image.new("RGB", (50, 20), color=255).save(img_p)
            self.samples.append(
                MultilingualHandwritingSample(
                    image_path=img_p,
                    text=f"తెలుగు {i}",
                    language="telugu",
                )
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_split_dataset_samples_proportions(self):
        """Tests standard 80/10/10 split on 20 samples."""
        result = split_dataset_samples(
            samples=self.samples,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            stratify_by_language=True,
        )

        self.assertEqual(result.total_samples, 20)
        self.assertGreaterEqual(len(result.train_samples), 14)
        self.assertGreaterEqual(len(result.val_samples), 2)
        self.assertGreaterEqual(len(result.test_samples), 2)

    def test_split_reproducibility_with_seed(self):
        """Tests that identical seed produces identical split partitions."""
        split_1 = split_dataset_samples(self.samples, seed=12345)
        split_2 = split_dataset_samples(self.samples, seed=12345)

        texts_train_1 = [s.text for s in split_1.train_samples]
        texts_train_2 = [s.text for s in split_2.train_samples]
        self.assertEqual(texts_train_1, texts_train_2)

    def test_language_stratification_preserves_ratios(self):
        """Tests that stratified splitting distributes both Kannada and Telugu into train and test."""
        result = split_dataset_samples(
            self.samples,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            stratify_by_language=True,
        )

        # Check language distribution in result
        dist = result.language_distribution
        self.assertIn("kannada", dist)
        self.assertIn("telugu", dist)

        self.assertGreater(dist["kannada"]["train"], 0)
        self.assertGreater(dist["telugu"]["train"], 0)
        self.assertGreater(dist["kannada"]["val"], 0)
        self.assertGreater(dist["telugu"]["val"], 0)

    def test_invalid_ratios_raise_error(self):
        """Tests that invalid split ratios raise ValueError."""
        with self.assertRaises(ValueError):
            split_dataset_samples(self.samples, train_ratio=0.7, val_ratio=0.1, test_ratio=0.1)  # sum = 0.9

        with self.assertRaises(ValueError):
            split_dataset_samples(self.samples, train_ratio=-0.1, val_ratio=0.5, test_ratio=0.6)

    def test_split_and_save_manifests(self):
        """Tests reading a manifest, splitting it, and writing output JSONL files."""
        input_manifest = self.base_dir / "all_data.jsonl"
        output_dir = self.base_dir / "splits"

        with open(input_manifest, "w", encoding="utf-8") as f:
            for s in self.samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

        split_result, paths = split_and_save_manifests(
            input_manifest=input_manifest,
            output_dir=output_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            prefix="kannada_exp_",
            root_dir=self.base_dir,
            verify_images=True,
        )

        self.assertIn("train", paths)
        self.assertIn("val", paths)
        self.assertIn("test", paths)

        self.assertTrue(paths["train"].exists())
        self.assertTrue(paths["val"].exists())
        self.assertTrue(paths["test"].exists())


if __name__ == "__main__":
    unittest.main()
