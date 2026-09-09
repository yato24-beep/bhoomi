"""Unit Tests for Multilingual Dataset Loader and Manifest Parser."""

import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.training.dataset import (
    HandwritingDataset,
    MultilingualHandwritingSample,
    create_split_datasets,
    parse_manifest_record,
)


class TestTrainingDataset(unittest.TestCase):
    """Test suite for HandwritingDataset and manifest parser."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

        # Create dummy sample images
        self.img_kannada = self.base_path / "kannada_01.png"
        self.img_telugu = self.base_path / "telugu_01.png"
        self.img_hindi = self.base_path / "hindi_01.png"

        for p in (self.img_kannada, self.img_telugu, self.img_hindi):
            img = Image.new("RGB", (100, 40), color=(255, 255, 255))
            img.save(p)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_manifest_record_valid_kannada(self):
        """Tests parsing a valid Kannada JSONL record with Unicode text."""
        raw_json = json.dumps({
            "image": str(self.img_kannada),
            "text": "ಕರ್ನಾಟಕ ಕಂದಾಯ ಇಲಾಖೆ",
            "language": "kannada",
            "script": "Kannada",
            "metadata": {"doc_id": "doc_101"},
        })

        sample = parse_manifest_record(raw_json)
        self.assertEqual(sample.text, "ಕರ್ನಾಟಕ ಕಂದಾಯ ಇಲಾಖೆ")
        self.assertEqual(sample.language, "kannada")
        self.assertEqual(sample.script, "Kannada")
        self.assertEqual(sample.metadata["doc_id"], "doc_101")
        self.assertEqual(sample.image_path, self.img_kannada)

    def test_parse_manifest_record_alias_fields(self):
        """Tests parsing records using alternative field names 'image_path' and 'ground_truth'."""
        data_dict = {
            "image_path": str(self.img_telugu),
            "ground_truth": "ఆంధ్రప్రదేశ్ రెవెన్యూ రికార్డు",
            "language": "Telugu",
        }

        sample = parse_manifest_record(data_dict)
        self.assertEqual(sample.text, "ఆంధ్రప్రదేశ్ రెవెన్యూ రికార్డు")
        self.assertEqual(sample.language, "telugu")
        self.assertEqual(sample.script, "Telugu")

    def test_parse_manifest_record_missing_required_fields(self):
        """Tests that records with missing image or text raise ValueError."""
        # Missing text
        with self.assertRaises(ValueError):
            parse_manifest_record({"image": "sample.png"})

        # Missing image
        with self.assertRaises(ValueError):
            parse_manifest_record({"text": "sample text"})

        # Empty string
        with self.assertRaises(ValueError):
            parse_manifest_record("   ")

        # Malformed JSON
        with self.assertRaises(ValueError):
            parse_manifest_record("{'bad_json': true")

    def test_multilingual_dataset_loading_and_filtering(self):
        """Tests loading a manifest containing Kannada, Telugu, and Hindi samples."""
        manifest_file = self.base_path / "multilingual_manifest.jsonl"
        with open(manifest_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"image": str(self.img_kannada), "text": "ಸರ್ವೆ ನಂ ೧೨", "language": "kannada"}) + "\n")
            f.write(json.dumps({"image": str(self.img_telugu), "text": "సర్వే నెం 34", "language": "telugu"}) + "\n")
            f.write(json.dumps({"image": str(self.img_hindi), "text": "खसरा संख्या ५६", "language": "hindi"}) + "\n")

        dataset = HandwritingDataset(manifest_path=manifest_file, validate_images=True)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.get_languages(), ["hindi", "kannada", "telugu"])

        # Filter by language
        kannada_subset = dataset.filter_by_language("kannada")
        self.assertEqual(len(kannada_subset), 1)
        self.assertEqual(kannada_subset[0]["text"], "ಸರ್ವೆ ನಂ ೧೨")
        self.assertEqual(kannada_subset[0]["language"], "kannada")

    def test_missing_image_validation(self):
        """Tests image existence checking during dataset instantiation."""
        manifest_file = self.base_path / "missing_img_manifest.jsonl"
        with open(manifest_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"image": "non_existent_crop.png", "text": "test"}) + "\n")

        # Strict validation raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            HandwritingDataset(
                manifest_path=manifest_file,
                root_dir=self.base_path,
                validate_images=True,
                ignore_missing=False,
            )

        # Lenient validation skips missing
        dataset = HandwritingDataset(
            manifest_path=manifest_file,
            root_dir=self.base_path,
            validate_images=True,
            ignore_missing=True,
        )
        self.assertEqual(len(dataset), 0)
        self.assertEqual(len(dataset.missing_samples), 1)

    def test_create_split_datasets(self):
        """Tests factory creation of train and validation splits."""
        train_man = self.base_path / "train.jsonl"
        val_man = self.base_path / "val.jsonl"

        with open(train_man, "w", encoding="utf-8") as f:
            f.write(json.dumps({"image": str(self.img_kannada), "text": "train 1", "language": "kannada"}) + "\n")
        with open(val_man, "w", encoding="utf-8") as f:
            f.write(json.dumps({"image": str(self.img_telugu), "text": "val 1", "language": "telugu"}) + "\n")

        splits = create_split_datasets(
            train_manifest=train_man,
            val_manifest=val_man,
            validate_images=True,
        )

        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertEqual(len(splits["train"]), 1)
        self.assertEqual(len(splits["val"]), 1)


if __name__ == "__main__":
    unittest.main()
