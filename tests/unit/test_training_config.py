"""Unit Tests for Training Configuration and Validation."""

import tempfile
import unittest
from pathlib import Path

from src.training.config import (
    AugmentationConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)


class TestTrainingConfig(unittest.TestCase):
    """Test suite for TrainingConfig dataclasses and validation."""

    def test_default_config_validation(self):
        """Tests that default configuration passes validation checks."""
        config = TrainingConfig()
        config.validate()
        self.assertEqual(config.experiment_name, "kannada_handwriting_baseline")
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.num_epochs, 3)
        self.assertIn("kannada", config.languages)

    def test_invalid_parameters_raise_value_error(self):
        """Tests that invalid parameter combinations trigger ValueError."""
        # Non-positive batch size
        with self.assertRaises(ValueError):
            TrainingConfig(batch_size=0).validate()

        # Non-positive epochs
        with self.assertRaises(ValueError):
            TrainingConfig(num_epochs=-1).validate()

        # Non-positive learning rate
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=0.0).validate()

        # Empty language list
        with self.assertRaises(ValueError):
            TrainingConfig(languages=[]).validate()

        # Unrealistic rotation degree
        with self.assertRaises(ValueError):
            TrainingConfig(
                augmentation=AugmentationConfig(rotation_range_deg=90.0)
            ).validate()

    def test_yaml_serialization_and_deserialization(self):
        """Tests saving and loading TrainingConfig to/from YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "test_config.yaml"

            original_config = TrainingConfig(
                experiment_name="indic_telugu_test",
                languages=["telugu", "kannada"],
                batch_size=8,
                num_epochs=10,
                learning_rate=1e-4,
                model=ModelConfig(model_version="v2", max_sequence_length=128),
                augmentation=AugmentationConfig(rotation_range_deg=4.0),
                dataset=DatasetConfig(train_manifest="datasets/train.jsonl"),
            )

            original_config.save_yaml(yaml_path)
            self.assertTrue(yaml_path.exists())

            loaded_config = TrainingConfig.from_yaml(yaml_path)
            self.assertEqual(loaded_config.experiment_name, "indic_telugu_test")
            self.assertEqual(loaded_config.languages, ["telugu", "kannada"])
            self.assertEqual(loaded_config.batch_size, 8)
            self.assertEqual(loaded_config.num_epochs, 10)
            self.assertEqual(loaded_config.learning_rate, 1e-4)
            self.assertEqual(loaded_config.model.model_version, "v2")
            self.assertEqual(loaded_config.model.max_sequence_length, 128)
            self.assertEqual(loaded_config.augmentation.rotation_range_deg, 4.0)
            self.assertEqual(loaded_config.dataset.train_manifest, "datasets/train.jsonl")


if __name__ == "__main__":
    unittest.main()
