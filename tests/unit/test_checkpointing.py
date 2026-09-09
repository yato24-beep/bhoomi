"""Unit Tests for Model Checkpointing and Training Metadata."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.training.checkpointing import (
    CheckpointMetadata,
    load_training_checkpoint,
    save_training_checkpoint,
)


class TestCheckpointing(unittest.TestCase):
    """Test suite for checkpoint saving and loading utilities."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_checkpoint_metadata(self):
        """Tests saving and reloading checkpoint metadata without requiring real model weights."""
        mock_model = MagicMock()
        mock_processor = MagicMock()

        meta = CheckpointMetadata(
            checkpoint_id="checkpoint-step-100",
            model_name="microsoft/trocr-small-handwritten",
            model_version="kannada_v1",
            languages=["kannada"],
            epoch=2,
            step=100,
            training_config={"batch_size": 4, "learning_rate": 5e-5},
            metrics={"val_cer": 0.052, "val_wer": 0.12},
            device="cpu",
        )

        saved_path = save_training_checkpoint(
            output_dir=self.output_dir,
            model=mock_model,
            processor=mock_processor,
            metadata=meta,
            is_best=True,
        )

        self.assertTrue(saved_path.exists())
        self.assertTrue((saved_path / "training_metadata.json").exists())

        # Verify best_checkpoint directory exists
        best_dir = self.output_dir / "best_checkpoint"
        self.assertTrue(best_dir.exists())
        self.assertTrue((best_dir / "training_metadata.json").exists())

        # Verify save_pretrained calls on mocks
        mock_model.save_pretrained.assert_called_once_with(str(saved_path))
        mock_processor.save_pretrained.assert_called_once_with(str(saved_path))

        # Test loading checkpoint
        loaded_meta, loaded_path = load_training_checkpoint(saved_path)
        self.assertEqual(loaded_meta["checkpoint_id"], "checkpoint-step-100")
        self.assertEqual(loaded_meta["model_version"], "kannada_v1")
        self.assertEqual(loaded_meta["languages"], ["kannada"])
        self.assertEqual(loaded_meta["epoch"], 2)
        self.assertEqual(loaded_meta["step"], 100)
        self.assertEqual(loaded_meta["metrics"]["val_cer"], 0.052)
        self.assertEqual(loaded_path, saved_path)

    def test_load_non_existent_checkpoint_raises_error(self):
        """Tests that loading missing directory or metadata raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_training_checkpoint(self.output_dir / "non_existent_ckpt")

        # Directory without metadata
        empty_dir = self.output_dir / "empty_ckpt"
        empty_dir.mkdir()
        with self.assertRaises(FileNotFoundError):
            load_training_checkpoint(empty_dir)


if __name__ == "__main__":
    unittest.main()
