"""Unit Tests for Modular Handwriting Trainer."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image

from src.training.config import TrainingConfig
from src.training.dataset import HandwritingDataset, MultilingualHandwritingSample
from src.training.trainer import HandwritingTrainer, default_collate_fn


class TestHandwritingTrainer(unittest.TestCase):
    """Test suite for HandwritingTrainer using mock dependencies."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

        # Create sample image
        self.sample_img_path = self.output_dir / "sample.png"
        img = Image.new("RGB", (64, 32), color=(255, 255, 255))
        img.save(self.sample_img_path)

        # Create mock samples
        self.samples = [
            MultilingualHandwritingSample(
                image_path=self.sample_img_path,
                text="ಕರ್ನಾಟಕ",
                language="kannada",
            ),
            MultilingualHandwritingSample(
                image_path=self.sample_img_path,
                text="ಸರ್ವೆ ನಂ ೧೨",
                language="kannada",
            ),
        ]

        self.dataset = HandwritingDataset(
            samples=self.samples,
            validate_images=True,
        )

        self.config = TrainingConfig(
            experiment_name="unit_test_experiment",
            languages=["kannada"],
            output_dir=str(self.output_dir),
            batch_size=2,
            eval_batch_size=2,
            num_epochs=1,
            device="cpu",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_collate_fn_fallback(self):
        """Tests collate function with standard dataset items."""
        items = [self.dataset[0], self.dataset[1]]
        batch = default_collate_fn(items)

        self.assertIn("texts", batch)
        self.assertIn("languages", batch)
        self.assertEqual(len(batch["texts"]), 2)
        self.assertEqual(batch["texts"][0], "ಕರ್ನಾಟಕ")

    def test_trainer_mock_training_step_and_epoch(self):
        """Tests train_step with mock model without downloading weights."""
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.loss = 0.4521
        mock_model.return_value = mock_outputs

        mock_optimizer = MagicMock()

        trainer = HandwritingTrainer(
            config=self.config,
            model=mock_model,
            train_dataset=self.dataset,
            optimizer=mock_optimizer,
        )

        batch = default_collate_fn([self.dataset[0], self.dataset[1]])
        loss = trainer.train_step(batch)

        self.assertAlmostEqual(loss, 0.4521, places=4)
        self.assertEqual(trainer.global_step, 1)

    def test_trainer_mock_evaluate(self):
        """Tests evaluate method with mock model and prediction generator."""
        mock_model = MagicMock()
        mock_model.generate_mock = MagicMock(return_value=["ಕರ್ನಾಟಕ", "ಸರ್ವೆ ನಂ ೧೨"])

        trainer = HandwritingTrainer(
            config=self.config,
            model=mock_model,
            eval_dataset=self.dataset,
        )

        report = trainer.evaluate()
        self.assertEqual(report.total_samples, 2)
        self.assertAlmostEqual(report.overall_cer, 0.0)
        self.assertIn("kannada", report.per_language)

    def test_trainer_train_mock_loop(self):
        """Tests end-to-end training loop execution using mock model and dataset."""
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.loss = 0.25
        mock_model.return_value = mock_outputs
        mock_model.generate_mock = MagicMock(return_value=["ಕರ್ನಾಟಕ", "ಸರ್ವೆ ನಂ ೧೨"])

        trainer = HandwritingTrainer(
            config=self.config,
            model=mock_model,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,
        )

        summary = trainer.train(num_epochs=1)
        self.assertEqual(summary["total_epochs"], 1)
        self.assertGreater(summary["total_steps"], 0)
        self.assertEqual(len(summary["history"]), 1)


if __name__ == "__main__":
    unittest.main()
