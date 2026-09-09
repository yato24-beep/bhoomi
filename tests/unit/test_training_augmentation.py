"""Unit Tests for Conservative Handwritten Document Augmentations."""

import unittest
import numpy as np
from PIL import Image

from src.training.augmentation import (
    HandwritingAugmentor,
    add_mild_gaussian_noise,
    adjust_brightness,
    adjust_contrast,
    small_rotation,
)
from src.training.config import AugmentationConfig


class TestTrainingAugmentation(unittest.TestCase):
    """Test suite for image augmentations."""

    def setUp(self):
        # Create RGB and Grayscale test images
        self.rgb_img = Image.new("RGB", (200, 80), color=(240, 240, 235))
        self.gray_img = Image.new("L", (200, 80), color=230)

    def test_small_rotation_preserves_dimensions_and_mode(self):
        """Tests that small rotation does not alter image dimensions or color mode."""
        rotated_rgb = small_rotation(self.rgb_img, angle_deg=2.5)
        self.assertEqual(rotated_rgb.size, self.rgb_img.size)
        self.assertEqual(rotated_rgb.mode, "RGB")

        rotated_gray = small_rotation(self.gray_img, angle_deg=-3.0)
        self.assertEqual(rotated_gray.size, self.gray_img.size)
        self.assertEqual(rotated_gray.mode, "L")

    def test_contrast_and_brightness_adjustments(self):
        """Tests contrast and brightness adjustment functions."""
        contrast_img = adjust_contrast(self.rgb_img, factor=1.15)
        self.assertEqual(contrast_img.size, self.rgb_img.size)

        bright_img = adjust_brightness(self.rgb_img, factor=0.95)
        self.assertEqual(bright_img.size, self.rgb_img.size)

    def test_mild_gaussian_noise(self):
        """Tests that mild gaussian noise produces bounded valid pixels."""
        noisy_img = add_mild_gaussian_noise(self.rgb_img, std=0.03)
        self.assertEqual(noisy_img.size, self.rgb_img.size)
        arr = np.array(noisy_img)
        self.assertTrue((arr >= 0).all() and (arr <= 255).all())

    def test_handwriting_augmentor_pipeline(self):
        """Tests end-to-end HandwritingAugmentor callable."""
        cfg = AugmentationConfig(
            enabled=True,
            rotation_range_deg=3.0,
            contrast_range=(0.9, 1.1),
            brightness_range=(0.9, 1.1),
            noise_std=0.02,
        )
        augmentor = HandwritingAugmentor(cfg)
        aug_result = augmentor(self.rgb_img)

        self.assertEqual(aug_result.size, self.rgb_img.size)
        self.assertEqual(aug_result.mode, "RGB")

    def test_disabled_augmentor_returns_unmodified_copy(self):
        """Tests that disabled augmentor returns identical image."""
        cfg = AugmentationConfig(enabled=False)
        augmentor = HandwritingAugmentor(cfg)
        result = augmentor(self.rgb_img)
        self.assertEqual(list(result.getdata()), list(self.rgb_img.getdata()))


if __name__ == "__main__":
    unittest.main()
