"""Conservative Handwritten Document Augmentation Utilities.

Implements safe, non-destructive transformations for handwritten text images:
- Sub-degree and small-angle rotations (±1° to ±5°) with boundary interpolation
- Subtle contrast and brightness scaling
- Mild sensor / paper noise simulation
- Preserves faint ink strokes, Indic loops (ottakshara), and diacritic marks (anusvara/visarga).
"""

import random
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.training.config import AugmentationConfig


def small_rotation(
    image: Image.Image,
    angle_deg: float,
    fill_color: Union[int, Tuple[int, ...]] = (255, 255, 255),
) -> Image.Image:
    """Applies a small rotation with bicubic interpolation and clean border fill."""
    if abs(angle_deg) < 0.05:
        return image.copy()
    fill = fill_color if image.mode == "RGB" else 255
    return image.rotate(
        angle_deg,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill,
    )


def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Subtly scales contrast while keeping dynamic range balanced."""
    factor = max(0.5, min(2.0, factor))
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    """Subtly scales brightness simulating archival paper exposure variations."""
    factor = max(0.5, min(1.8, factor))
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def add_mild_gaussian_noise(image: Image.Image, std: float = 0.02) -> Image.Image:
    """Simulates mild paper scanner grain noise without corrupting character strokes."""
    if std <= 0.0:
        return image.copy()

    img_arr = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=std, size=img_arr.shape)
    noisy_arr = np.clip(img_arr + noise, 0.0, 1.0)
    out_arr = (noisy_arr * 255.0).astype(np.uint8)
    return Image.fromarray(out_arr, mode=image.mode)


class HandwritingAugmentor:
    """Callable augmentation pipeline applying conservative stochastic transformations."""

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """Initializes augmentor with configuration limits.

        Args:
            config: AugmentationConfig parameters. If None, default conservative settings are used.
        """
        self.config = config or AugmentationConfig()

    def __call__(self, image: Image.Image) -> Image.Image:
        """Applies stochastic augmentation to input PIL Image.

        Args:
            image: Input PIL Image.

        Returns:
            Image.Image: Augmented image matching original mode and dimensions.
        """
        if not self.config.enabled:
            return image.copy()

        out = image.copy()

        # 1. Conservative stochastic rotation (e.g. within [-3.0, +3.0] deg)
        if self.config.rotation_range_deg > 0.0:
            angle = random.uniform(-self.config.rotation_range_deg, self.config.rotation_range_deg)
            out = small_rotation(out, angle)

        # 2. Stochastic contrast adjustment
        if self.config.contrast_range[0] < self.config.contrast_range[1]:
            contrast_factor = random.uniform(self.config.contrast_range[0], self.config.contrast_range[1])
            out = adjust_contrast(out, contrast_factor)

        # 3. Stochastic brightness adjustment
        if self.config.brightness_range[0] < self.config.brightness_range[1]:
            bright_factor = random.uniform(self.config.brightness_range[0], self.config.brightness_range[1])
            out = adjust_brightness(out, bright_factor)

        # 4. Optional mild paper grain noise
        if self.config.noise_std > 0.0 and random.random() < 0.5:
            out = add_mild_gaussian_noise(out, std=self.config.noise_std)

        # 5. Optional slight smoothing
        if self.config.blur_kernel_prob > 0.0 and random.random() < self.config.blur_kernel_prob:
            out = out.filter(ImageFilter.GaussianBlur(radius=0.5))

        return out
