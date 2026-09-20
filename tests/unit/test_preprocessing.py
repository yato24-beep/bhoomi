"""Unit tests for document image preprocessing and enhancement."""

import unittest
import numpy as np
from PIL import Image, ImageDraw

from src.preprocessing.image_enhancement import (
    adaptive_soft_threshold,
    deskew_image,
    enhance_contrast,
    estimate_skew_angle,
    light_denoise,
    load_image_as_pil,
    preprocess_document_image,
    to_grayscale,
)


class TestImagePreprocessing(unittest.TestCase):
    """Test suite for image preprocessing functions."""

    def setUp(self):
        """Creates sample test images for processing tests."""
        # Clean RGB test image with dark text box
        self.rgb_img = Image.new("RGB", (200, 100), color=(250, 248, 240))
        draw = ImageDraw.Draw(self.rgb_img)
        draw.rectangle([20, 30, 180, 70], fill=(40, 40, 40))

        # Skewed test image
        self.skewed_img = Image.new("L", (300, 300), color=255)
        draw_skew = ImageDraw.Draw(self.skewed_img)
        draw_skew.rectangle([50, 120, 250, 160], fill=0)
        # Rotate by 5 degrees
        self.skewed_img = self.skewed_img.rotate(5.0, fillcolor=255)

    def test_load_image_as_pil(self):
        """Tests image loading from PIL, NumPy array, and invalid inputs."""
        # From PIL
        pil_out = load_image_as_pil(self.rgb_img)
        self.assertEqual(pil_out.size, (200, 100))

        # From NumPy array
        np_arr = np.zeros((80, 120, 3), dtype=np.uint8)
        np_out = load_image_as_pil(np_arr)
        self.assertEqual(np_out.size, (120, 80))

        # Invalid type
        with self.assertRaises(TypeError):
            load_image_as_pil(12345)  # type: ignore

    def test_to_grayscale(self):
        """Tests conversion to grayscale mode 'L'."""
        gray = to_grayscale(self.rgb_img)
        self.assertEqual(gray.mode, "L")
        self.assertEqual(gray.size, (200, 100))

        # Grayscale idempotency
        already_gray = to_grayscale(gray)
        self.assertEqual(already_gray.mode, "L")

    def test_enhance_contrast(self):
        """Tests contrast enhancement and audit metadata."""
        enhanced, meta = enhance_contrast(self.rgb_img, contrast_factor=1.4)
        self.assertEqual(enhanced.mode, "L")
        self.assertEqual(meta["contrast_factor"], 1.4)
        self.assertIn("auto_contrast_cutoff", meta)

    def test_light_denoise(self):
        """Tests gentle noise reduction with median and gaussian filters."""
        denoised_med, meta_med = light_denoise(self.rgb_img, filter_type="median", kernel_size=3)
        self.assertEqual(denoised_med.mode, "L")
        self.assertEqual(meta_med["denoise_filter"], "median")

        denoised_gauss, meta_gauss = light_denoise(self.rgb_img, filter_type="gaussian", kernel_size=3)
        self.assertEqual(meta_gauss["denoise_filter"], "gaussian")

    def test_estimate_skew_and_deskew(self):
        """Tests estimation of skew angle and correction."""
        angle = estimate_skew_angle(self.skewed_img, max_angle=10.0)
        self.assertIsInstance(angle, float)

        deskewed, meta = deskew_image(self.skewed_img, max_angle=10.0)
        self.assertIn("detected_skew_angle", meta)
        self.assertIn("deskew_applied", meta)
        self.assertEqual(deskewed.size, (300, 300))

    def test_adaptive_soft_threshold(self):
        """Tests adaptive thresholding binarization."""
        bin_img, meta = adaptive_soft_threshold(self.rgb_img)
        self.assertEqual(bin_img.mode, "L")
        self.assertIn("threshold_method", meta)

    def test_full_preprocessing_pipeline(self):
        """Tests full end-to-end preprocessing pipeline and audit trail."""
        res = preprocess_document_image(
            image_input=self.rgb_img,
            apply_deskew=True,
            apply_denoise=True,
            apply_contrast=True,
            apply_thresholding=False,  # default for handwriting preservation
        )

        self.assertEqual(res.image.mode, "L")
        self.assertEqual(res.original_size, (200, 100))
        self.assertIn("grayscale_conversion", res.audit_metadata["pipeline_steps"])
        # Contrast step is emitted as CLAHE variant; also accept clean-document skip
        contrast_steps = {"contrast_enhancement_clahe", "contrast_skipped_clean_document"}
        self.assertTrue(
            any(s in res.audit_metadata["pipeline_steps"] for s in contrast_steps),
            f"Expected a contrast step in pipeline_steps, got: {res.audit_metadata['pipeline_steps']}"
        )
        self.assertIn("light_denoise", res.audit_metadata["pipeline_steps"])
        self.assertEqual(res.audit_metadata["thresholding"]["reason"], "faint_handwriting_preservation")


if __name__ == "__main__":
    unittest.main()
