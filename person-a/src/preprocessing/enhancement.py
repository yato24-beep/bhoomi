"""person-a/src/preprocessing/enhancement.py
OpenCV adaptive image enhancements: CLAHE, Bilateral Denoising, Unsharp Sharpening, and Adaptive Binarization.
"""

from typing import List, Tuple
import cv2
import numpy as np


class ImageEnhancer:
    """Applies selective, quality-driven OpenCV enhancements."""

    def __init__(
        self,
        clahe_clip: float = 2.0,
        clahe_grid: int = 8,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
    ):
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization in LAB space."""
        if len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))
            return clahe.apply(image)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))
        cl = clahe.apply(l_chan)
        merged = cv2.merge((cl, a_chan, b_chan))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Edge-preserving bilateral filtering to reduce noise while maintaining character sharpness."""
        return cv2.bilateralFilter(
            image,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space,
        )

    def sharpen(self, image: np.ndarray, amount: float = 1.0) -> np.ndarray:
        """Unsharp masking for edge and character stroke crispness."""
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return sharpened

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Adaptive Gaussian thresholding for high-contrast binarized OCR feed."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
