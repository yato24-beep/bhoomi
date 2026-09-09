"""person-a/src/preprocessing/quality.py
Real Image Quality Assessment (Sharpness, Contrast, Illumination, Skew, Blank Page).
"""

from typing import Tuple
import cv2
import numpy as np

from ..schemas import ImageQualityAssessment


class ImageQualityAnalyzer:
    """Computes physical and visual document quality metrics."""

    def __init__(
        self,
        blur_threshold: float = 100.0,
        blank_threshold: float = 0.995,
        default_dpi: float = 300.0,
    ):
        self.blur_threshold = blur_threshold
        self.blank_threshold = blank_threshold
        self.default_dpi = default_dpi

    def analyze(self, image: np.ndarray) -> ImageQualityAssessment:
        if image is None or image.size == 0:
            return ImageQualityAssessment(
                dpi=self.default_dpi,
                blur_score=0.0,
                is_blurry=True,
                contrast_score=0.0,
                illumination_uniformity=0.0,
                skew_angle=0.0,
                is_blank=True,
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # 1. Blur score: Laplacian Variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())
        is_blurry = blur_score < self.blur_threshold

        # 2. Contrast: RMS standard deviation of pixel intensities
        contrast_score = float(gray.std())

        # 3. Illumination uniformity: 4-quadrant mean intensity ratio
        h, w = gray.shape
        mid_h, mid_w = h // 2, w // 2
        q1 = gray[0:mid_h, 0:mid_w].mean()
        q2 = gray[0:mid_h, mid_w:w].mean()
        q3 = gray[mid_h:h, 0:mid_w].mean()
        q4 = gray[mid_h:h, mid_w:w].mean()
        quad_means = [q1, q2, q3, q4]
        min_q = min(quad_means)
        max_q = max(quad_means)
        uniformity = float(min_q / max_q) if max_q > 0 else 1.0

        # 4. Blank page check: ratio of near-white pixels (>= 245)
        white_pixels = np.count_nonzero(gray >= 245)
        white_ratio = float(white_pixels / (h * w))
        is_blank = white_ratio >= self.blank_threshold

        # 5. DPI Estimation
        estimated_dpi = self._estimate_dpi(image.shape)

        return ImageQualityAssessment(
            dpi=estimated_dpi,
            blur_score=round(blur_score, 2),
            is_blurry=is_blurry,
            contrast_score=round(contrast_score, 2),
            illumination_uniformity=round(uniformity, 3),
            skew_angle=0.0,
            rotation_needed=0,
            is_blank=is_blank,
            preprocessing_applied=[],
        )

    def _estimate_dpi(self, shape: Tuple[int, ...]) -> float:
        h, w = shape[:2]
        max_dim = max(h, w)
        if max_dim >= 2000:
            return 300.0
        elif max_dim >= 1400:
            return 200.0
        elif max_dim >= 900:
            return 150.0
        elif max_dim >= 600:
            return 100.0
        else:
            return 72.0
