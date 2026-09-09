"""person-a/src/preprocessing/deskew.py
Document orientation and skew angle detection via Radon transform / Hough lines.
"""

from typing import Tuple
import cv2
import numpy as np


class DocumentDeskewer:
    """Detects and corrects rotational skew and orientation in document images."""

    def __init__(self, max_angle: float = 45.0):
        self.max_angle = max_angle

    def estimate_skew(self, image: np.ndarray) -> float:
        """Estimate skew angle in degrees using Hough Line Transform on morphological edges."""
        if image is None or image.size == 0:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        if lines is None or len(lines) == 0:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -self.max_angle <= angle <= self.max_angle:
                angles.append(angle)

        if not angles:
            return 0.0

        median_angle = float(np.median(angles))
        return round(median_angle, 2)

    def deskew(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by the detected angle to make text horizontal."""
        if abs(angle) < 0.2:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(
            image, rot_mat, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return deskewed

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect and correct skew in one step."""
        angle = self.estimate_skew(image)
        corrected = self.deskew(image, angle)
        return corrected, angle
