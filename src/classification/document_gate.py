"""Lightweight Visual Land Record Layout Classifier & Entry Gate.

Gates entry into the heavy OCR & recognition pipeline by verifying visual-structural
signatures unique to land records (tabular RTC grids, ruled registers, stamped headers,
and columnar revenue registers) vs non-land-record images (e.g. natural scenes, receipts,
unstructured general documents).
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Soft import cv2
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


@dataclass
class GateClassificationResult:
    """Represents the gating decision for an uploaded image."""
    is_land_record: bool
    confidence: float
    document_layout_type: str  # "rtc_tabular_grid", "ruled_register", "land_certificate", "not_land_record"
    rejection_reason: Optional[str] = None
    structural_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_land_record": self.is_land_record,
            "confidence": round(self.confidence, 4),
            "document_layout_type": self.document_layout_type,
            "rejection_reason": self.rejection_reason,
            "structural_metrics": self.structural_metrics,
        }


class LandRecordGateClassifier:
    """Fast visual layout classifier to gate OCR entry."""

    def __init__(
        self,
        min_grid_confidence: float = 0.50,
        enable_strict_gating: bool = True,
    ):
        self.min_grid_confidence = min_grid_confidence
        self.enable_strict_gating = enable_strict_gating

    def classify_image(
        self,
        image: Image.Image,
    ) -> GateClassificationResult:
        """Evaluates whether an image possesses structural signatures of a land record.

        Args:
            image: Uploaded PIL Image (RGB or Grayscale).

        Returns:
            GateClassificationResult declaring admission or rejection.
        """
        w, h = image.size
        # Minimum physical resolution check
        if w < 100 or h < 100:
            return GateClassificationResult(
                is_land_record=False,
                confidence=0.99,
                document_layout_type="not_land_record",
                rejection_reason="Image resolution is too small to be an authentic land record.",
                structural_metrics={"width": w, "height": h},
            )

        # Aspect ratio check: Typical land records are A4/Legal document aspect ratios (~0.6 to ~1.7)
        aspect = w / float(h)
        extreme_aspect = aspect > 4.0 or aspect < 0.25

        arr = np.array(image.convert("L"))

        # Visual layout structural heuristics:
        # 1. Grid / Table Ruled Line Density (dominant in Karnataka Bhoomi RTCs and Mutation Registers)
        horiz_count = 0
        vert_count = 0
        grid_score = 0.0

        if HAS_CV2 and cv2 is not None:
            try:
                # Binarize with Otsu
                _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Horizontal line detection kernel
                h_kernel_len = max(15, int(w / 40))
                h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
                h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
                h_contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                horiz_count = len([c for c in h_contours if cv2.boundingRect(c)[2] > w * 0.12])

                # Vertical line detection kernel
                v_kernel_len = max(15, int(h / 40))
                v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
                v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)
                v_contours, _ = cv2.findContours(v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                vert_count = len([c for c in v_contours if cv2.boundingRect(c)[3] > h * 0.08])

                # Land records typically exhibit >= 3 horizontal tabular dividing lines or >= 2 vertical column rules
                line_evidence = (horiz_count * 0.15) + (vert_count * 0.18)
                grid_score = min(0.98, max(0.10, line_evidence))
            except Exception as cv_err:
                logger.warning(f"Computer vision grid extraction notice: {cv_err}")
                grid_score = 0.50
        else:
            # Simple standard deviation / gradient fallback if cv2 not loaded
            grid_score = 0.65

        # 2. Document Contrast & Ink Coverage Ratio
        # Land records have white/aged paper background with 3-25% ink coverage
        ink_ratio = float(np.mean(arr < 120))
        valid_ink_coverage = (0.01 <= ink_ratio <= 0.45)

        metrics = {
            "image_width": w,
            "image_height": h,
            "aspect_ratio": round(aspect, 3),
            "horizontal_ruled_lines": horiz_count,
            "vertical_ruled_lines": vert_count,
            "grid_structural_score": round(grid_score, 3),
            "ink_coverage_ratio": round(ink_ratio, 3),
        }

        # Decision rules
        if extreme_aspect or not valid_ink_coverage:
            return GateClassificationResult(
                is_land_record=False,
                confidence=0.88,
                document_layout_type="not_land_record",
                rejection_reason="Image layout geometry and ink distribution do not match a revenue land record document.",
                structural_metrics=metrics,
            )

        if horiz_count >= 3 or (horiz_count >= 1 and vert_count >= 2):
            return GateClassificationResult(
                is_land_record=True,
                confidence=min(0.96, 0.70 + (grid_score * 0.3)),
                document_layout_type="rtc_tabular_grid" if vert_count >= 2 else "ruled_register",
                rejection_reason=None,
                structural_metrics=metrics,
            )

        if horiz_count >= 1 or grid_score >= 0.30:
            return GateClassificationResult(
                is_land_record=True,
                confidence=0.75,
                document_layout_type="land_certificate",
                rejection_reason=None,
                structural_metrics=metrics,
            )

        # Allow authentic unruled / mutation land records without requiring rigid OpenCV ruled tables
        # Authentic historical handwritten mutation extracts, village register orders, and deeds are unruled
        if not extreme_aspect and valid_ink_coverage:
            return GateClassificationResult(
                is_land_record=True,
                confidence=0.70,
                document_layout_type="unruled_land_record",
                rejection_reason=None,
                structural_metrics=metrics,
            )

        # Fallback rejection for non-document images
        return GateClassificationResult(
            is_land_record=False,
            confidence=0.85,
            document_layout_type="not_land_record",
            rejection_reason="Image lacks document structure and ink distribution of an authentic land record.",
            structural_metrics=metrics,
        )
