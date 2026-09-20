"""Visual Script & Handwriting Style Classifier for Multi-Modal OCR Routing.

Classifies image regions into:
- 'printed' (machine typeset, uniform strokes, sharp edges, fixed baselines)
- 'handwritten' (irregular stroke widths, cursive/pen strokes, curved baselines)

Computes continuous `routing_confidence` based on:
1. Stroke Width Variation (CV of stroke thickness via distance transform)
2. Contour & Gradient Edge Uniformity
3. Low-cost probing differential (EasyOCR printed score vs handwriting artifacting)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger("style_classifier")

# Soft import cv2
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


@dataclass
class StyleClassificationResult:
    """Classification decision and confidence for a text crop."""
    is_handwritten: bool
    style_label: str  # 'printed' or 'handwritten'
    routing_confidence: float
    stroke_width_cv: float
    contour_irregularity: float
    probe_confidence: Optional[float] = None
    reason: str = "visual_feature_analysis"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_handwritten": self.is_handwritten,
            "style_label": self.style_label,
            "routing_confidence": round(self.routing_confidence, 4),
            "stroke_width_cv": round(self.stroke_width_cv, 4),
            "contour_irregularity": round(self.contour_irregularity, 4),
            "probe_confidence": round(self.probe_confidence, 4) if self.probe_confidence is not None else None,
            "reason": self.reason,
        }


class ScriptStyleClassifier:
    """Classifies text crops into printed vs handwritten styles to guide automatic routing."""

    def __init__(
        self,
        stroke_cv_threshold: float = 0.36,
        high_confidence_printed_gate: float = 0.85,
    ):
        self.stroke_cv_threshold = stroke_cv_threshold
        self.high_confidence_printed_gate = high_confidence_printed_gate

    def classify_crop(
        self,
        image_crop: Image.Image,
        easyocr_probe_confidence: Optional[float] = None,
    ) -> StyleClassificationResult:
        """Evaluates whether an image crop is printed or handwritten.

        Args:
            image_crop: PIL Image crop of a single text line or word.
            easyocr_probe_confidence: Optional confidence score from an initial EasyOCR pass.

        Returns:
            StyleClassificationResult with is_handwritten flag and routing_confidence.
        """
        # Shortcut: if EasyOCR already recognized this crop with very high confidence (>= 0.85),
        # it is decisively printed text (printed Kannada/English model).
        if easyocr_probe_confidence is not None and easyocr_probe_confidence >= self.high_confidence_printed_gate:
            return StyleClassificationResult(
                is_handwritten=False,
                style_label="printed",
                routing_confidence=min(0.99, float(easyocr_probe_confidence)),
                stroke_width_cv=0.15,
                contour_irregularity=0.10,
                probe_confidence=float(easyocr_probe_confidence),
                reason="high_confidence_easyocr_printed_match",
            )

        # Visual Stroke Feature Extraction
        arr = np.array(image_crop.convert("L"))
        h, w = arr.shape

        if h < 8 or w < 8:
            # Too small to reliably extract stroke features; default to printed
            return StyleClassificationResult(
                is_handwritten=False,
                style_label="printed",
                routing_confidence=0.60,
                stroke_width_cv=0.20,
                contour_irregularity=0.15,
                probe_confidence=easyocr_probe_confidence,
                reason="low_resolution_default_printed",
            )

        stroke_cv = 0.25
        contour_irreg = 0.20

        if HAS_CV2 and cv2 is not None:
            try:
                # 1. Binarize with Otsu
                _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # 2. Distance transform to compute stroke width distribution
                dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
                foreground_dist = dist_map[binary > 0]

                if len(foreground_dist) > 20:
                    mean_w = float(np.mean(foreground_dist))
                    std_w = float(np.std(foreground_dist))
                    if mean_w > 0.1:
                        # Coefficient of variation (CV) of stroke width
                        stroke_cv = std_w / mean_w

                # 3. Contour perimeter vs convex hull irregularity
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                irreg_scores = []
                for c in contours:
                    if cv2.contourArea(c) > 25:
                        hull = cv2.convexHull(c)
                        hull_area = cv2.contourArea(hull)
                        c_area = cv2.contourArea(c)
                        if hull_area > 0:
                            # Higher ratio means more concave loops / cursive handwritten variations
                            irreg_scores.append(1.0 - (c_area / hull_area))
                if irreg_scores:
                    contour_irreg = float(np.mean(irreg_scores))

            except Exception as cv_err:
                logger.debug("CV stroke analysis notice: %s", cv_err)

        # Decision synthesis
        # Printed text typically has stroke_cv < 0.35 and contour_irreg < 0.30
        # Handwritten text has stroke_cv >= 0.36 and higher contour variability
        is_hw = (stroke_cv >= self.stroke_cv_threshold) or (contour_irreg >= 0.40)

        # Modulate with probe confidence if available:
        # If EasyOCR returned low confidence (< 0.45) on an authentic text crop,
        # it strongly indicates handwriting that EasyOCR could not decode.
        if easyocr_probe_confidence is not None:
            if easyocr_probe_confidence < 0.45 and (stroke_cv >= 0.30 or contour_irreg >= 0.30):
                is_hw = True

        # Compute calibrated routing confidence
        if is_hw:
            confidence = min(0.98, max(0.65, 0.50 + (stroke_cv * 0.8) + (contour_irreg * 0.4)))
            label = "handwritten"
        else:
            confidence = min(0.98, max(0.65, 0.50 + ((0.40 - stroke_cv) * 0.8)))
            label = "printed"

        return StyleClassificationResult(
            is_handwritten=is_hw,
            style_label=label,
            routing_confidence=confidence,
            stroke_width_cv=stroke_cv,
            contour_irregularity=contour_irreg,
            probe_confidence=easyocr_probe_confidence,
            reason="stroke_variance_and_contour_analysis",
        )


_STYLE_CLASSIFIER_INSTANCE = None


def get_style_classifier() -> ScriptStyleClassifier:
    """Returns or creates the shared ScriptStyleClassifier singleton."""
    global _STYLE_CLASSIFIER_INSTANCE
    if _STYLE_CLASSIFIER_INSTANCE is None:
        _STYLE_CLASSIFIER_INSTANCE = ScriptStyleClassifier()
    return _STYLE_CLASSIFIER_INSTANCE
