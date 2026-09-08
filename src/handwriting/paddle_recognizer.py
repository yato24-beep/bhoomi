"""PaddleOCR-based recognition backend for Kannada and Indic regional scripts.

Implements BaseHandwritingRecognizer as a real baseline OCR engine for printed
and scanned script lines, with transparent confidence reporting and fallbacks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from schemas import BoundingBox, OCRResult
from src.handwriting.confidence import build_confidence_audit_trail
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput
from src.preprocessing.image_enhancement import (
    load_image_as_pil,
    preprocess_document_image,
)

# Set environment flag to avoid OpenMP duplicate runtime collision on Windows
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Preload torch if present to resolve Windows DLL resolution order in albumentations
try:
    import torch
except Exception:
    pass

# Soft import of PaddleOCR to avoid crashing in environments without Paddle
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except (ImportError, OSError):
    PaddleOCR = None
    HAS_PADDLEOCR = False


# Language code mapping to PaddleOCR model identifiers
PADDLE_LANG_MAPPING: Dict[str, str] = {
    "kannada": "ka",
    "kn": "ka",
    "ka": "ka",
    "telugu": "te",
    "te": "te",
    "tamil": "ta",
    "ta": "ta",
    "hindi": "devanagari",
    "devanagari": "devanagari",
    "hi": "devanagari",
    "english": "en",
    "en": "en",
}


class PaddleKannadaRecognizer(BaseHandwritingRecognizer):
    """Regional script recognizer powered by PaddleOCR (Kannada baseline).

    Note on Capability & Ownership Boundary:
    PaddleOCR Indic models provide strong baseline recognition for printed and clear
    script texts. For unstructured, cursive, or degraded archival handwritten land records,
    this serves as the initial baseline and fallback while TrOCR fine-tuning is conducted.
    """

    def __init__(
        self,
        model_name: str = "paddleocr-kannada",
        model_version: str = "ppocr_v4_kannada",
        paddle_engine: Optional[Any] = None,
        lang: str = "kannada",
        use_angle_cls: bool = True,
        preprocess_input: bool = True,
        enable_thresholding: bool = False,
    ):
        """Initializes the PaddleOCR Kannada recognizer.

        Args:
            model_name: Identifier for this recognizer engine.
            model_version: Underlying model version / checkpoint tag.
            paddle_engine: Pre-initialized PaddleOCR instance or mock for dependency injection.
            lang: Target language script (default: 'kannada').
            use_angle_cls: Whether to enable text orientation classifier in Paddle.
            preprocess_input: Whether to run document image enhancement before inference.
            enable_thresholding: If True, applies adaptive thresholding in preprocessing.
                (Default: False to preserve faint handwritten ink).
        """
        super().__init__(model_name=model_name, model_version=model_version)
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.preprocess_input = preprocess_input
        self.enable_thresholding = enable_thresholding

        # Map language string to Paddle model code (e.g. 'kannada' -> 'ka')
        norm_key = str(lang).strip().lower()
        self._paddle_lang = PADDLE_LANG_MAPPING.get(norm_key, norm_key)

        self._engine: Optional[Any] = None
        self._engine_available: bool = False
        self._init_error: Optional[str] = None

        if paddle_engine is not None:
            self._engine = paddle_engine
            self._engine_available = True
        elif HAS_PADDLEOCR and PaddleOCR is not None:
            try:
                import os
                os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                self._engine = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang=self._paddle_lang,
                    show_log=False,
                )
                self._engine_available = True
            except Exception as e:
                self._engine_available = False
                self._init_error = f"PaddleOCR initialization failed for lang '{self._paddle_lang}': {str(e)}"
        else:
            self._engine_available = False
            self._init_error = "PaddleOCR library is not installed in current Python environment."

    @property
    def is_available(self) -> bool:
        """Returns whether the underlying PaddleOCR engine is successfully loaded."""
        return self._engine_available

    def _extract_bounding_box_from_polygon(self, polygon: List[List[float]]) -> Optional[BoundingBox]:
        """Converts 4-point PaddleOCR polygon [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] to BoundingBox."""
        try:
            xs = [pt[0] for pt in polygon]
            ys = [pt[1] for pt in polygon]
            return BoundingBox(
                x_min=int(round(min(xs))),
                y_min=int(round(min(ys))),
                x_max=int(round(max(xs))),
                y_max=int(round(max(ys))),
            )
        except Exception:
            return None

    def recognize_handwriting(
        self,
        image: ImageInput,
        bbox: Optional[BoundingBox] = None,
        page_number: Optional[int] = None,
        preprocessing_info: Optional[Dict[str, Any]] = None,
        is_handwritten: Optional[bool] = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Runs OCR on an image crop, returning recognized text and genuine confidence.

        Args:
            image: Image crop input (file path, PIL Image, or NumPy array).
            bbox: Optional caller-provided bounding box.
            page_number: Optional 1-indexed document page number.
            preprocessing_info: Optional upstream preprocessing metadata.
            is_handwritten: Optional flag (defaults to None or True).
            **kwargs: Extra parameters passed to PaddleOCR.

        Returns:
            OCRResult with recognized text, true model confidence (or None),
            and comprehensive audit metadata.
        """
        # Step 1: Preprocess image if configured
        prep_metadata: Dict[str, Any] = {}
        if preprocessing_info:
            prep_metadata.update(preprocessing_info)

        pil_image: Image.Image
        if self.preprocess_input:
            prep_res = preprocess_document_image(
                image_input=image,
                apply_thresholding=self.enable_thresholding,
            )
            pil_image = prep_res.image
            prep_metadata["preprocessing_pipeline"] = prep_res.audit_metadata
        else:
            pil_image = load_image_as_pil(image)

        # Convert to RGB NumPy array for PaddleOCR
        np_image = np.array(pil_image.convert("RGB"))

        # Step 2: Handle uninstalled or unavailable engine gracefully without fabricating output
        if not self._engine_available or self._engine is None:
            audit = build_confidence_audit_trail(
                raw_token_probabilities=None,
                calculation_method="unavailable",
                custom_metadata={
                    "engine_status": "unavailable",
                    "reason": self._init_error or "Engine not initialized",
                    "capability": "printed_baseline_with_handwriting_fallback",
                    "target_language": self.lang,
                    "preprocessing": prep_metadata,
                },
            )
            return OCRResult(
                text="",
                confidence=None,
                bbox=bbox,
                is_handwritten=is_handwritten,
                page_number=page_number,
                model_name=self.model_name,
                model_version=self.model_version,
                metadata=audit,
            )

        # Step 3: Run inference with real PaddleOCR engine
        try:
            # PaddleOCR returns: [ [ [polygon, (text, confidence)], ... ] ]
            raw_results = self._engine.ocr(np_image, cls=self.use_angle_cls, **kwargs)
        except Exception as err:
            audit = build_confidence_audit_trail(
                raw_token_probabilities=None,
                calculation_method="inference_error",
                custom_metadata={
                    "error": str(err),
                    "engine_status": "error",
                    "target_language": self.lang,
                    "preprocessing": prep_metadata,
                },
            )
            return OCRResult(
                text="",
                confidence=None,
                bbox=bbox,
                is_handwritten=is_handwritten,
                page_number=page_number,
                model_name=self.model_name,
                model_version=self.model_version,
                metadata=audit,
            )

        # Step 4: Parse PaddleOCR response structure
        recognized_lines: List[str] = []
        confidences: List[float] = []
        detected_boxes: List[Optional[BoundingBox]] = []

        if raw_results and raw_results[0]:
            for line_entry in raw_results[0]:
                if not line_entry or len(line_entry) < 2:
                    continue
                poly, (text_content, conf_val) = line_entry[0], line_entry[1]

                if text_content and text_content.strip():
                    recognized_lines.append(text_content.strip())
                    if conf_val is not None and isinstance(conf_val, (int, float)):
                        confidences.append(float(conf_val))
                    det_bbox = self._extract_bounding_box_from_polygon(poly)
                    detected_boxes.append(det_bbox)

        combined_text = " ".join(recognized_lines)

        # Compute genuine aggregated confidence if confidences exist; do NOT fabricate
        final_confidence: Optional[float] = None
        if confidences:
            final_confidence = round(float(np.mean(confidences)), 4)

        # Assign bounding box (prioritize caller bbox, fallback to detected union)
        effective_bbox = bbox
        if effective_bbox is None and detected_boxes and detected_boxes[0] is not None:
            valid_boxes = [b for b in detected_boxes if b is not None]
            if valid_boxes:
                effective_bbox = BoundingBox(
                    x_min=min(b.x_min for b in valid_boxes),
                    y_min=min(b.y_min for b in valid_boxes),
                    x_max=max(b.x_max for b in valid_boxes),
                    y_max=max(b.y_max for b in valid_boxes),
                )

        audit_trail = build_confidence_audit_trail(
            raw_token_probabilities=confidences if confidences else None,
            calculation_method="paddle_line_mean" if confidences else "unavailable",
            custom_metadata={
                "engine_status": "active",
                "detected_lines_count": len(recognized_lines),
                "target_language": self.lang,
                "capability": "printed_baseline_regional_ocr",
                "preprocessing": prep_metadata,
            },
        )

        return OCRResult(
            text=combined_text,
            confidence=final_confidence,
            bbox=effective_bbox,
            is_handwritten=is_handwritten,
            page_number=page_number,
            model_name=self.model_name,
            model_version=self.model_version,
            metadata=audit_trail,
        )
