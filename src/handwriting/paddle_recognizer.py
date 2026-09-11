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

# Set environment flags to avoid OpenMP duplicate runtime collision and OneDNN PIR crash on CPU
import os
import sys
import site
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

# Windows DLL Resolution: ensure torch/lib is in DLL directory search path before torch/albumentations import
if sys.platform == "win32":
    search_paths = []
    try:
        search_paths.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        search_paths.append(site.getusersitepackages())
    except Exception:
        pass
    for sp in search_paths:
        tlib = os.path.join(sp, "torch", "lib")
        if os.path.isdir(tlib):
            try:
                os.add_dll_directory(tlib)
            except Exception:
                pass
            os.environ["PATH"] = tlib + ";" + os.environ.get("PATH", "")

# Preload torch safely to resolve Windows DLL resolution order in albumentations
try:
    import torch
except Exception:
    pass

# Global cache for PaddleOCR instances to avoid expensive re-initializations
_GLOBAL_PADDLE_ENGINES: Dict[Tuple[str, bool], Any] = {}

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
        auto_load: bool = False,
        **kwargs: Any,
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
            auto_load: Whether to eagerly initialize PaddleOCR engine on creation.
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
        self._auto_load = auto_load

        if paddle_engine is not None:
            self._engine = paddle_engine
            self._engine_available = True
        elif self._auto_load:
            self._ensure_engine()

    def _ensure_engine(self) -> bool:
        """Lazily initialize the PaddleOCR engine if not already initialized."""
        if self._engine is not None:
            return True
        cache_key = (self._paddle_lang, self.use_angle_cls)
        if cache_key in _GLOBAL_PADDLE_ENGINES:
            self._engine = _GLOBAL_PADDLE_ENGINES[cache_key]
            self._engine_available = True
            return True
        if self._init_error is not None:
            return False
        if HAS_PADDLEOCR and PaddleOCR is not None:
            try:
                import os
                os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
                os.environ["FLAGS_use_mkldnn"] = "0"
                os.environ["FLAGS_enable_pir_api"] = "0"
                os.environ["FLAGS_enable_pir_in_executor"] = "0"
                try:
                    import paddle
                    paddle.set_flags({"FLAGS_use_mkldnn": False})
                except Exception:
                    pass

                try:
                    self._engine = PaddleOCR(
                        use_angle_cls=self.use_angle_cls,
                        lang=self._paddle_lang,
                        enable_mkldnn=False,
                    )
                except TypeError:
                    try:
                        self._engine = PaddleOCR(
                            use_angle_cls=self.use_angle_cls,
                            lang=self._paddle_lang,
                        )
                    except TypeError:
                        self._engine = PaddleOCR(lang=self._paddle_lang)
                _GLOBAL_PADDLE_ENGINES[cache_key] = self._engine
                self._engine_available = True
                return True
            except Exception as e:
                self._engine_available = False
                self._init_error = f"PaddleOCR initialization failed for lang '{self._paddle_lang}': {str(e)}"
                return False
        else:
            self._engine_available = False
            self._init_error = "PaddleOCR library is not installed in current Python environment."
            return False

    @property
    def is_available(self) -> bool:
        """Returns whether the underlying PaddleOCR engine is successfully loaded."""
        if self._engine is not None:
            return True
        return HAS_PADDLEOCR and (PaddleOCR is not None)

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
        already_preprocessed = bool(
            preprocessing_info
            and (
                "preprocessing_pipeline" in preprocessing_info
                or "pipeline_steps" in preprocessing_info
            )
        )
        if self.preprocess_input and not already_preprocessed:
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

        # Step 2: Ensure engine is initialized and handle unavailable engine gracefully without fabricating output
        self._ensure_engine()
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
            try:
                raw_results = self._engine.ocr(np_image, cls=self.use_angle_cls, **kwargs)
            except TypeError:
                raw_results = self._engine.ocr(np_image, **kwargs)
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

        # Step 4: Parse PaddleOCR response structure (supports both PaddleOCR 2.x and PaddleX 3.x)
        recognized_lines: List[str] = []
        confidences: List[float] = []
        detected_boxes: List[Optional[BoundingBox]] = []

        if raw_results and raw_results[0] is not None:
            first_res = raw_results[0]

            # Case A: PaddleX / PaddleOCR 3.x structure (OCRResult or dict mapping)
            if hasattr(first_res, "get") or (hasattr(first_res, "__getitem__") and not isinstance(first_res, (list, tuple))):
                try:
                    texts = first_res.get("rec_texts", []) if hasattr(first_res, "get") else first_res["rec_texts"]
                    scores = first_res.get("rec_scores", []) if hasattr(first_res, "get") else first_res.get("rec_scores", [])
                    boxes = first_res.get("rec_boxes", []) if hasattr(first_res, "get") else []
                    polys = first_res.get("rec_polys", first_res.get("dt_polys", [])) if hasattr(first_res, "get") else []

                    for i, text_val in enumerate(texts):
                        if text_val and str(text_val).strip():
                            clean_text = str(text_val).strip()
                            recognized_lines.append(clean_text)

                            if i < len(scores) and scores[i] is not None:
                                try:
                                    confidences.append(float(scores[i]))
                                except (ValueError, TypeError):
                                    pass

                            det_bbox = None
                            if i < len(boxes) and boxes[i] is not None and len(boxes[i]) == 4:
                                try:
                                    det_bbox = BoundingBox(
                                        x_min=int(round(float(boxes[i][0]))),
                                        y_min=int(round(float(boxes[i][1]))),
                                        x_max=int(round(float(boxes[i][2]))),
                                        y_max=int(round(float(boxes[i][3]))),
                                    )
                                except Exception:
                                    det_bbox = None
                            elif i < len(polys) and polys[i] is not None:
                                det_bbox = self._extract_bounding_box_from_polygon(polys[i])

                            detected_boxes.append(det_bbox)
                except Exception:
                    pass

            # Case B: PaddleOCR 2.x structure (list of [polygon, (text, confidence)])
            if not recognized_lines and isinstance(first_res, (list, tuple)):
                for line_entry in first_res:
                    if not line_entry or len(line_entry) < 2:
                        continue
                    poly, text_info = line_entry[0], line_entry[1]
                    if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                        text_content, conf_val = text_info[0], text_info[1]
                    else:
                        text_content, conf_val = str(text_info), None

                    if text_content and text_content.strip():
                        recognized_lines.append(text_content.strip())
                        if conf_val is not None and isinstance(conf_val, (int, float)):
                            confidences.append(float(conf_val))
                        det_bbox = self._extract_bounding_box_from_polygon(poly)
                        detected_boxes.append(det_bbox)

        # Step 4b: Bilingual / mixed-script refinement
        # If the target language is Kannada but the document contains English headers/numbers/text,
        # refine lines where Latin/ASCII characters dominate or Kannada confidence is low (< 0.70)
        # using the English PaddleOCR recognizer
        line_engines: List[str] = [self.model_name] * len(recognized_lines)
        line_langs: List[str] = [self.lang] * len(recognized_lines)

        if self.lang in ("kannada", "kn") and HAS_PADDLEOCR:
            en_cache_key = ("en", self.use_angle_cls)
            en_engine = _GLOBAL_PADDLE_ENGINES.get(en_cache_key)
            if en_engine is None and HAS_PADDLEOCR:
                try:
                    en_engine = PaddleOCR(use_angle_cls=self.use_angle_cls, lang="en", enable_mkldnn=False)
                    _GLOBAL_PADDLE_ENGINES[en_cache_key] = en_engine
                except Exception:
                    en_engine = None

            if en_engine is not None and pil_image is not None:
                img_w, img_h = pil_image.size
                for l_idx, (l_txt, l_cnf, l_box) in enumerate(zip(recognized_lines, confidences, detected_boxes)):
                    if l_box is None:
                        continue
                    has_kn = any('\u0c80' <= c <= '\u0cff' for c in l_txt)
                    ascii_ratio = sum(1 for c in l_txt if ord(c) < 128) / max(len(l_txt), 1)
                    # Candidate for English refinement:
                    # No Kannada unicode characters, or mostly ASCII, or low confidence
                    if not has_kn or ascii_ratio > 0.6 or (l_cnf is not None and l_cnf < 0.65):
                        crop_x1 = max(0, min(l_box.x_min, img_w - 1))
                        crop_y1 = max(0, min(l_box.y_min, img_h - 1))
                        crop_x2 = max(crop_x1 + 1, min(l_box.x_max, img_w))
                        crop_y2 = max(crop_y1 + 1, min(l_box.y_max, img_h))
                        if crop_x2 - crop_x1 < 4 or crop_y2 - crop_y1 < 4:
                            continue
                        line_crop = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                        crop_np = np.array(line_crop.convert("RGB"))
                        try:
                            # Pass 1: Test with English engine (essential for numbers, forms, dates, survey nos)
                            en_res = en_engine.ocr(crop_np, det=False, cls=self.use_angle_cls)
                            if en_res and en_res[0] and en_res[0][0]:
                                en_txt, en_cnf = en_res[0][0][0], float(en_res[0][0][1])
                                en_clean = en_txt.strip()
                                # Prefer English if English has higher confidence or is heavily numeric/legal header
                                is_numeric_or_code = any(c.isdigit() for c in en_clean) and sum(1 for c in en_clean if c.isalnum() or c in "/-.,") / max(len(en_clean), 1) > 0.7
                                is_header = any(w in en_clean.upper() for w in ["MUTATION", "REGISTER", "RULE", "FORM", "ORDER", "SURVEY", "TALUK", "VILLAGE"])
                                if en_clean and en_cnf > 0.40:
                                    should_use_en = False
                                    if not has_kn and en_cnf > (l_cnf or 0.0):
                                        should_use_en = True
                                    elif is_header or (is_numeric_or_code and en_cnf > 0.55):
                                        should_use_en = True
                                    elif en_cnf > (l_cnf or 0.0) + 0.10:
                                        should_use_en = True

                                    if should_use_en:
                                        recognized_lines[l_idx] = en_clean
                                        confidences[l_idx] = en_cnf
                                        line_engines[l_idx] = "paddleocr-english"
                                        line_langs[l_idx] = "english"
                                        continue

                            # Pass 2: If Kannada line had low confidence (< 0.65), run Multi-Pass on upscaled/enhanced crop
                            if has_kn and (l_cnf is None or l_cnf < 0.65):
                                cw, ch = line_crop.size
                                # Try 1.5x upscaled crop with subtle sharpening
                                up_crop = line_crop.resize((int(cw * 1.5), int(ch * 1.5)), Image.Resampling.LANCZOS)
                                up_np = np.array(up_crop.convert("RGB"))
                                kn_retry = self._engine.ocr(up_np, det=False, cls=self.use_angle_cls)
                                if kn_retry and kn_retry[0] and kn_retry[0][0]:
                                    r_txt, r_cnf = kn_retry[0][0][0], float(kn_retry[0][0][1])
                                    r_clean = r_txt.strip()
                                    if r_clean and r_cnf > (l_cnf or 0.0):
                                        recognized_lines[l_idx] = r_clean
                                        confidences[l_idx] = r_cnf
                                        line_engines[l_idx] = "paddleocr-kannada-multipass"
                        except Exception:
                            pass

        combined_text = " ".join(recognized_lines) if recognized_lines else ""

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

        line_details: List[Dict[str, Any]] = []
        for line_txt, conf_val, det_box, l_eng, l_lng in zip(
            recognized_lines, confidences, detected_boxes, line_engines, line_langs
        ):
            line_details.append({
                "text": line_txt,
                "confidence": conf_val if conf_val is not None else None,
                "bbox": {
                    "x_min": det_box.x_min,
                    "y_min": det_box.y_min,
                    "x_max": det_box.x_max,
                    "y_max": det_box.y_max,
                } if det_box is not None else None,
                "engine": l_eng,
                "language": l_lng,
            })

        audit_trail = build_confidence_audit_trail(
            raw_token_probabilities=confidences if confidences else None,
            calculation_method="paddle_line_mean" if confidences else "unavailable",
            custom_metadata={
                "engine_status": "active",
                "detected_lines_count": len(recognized_lines),
                "target_language": self.lang,
                "capability": "printed_baseline_regional_ocr",
                "preprocessing": prep_metadata,
                "line_details": line_details,
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
