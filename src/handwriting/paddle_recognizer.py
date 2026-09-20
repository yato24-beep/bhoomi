"""PaddleOCR-based recognition backend for Kannada and Indic regional scripts.

Implements BaseHandwritingRecognizer as a real baseline OCR engine for printed
and scanned script lines, with transparent confidence reporting and fallbacks.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger("paddle_recognizer")

from schemas import BoundingBox, OCRResult
from src.detection.text_detector import DetectedLineGroup, DetectedWordBox
from src.handwriting.confidence import build_confidence_audit_trail
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput
from src.preprocessing.image_enhancement import (
    load_image_as_pil,
    preprocess_document_image,
)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

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
        use_angle_cls: bool = False,
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
            use_angle_cls: Whether to enable text orientation classifier in Paddle (default: False for Indic/Kannada).
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
                except Exception as p_err:
                    logger.debug("Paddle set_flags notice: %s", p_err)

                try:
                    self._engine = PaddleOCR(
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=self.use_angle_cls,
                        lang=self._paddle_lang,
                        enable_mkldnn=False,
                    )
                except (TypeError, ValueError):
                    try:
                        self._engine = PaddleOCR(
                            use_textline_orientation=self.use_angle_cls,
                            lang=self._paddle_lang,
                            enable_mkldnn=False,
                        )
                    except (TypeError, ValueError):
                        try:
                            self._engine = PaddleOCR(
                                use_angle_cls=self.use_angle_cls,
                                lang=self._paddle_lang,
                                enable_mkldnn=False,
                            )
                        except (TypeError, ValueError):
                            try:
                                self._engine = PaddleOCR(
                                    use_textline_orientation=self.use_angle_cls,
                                    lang=self._paddle_lang,
                                )
                            except (TypeError, ValueError):
                                try:
                                    self._engine = PaddleOCR(
                                        use_angle_cls=self.use_angle_cls,
                                        lang=self._paddle_lang,
                                    )
                                except (TypeError, ValueError):
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

    def _segment_line_into_words(
        self,
        np_image: np.ndarray,
        line_bbox: Optional[BoundingBox] = None,
        min_gap_px: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, Optional[BoundingBox]]]:
        """Segments a text line into word image crops and relative bounding boxes."""
        h, w = np_image.shape[:2]
        if h < 8 or w < 8:
            return [(np_image, line_bbox)]

        if HAS_CV2 and cv2 is not None:
            if len(np_image.shape) == 3:
                gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = np_image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            if len(np_image.shape) == 3:
                gray = (0.299 * np_image[:, :, 0] + 0.587 * np_image[:, :, 1] + 0.114 * np_image[:, :, 2]).astype(np.uint8)
            else:
                gray = np_image
            thresh = int(np.mean(gray))
            binary = np.where(gray < thresh, 255, 0).astype(np.uint8)

        v_proj = np.sum(binary > 0, axis=0)
        noise_thresh = max(1, int(round(h * 0.03)))
        if min_gap_px is None:
            min_gap_px = max(4, int(round(h * 0.12)))

        is_space = v_proj <= noise_thresh
        ink_cols = np.where(v_proj > noise_thresh)[0]
        if len(ink_cols) == 0:
            return [(np_image, line_bbox)]

        first_ink, last_ink = int(ink_cols[0]), int(ink_cols[-1])

        gaps: List[Tuple[int, int]] = []
        start = None
        for x in range(first_ink, last_ink + 1):
            sp = is_space[x]
            if sp and start is None:
                start = x
            elif not sp and start is not None:
                if (x - start) >= min_gap_px:
                    gaps.append((start, x))
                start = None

        word_spans: List[Tuple[int, int]] = []
        curr_x = first_ink
        for g_start, g_end in gaps:
            if (g_start - curr_x) >= 4:
                word_spans.append((curr_x, g_start))
            curr_x = g_end
        if (last_ink + 1 - curr_x) >= 4:
            word_spans.append((curr_x, last_ink + 1))

        if not word_spans:
            word_spans = [(first_ink, last_ink + 1)]

        word_items: List[Tuple[np.ndarray, Optional[BoundingBox]]] = []
        base_x = line_bbox.x_min if line_bbox else 0.0
        base_y = line_bbox.y_min if line_bbox else 0.0
        base_ymax = line_bbox.y_max if line_bbox else float(h)

        for wx1, wx2 in word_spans:
            w_crop = np_image[:, wx1:wx2]
            w_box = BoundingBox(
                x_min=base_x + float(wx1),
                y_min=base_y,
                x_max=base_x + float(wx2),
                y_max=base_ymax,
            )
            word_items.append((w_crop, w_box))

        return word_items

    def _recognize_line_with_words(
        self,
        np_image: np.ndarray,
        paddlex_rec: Any,
        bbox: Optional[BoundingBox] = None,
    ) -> Tuple[str, float]:
        """Recognizes a text line crop by segmenting words and reconstructing inter-word spacing."""
        word_items = self._segment_line_into_words(np_image, line_bbox=bbox)

        if len(word_items) <= 1:
            rec_out = list(paddlex_rec.predict(np_image))
            if rec_out and len(rec_out) > 0:
                first_out = rec_out[0]
                t_val = first_out.get("rec_text", "") if hasattr(first_out, "get") else getattr(first_out, "rec_text", "")
                s_val = first_out.get("rec_score", 0.0) if hasattr(first_out, "get") else getattr(first_out, "rec_score", 0.0)
                return str(t_val).strip(), float(s_val)
            return "", 0.0

        word_texts: List[str] = []
        word_scores: List[float] = []
        word_boxes: List[DetectedWordBox] = []

        for w_crop, w_box in word_items:
            rec_out = list(paddlex_rec.predict(w_crop))
            if rec_out and len(rec_out) > 0:
                first_out = rec_out[0]
                t_val = first_out.get("rec_text", "") if hasattr(first_out, "get") else getattr(first_out, "rec_text", "")
                s_val = first_out.get("rec_score", 0.0) if hasattr(first_out, "get") else getattr(first_out, "rec_score", 0.0)
                txt = str(t_val).strip()
                word_texts.append(txt)
                word_scores.append(float(s_val))
            else:
                word_texts.append("")
                word_scores.append(0.0)

            word_boxes.append(DetectedWordBox(bbox=w_box))

        eff_bbox = bbox or BoundingBox(x_min=0, y_min=0, x_max=np_image.shape[1], y_max=np_image.shape[0])
        lg = DetectedLineGroup(
            line_index=1,
            line_bbox=eff_bbox,
            word_boxes=word_boxes,
        )
        reconstructed = lg.reconstruct_line_text(word_texts, space_gap_ratio=0.12)

        total_chars = sum(len(t) for t in word_texts)
        if total_chars > 0:
            mean_conf = sum(sc * len(t) for t, sc in zip(word_texts, word_scores)) / total_chars
        elif word_scores:
            mean_conf = sum(word_scores) / len(word_scores)
        else:
            mean_conf = 0.0

        return reconstructed, float(mean_conf)

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
        recognized_lines: List[str] = []
        confidences: List[float] = []
        detected_boxes: List[Optional[BoundingBox]] = []

        try:
            # Check if this is a line crop and direct text recognition predictor is available
            paddlex_rec = getattr(getattr(getattr(self._engine, "paddlex_pipeline", None), "_pipeline", None), "text_rec_model", None)
            is_line_crop = (bbox is not None) or (np_image.shape[0] < 150) or (np_image.shape[1] / max(np_image.shape[0], 1) > 2.5)

            if paddlex_rec is not None and is_line_crop:
                try:
                    line_text, line_conf = self._recognize_line_with_words(np_image, paddlex_rec, bbox=bbox)
                    if line_text:
                        recognized_lines.append(line_text)
                        confidences.append(line_conf)
                        detected_boxes.append(bbox)
                except Exception as rec_err:
                    logger.debug("Word-segmented line recognition notice: %s", rec_err)

            raw_results = None
            if not recognized_lines:
                # PaddleOCR returns: [ [ [polygon, (text, confidence)], ... ] ]
                # Filter kwargs to prevent extraneous pipeline parameters (e.g. enable_multipass) from failing PaddleX predict()
                paddle_kwargs = {k: v for k, v in kwargs.items() if k not in ("enable_multipass", "multipass", "page_number", "cls")}
                try:
                    raw_results = self._engine.ocr(np_image, **paddle_kwargs)
                except (TypeError, ValueError):
                    raw_results = self._engine.ocr(np_image)
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
        if raw_results and raw_results[0] is not None and not recognized_lines:
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
                                except (ValueError, TypeError) as score_err:
                                    logger.debug("Failed parsing confidence score: %s", score_err)

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
                except Exception as parse_err:
                    logger.warning("PaddleX 3.x result parsing error: %s", parse_err)

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
                    en_engine = PaddleOCR(
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=self.use_angle_cls,
                        lang="en",
                        enable_mkldnn=False,
                    )
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
                    # STRICTLY when NO Kannada unicode characters exist AND mostly ASCII or very low confidence
                    if not has_kn and (ascii_ratio > 0.7 or (l_cnf is not None and l_cnf < 0.40)):
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
                            try:
                                en_res = en_engine.ocr(crop_np)
                            except Exception:
                                en_res = None

                            en_clean = ""
                            en_cnf = 0.0
                            if en_res and en_res[0]:
                                item = en_res[0]
                                if hasattr(item, "get") or hasattr(item, "rec_texts"):
                                    texts = item.get("rec_texts", []) if hasattr(item, "get") else getattr(item, "rec_texts", [])
                                    scores = item.get("rec_scores", []) if hasattr(item, "get") else getattr(item, "rec_scores", [])
                                    en_clean = " ".join(str(t).strip() for t in texts if t).strip()
                                    en_cnf = float(scores[0]) if scores else 0.0
                                elif isinstance(item, (list, tuple)) and item and isinstance(item[0], (list, tuple)):
                                    en_clean = str(item[0][0]).strip()
                                    en_cnf = float(item[0][1])

                            if en_clean and en_cnf > 0.40:
                                is_numeric_or_code = any(c.isdigit() for c in en_clean) and sum(1 for c in en_clean if c.isalnum() or c in "/-.,") / max(len(en_clean), 1) > 0.7
                                is_header = any(w in en_clean.upper() for w in ["MUTATION", "REGISTER", "RULE", "FORM", "ORDER", "SURVEY", "TALUK", "VILLAGE"])
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
                                up_crop = line_crop.resize((int(cw * 1.5), int(ch * 1.5)), Image.Resampling.LANCZOS)
                                up_np = np.array(up_crop.convert("RGB"))
                                try:
                                    kn_retry = self._engine.ocr(up_np)
                                except Exception:
                                    kn_retry = None

                                if kn_retry and kn_retry[0]:
                                    k_item = kn_retry[0]
                                    r_clean = ""
                                    r_cnf = 0.0
                                    if hasattr(k_item, "get") or hasattr(k_item, "rec_texts"):
                                        r_texts = k_item.get("rec_texts", []) if hasattr(k_item, "get") else getattr(k_item, "rec_texts", [])
                                        r_scores = k_item.get("rec_scores", []) if hasattr(k_item, "get") else getattr(k_item, "rec_scores", [])
                                        r_clean = " ".join(str(t).strip() for t in r_texts if t).strip()
                                        r_cnf = float(r_scores[0]) if r_scores else 0.0
                                    elif isinstance(k_item, (list, tuple)) and k_item and isinstance(k_item[0], (list, tuple)):
                                        r_clean = str(k_item[0][0]).strip()
                                        r_cnf = float(k_item[0][1])

                                    if r_clean and r_cnf > (l_cnf or 0.0):
                                        recognized_lines[l_idx] = r_clean
                                        confidences[l_idx] = r_cnf
                                        line_engines[l_idx] = "paddleocr-kannada-multipass"
                        except Exception as retry_err:
                            logger.debug("PaddleOCR multipass retry notice: %s", retry_err)

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
