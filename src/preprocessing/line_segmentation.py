"""Line segmentation and text block detection for handwritten land record documents.

Decomposes complex photographed village documents (such as mutation registers) into
individual text lines in natural reading order, preserving tabular columns, paragraph
structure, and generating high-quality crops for line-level handwriting OCR.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from schemas import BoundingBox

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


@dataclass
class LineCrop:
    """Represents a single segmented handwritten or printed text line crop."""
    line_id: str
    bbox: BoundingBox
    image_crop: Image.Image
    reading_order_index: int
    column_index: int = 0
    is_handwritten: bool = True
    language: str = "kannada"
    script: str = "Kannada"
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLineSegmenter:
    """Robust line segmenter combining morphological projection and contour grouping."""

    def __init__(
        self,
        min_line_height: int = 14,
        max_line_height: int = 180,
        min_line_width: int = 25,
        padding_px: int = 6,
        use_detector_fallback: bool = True,
    ):
        self.min_line_height = min_line_height
        self.max_line_height = max_line_height
        self.min_line_width = min_line_width
        self.padding_px = padding_px
        self.use_detector_fallback = use_detector_fallback
        self._paddle_ocr = None
        self._paddle_attempted = False

    def _get_paddle_ocr(self) -> Any:
        """Lazily initializes and caches a single PaddleOCR instance."""
        if self._paddle_ocr is None and not self._paddle_attempted:
            self._paddle_attempted = True
            import os
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            os.environ["FLAGS_use_mkldnn"] = "0"
            os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
            os.environ["FLAGS_enable_pir_api"] = "0"
            os.environ["FLAGS_enable_pir_in_executor"] = "0"
            try:
                import paddle
                paddle.set_flags({"FLAGS_use_mkldnn": False})
            except Exception:
                pass
            try:
                from paddleocr import PaddleOCR
                try:
                    self._paddle_ocr = PaddleOCR(
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        lang="ka",
                        enable_mkldnn=False,
                    )
                except (TypeError, ValueError):
                    try:
                        self._paddle_ocr = PaddleOCR(
                            use_textline_orientation=False,
                            lang="ka",
                            enable_mkldnn=False,
                        )
                    except (TypeError, ValueError):
                        try:
                            self._paddle_ocr = PaddleOCR(
                                use_angle_cls=False,
                                lang="ka",
                                enable_mkldnn=False,
                            )
                        except (TypeError, ValueError):
                            try:
                                self._paddle_ocr = PaddleOCR(
                                    use_textline_orientation=False,
                                    lang="ka",
                                )
                            except (TypeError, ValueError):
                                try:
                                    self._paddle_ocr = PaddleOCR(
                                        use_angle_cls=False,
                                        lang="ka",
                                    )
                                except (TypeError, ValueError):
                                    self._paddle_ocr = PaddleOCR(lang="ka")
            except Exception as e:
                logger.info(f"PaddleOCR not available in DocumentLineSegmenter: {e}")
                self._paddle_ocr = None
        return self._paddle_ocr

    def segment_into_lines(
        self,
        image: Image.Image,
        page_number: int = 1,
    ) -> List[LineCrop]:
        """Segments a document image into ordered individual text lines.

        Args:
            image: Preprocessed PIL Image (grayscale or RGB).
            page_number: Document page number.

        Returns:
            List of LineCrop objects sorted in document reading order.
        """
        w, h = image.size
        # Method 1: Try detector-based line discovery if Paddle is available
        lines = []
        if self.use_detector_fallback:
            try:
                lines = self._segment_via_paddle(image, page_number)
            except Exception as exc:
                logger.warning(f"Paddle line detection fallback: {exc}")
                lines = []

        # Method 2: Only use morphology fallback if Paddle produced ZERO usable lines or failed
        if not lines and HAS_CV2 and cv2 is not None:
            lines = self._segment_via_morphology(image, page_number)

        # Method 3: Fallback uniform horizontal slice grid if image has text but segmentation failed
        if not lines:
            lines = self._fallback_slice_grid(image, page_number)

        # Sort lines in natural reading order (detect columns, top to bottom within column)
        sorted_lines = self._sort_in_reading_order(lines, image_width=w, image_height=h)
        return sorted_lines

    def _segment_via_paddle(self, image: Image.Image, page_number: int) -> List[LineCrop]:
        """Discovers text line bounding boxes using Paddle OCR, clusters word tokens into full lines,
        and accurately classifies script (Kannada, English, Alphanumeric) and handwriting status."""
        ocr = self._get_paddle_ocr()
        if ocr is None:
            return []

        arr = np.array(image.convert("RGB"))
        try:
            res = ocr.ocr(arr)
        except Exception as ocr_err:
            logger.warning(f"Paddle OCR execution notice: {ocr_err}")
            return []

        if not res or not res[0]:
            return []

        w, h = image.size
        first_res = res[0]
        extracted_items = []

        if isinstance(first_res, dict) or hasattr(first_res, "get") or hasattr(first_res, "rec_texts"):
            # PaddleX 3.x OCRResult dictionary format
            rec_texts = first_res.get("rec_texts", []) if hasattr(first_res, "get") else getattr(first_res, "rec_texts", [])
            rec_scores = first_res.get("rec_scores", []) if hasattr(first_res, "get") else getattr(first_res, "rec_scores", [])
            rec_boxes = first_res.get("rec_boxes", []) if hasattr(first_res, "get") else getattr(first_res, "rec_boxes", [])
            rec_polys = first_res.get("rec_polys", []) if hasattr(first_res, "get") else getattr(first_res, "rec_polys", [])

            for i in range(len(rec_texts)):
                rec_text = str(rec_texts[i]).strip()
                rec_conf = float(rec_scores[i]) if i < len(rec_scores) else 0.80
                if not rec_text:
                    continue

                if i < len(rec_boxes) and rec_boxes[i] is not None:
                    bx = rec_boxes[i]
                    x1 = max(0, int(bx[0]))
                    y1 = max(0, int(bx[1]))
                    x2 = min(w, int(bx[2]))
                    y2 = min(h, int(bx[3]))
                elif i < len(rec_polys) and rec_polys[i] is not None:
                    poly = rec_polys[i]
                    xs = [pt[0] for pt in poly]
                    ys = [pt[1] for pt in poly]
                    x1 = max(0, int(min(xs)))
                    y1 = max(0, int(min(ys)))
                    x2 = min(w, int(max(xs)))
                    y2 = min(h, int(max(ys)))
                else:
                    continue

                box_w = x2 - x1
                box_h = y2 - y1
                if box_w < 5 or box_h < 5:
                    continue

                y_mid = (y1 + y2) / 2.0
                extracted_items.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "y_mid": y_mid, "h": box_h, "w": box_w,
                    "text": rec_text, "conf": rec_conf,
                })
        elif isinstance(first_res, (list, tuple)):
            # Classic PaddleOCR 2.x list format
            for item in first_res:
                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
                    b = item[0]
                    rec_text = str(item[1][0]).strip()
                    rec_conf = float(item[1][1])
                else:
                    continue

                if not rec_text:
                    continue

                xs = [pt[0] for pt in b]
                ys = [pt[1] for pt in b]
                x1 = max(0, int(min(xs)))
                y1 = max(0, int(min(ys)))
                x2 = min(w, int(max(xs)))
                y2 = min(h, int(max(ys)))

                box_w = x2 - x1
                box_h = y2 - y1
                if box_w < 5 or box_h < 5:
                    continue

                y_mid = (y1 + y2) / 2.0
                extracted_items.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "y_mid": y_mid, "h": box_h, "w": box_w,
                    "text": rec_text, "conf": rec_conf,
                })

        if not extracted_items:
            return []

        # Sort all items primarily top-to-bottom
        extracted_items.sort(key=lambda item: item["y1"])

        # Cluster word boxes into horizontal text lines
        line_clusters: List[List[Dict[str, Any]]] = []
        for item in extracted_items:
            assigned = False
            for cluster in line_clusters:
                # Compare with cluster average vertical position
                cluster_y_mids = [c["y_mid"] for c in cluster]
                cluster_heights = [c["h"] for c in cluster]
                avg_y_mid = sum(cluster_y_mids) / len(cluster_y_mids)
                avg_h = sum(cluster_heights) / len(cluster_heights)

                # Tolerance: overlap in Y band
                y_diff = abs(item["y_mid"] - avg_y_mid)
                if y_diff < max(avg_h * 0.65, item["h"] * 0.65, 12.0):
                    cluster.append(item)
                    assigned = True
                    break

            if not assigned:
                line_clusters.append([item])

        # Sort each line cluster left-to-right and construct unified LineCrops
        raw_crops: List[LineCrop] = []
        for idx, cluster in enumerate(line_clusters):
            cluster.sort(key=lambda c: c["x1"])

            lx1 = max(0, min(c["x1"] for c in cluster) - self.padding_px)
            ly1 = max(0, min(c["y1"] for c in cluster) - self.padding_px)
            lx2 = min(w, max(c["x2"] for c in cluster) + self.padding_px)
            ly2 = min(h, max(c["y2"] for c in cluster) + self.padding_px)

            # Combined text with space separation
            line_text = " ".join(c["text"] for c in cluster if c["text"]).strip()
            if not line_text:
                continue

            # Length-weighted confidence
            total_chars = sum(len(c["text"]) for c in cluster)
            if total_chars > 0:
                line_conf = sum(c["conf"] * len(c["text"]) for c in cluster) / total_chars
            else:
                line_conf = sum(c["conf"] for c in cluster) / len(cluster)

            crop_img = image.crop((lx1, ly1, lx2, ly2))
            bbox = BoundingBox(x_min=lx1, y_min=ly1, x_max=lx2, y_max=ly2)

            # Analyze script composition
            has_kannada = any('\u0c80' <= ch <= '\u0cff' for ch in line_text)
            has_latin = any(ch.isascii() and ch.isalpha() for ch in line_text)
            has_digits = any(ch.isdigit() for ch in line_text)

            if has_latin and not has_kannada:
                # English text (e.g., "Mrs. Dorothy Charles", "Bruhat Bangalore", "Property No")
                language = "english"
                script = "Latin"
                is_handwritten = False
            elif has_kannada and not has_latin:
                # Pure Kannada text (e.g., "ಬೃಹತ್ ಬೆಂಗಳೂರು ಮಹಾನಗರ ಪಾಲಿಕೆ", "ದೃಢೀಕರಣ ಪತ್ರ")
                language = "kannada"
                script = "Kannada"
                # Keep as printed if Paddle recognized it cleanly (conf > 0.45)
                is_handwritten = False
            elif has_digits and not has_latin and not has_kannada:
                # Pure numbers / dates / survey IDs (e.g., "68-76-470/a", "29-05-2024", "500.00")
                language = "english"
                script = "Numeric"
                is_handwritten = False
            else:
                # Mixed Kannada + English / Numbers
                language = "kannada"
                script = "Mixed"
                is_handwritten = False

            meta = {
                "source": "paddle_ocr_clustered",
                "pre_recognized_text": line_text,
                "pre_confidence": round(float(line_conf), 4),
                "page_number": page_number,
                "token_count": len(cluster),
            }

            raw_crops.append(
                LineCrop(
                    line_id=f"line_{idx+1:03d}",
                    bbox=bbox,
                    image_crop=crop_img,
                    reading_order_index=idx,
                    is_handwritten=is_handwritten,
                    language=language,
                    script=script,
                    metadata=meta,
                )
            )

        return raw_crops

    def _segment_via_morphology(self, image: Image.Image, page_number: int) -> List[LineCrop]:
        """Segments lines using horizontal morphological dilation and connected components."""
        gray = np.array(image.convert("L"))
        h, w = gray.shape

        # Otsu thresholding for stroke mask
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove vertical grid lines if document is a tabular register
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)
        stroke_no_vlines = cv2.subtract(thresh, vert_lines)

        # Dilate horizontally to merge character components into text line bars
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 4))
        dilated = cv2.dilate(stroke_no_vlines, horiz_kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lines: List[LineCrop] = []

        for idx, cnt in enumerate(contours):
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < self.min_line_width or bh < self.min_line_height or bh > self.max_line_height:
                continue

            x1 = max(0, x - self.padding_px)
            y1 = max(0, y - self.padding_px)
            x2 = min(w, x + bw + self.padding_px)
            y2 = min(h, y + bh + self.padding_px)

            bbox = BoundingBox(x_min=float(x1), y_min=float(y1), x_max=float(x2), y_max=float(y2))
            crop_img = image.crop((x1, y1, x2, y2))

            # Try quick recognition on morphology crop to classify script/handwriting
            rec_text = ""
            rec_conf = 0.0
            ocr = self._get_paddle_ocr()
            if ocr is not None:
                try:
                    c_arr = np.array(crop_img.convert("RGB"))
                    c_res = ocr.ocr(c_arr)
                    if c_res and c_res[0]:
                        cf = c_res[0]
                        if isinstance(cf, dict) or hasattr(cf, "get") or hasattr(cf, "rec_texts"):
                            c_texts = [str(t).strip() for t in (cf.get("rec_texts", []) if hasattr(cf, "get") else getattr(cf, "rec_texts", [])) if str(t).strip()]
                            c_confs = [float(s) for s in (cf.get("rec_scores", []) if hasattr(cf, "get") else getattr(cf, "rec_scores", []))]
                            rec_text = " ".join(c_texts).strip()
                            rec_conf = sum(c_confs) / len(c_confs) if c_confs else 0.0
                        elif isinstance(cf, (list, tuple)):
                            c_texts = [
                                str(item[1][0]).strip()
                                for item in cf
                                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple))
                            ]
                            c_confs = [
                                float(item[1][1])
                                for item in cf
                                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple))
                            ]
                            rec_text = " ".join(t for t in c_texts if t).strip()
                            rec_conf = sum(c_confs) / len(c_confs) if c_confs else 0.0
                except Exception as exc:
                    logger.debug(f"Morphology crop recognition notice: {exc}")

            has_kannada = any('\u0c80' <= ch <= '\u0cff' for ch in rec_text) if rec_text else False
            has_latin = any(ch.isascii() and ch.isalpha() for ch in rec_text) if rec_text else False
            has_digits = any(ch.isdigit() for ch in rec_text) if rec_text else False

            if has_latin and not has_kannada:
                language = "english"
                script = "Latin"
                is_handwritten = False
            elif has_kannada and not has_latin:
                language = "kannada"
                script = "Kannada"
                is_handwritten = False
            elif has_digits and not has_latin and not has_kannada:
                language = "english"
                script = "Numeric"
                is_handwritten = False
            elif rec_text:
                language = "kannada"
                script = "Mixed"
                is_handwritten = False
            else:
                language = "kannada"
                script = "Kannada"
                is_handwritten = True

            meta: Dict[str, Any] = {
                "source": "morphology",
                "page_number": page_number,
            }
            if rec_text:
                meta["pre_recognized_text"] = rec_text
                meta["pre_confidence"] = round(float(rec_conf), 4)

            lines.append(
                LineCrop(
                    line_id=f"morph_line_{idx+1:03d}",
                    bbox=bbox,
                    image_crop=crop_img,
                    reading_order_index=idx,
                    is_handwritten=is_handwritten,
                    language=language,
                    script=script,
                    metadata=meta,
                )
            )

        return lines

    def _fallback_slice_grid(self, image: Image.Image, page_number: int) -> List[LineCrop]:
        """Fallback: horizontal stripe partitioning when stroke extraction is uncertain."""
        w, h = image.size
        num_stripes = max(5, min(20, h // 60))
        stripe_h = h // num_stripes
        lines: List[LineCrop] = []

        for idx in range(num_stripes):
            y1 = idx * stripe_h
            y2 = min(h, (idx + 1) * stripe_h + self.padding_px)
            crop_img = image.crop((0, y1, w, y2))
            bbox = BoundingBox(x_min=0, y_min=y1, x_max=w, y_max=y2)
            lines.append(
                LineCrop(
                    line_id=f"grid_line_{idx+1:03d}",
                    bbox=bbox,
                    image_crop=crop_img,
                    reading_order_index=idx,
                    is_handwritten=True,
                    language="kannada",
                    script="Kannada",
                    metadata={"source": "grid_slice", "page_number": page_number},
                )
            )
        return lines

    def _sort_in_reading_order(
        self,
        lines: List[LineCrop],
        image_width: int,
        image_height: int,
    ) -> List[LineCrop]:
        """Sorts line crops in natural reading order considering tabular column layouts."""
        if not lines:
            return []

        # Detect column boundaries if line x_midpoints cluster into multiple columns
        x_mids = [(l.bbox.x_min + l.bbox.x_max) / 2.0 for l in lines]
        # Cluster into 3 broad document column regions if document is wide (landscape)
        is_wide = image_width > image_height

        def get_column_idx(x_mid: float) -> int:
            if not is_wide:
                return 0
            if x_mid < image_width * 0.28:
                return 0
            elif x_mid < image_width * 0.70:
                return 1
            else:
                return 2

        for l in lines:
            x_mid = (l.bbox.x_min + l.bbox.x_max) / 2.0
            l.column_index = get_column_idx(x_mid)

        # Sort: Primary by column_index, Secondary by vertical position y_min
        # For header lines spanning across the top (y < 12% of height), place them first regardless of column
        def sort_key(l: LineCrop) -> Tuple[int, int, float]:
            is_top_header = 0 if l.bbox.y_min < image_height * 0.12 else 1
            return (is_top_header, l.column_index, l.bbox.y_min)

        sorted_list = sorted(lines, key=sort_key)
        for order_idx, l in enumerate(sorted_list):
            l.reading_order_index = order_idx + 1
            l.line_id = f"line_{order_idx+1:03d}"

        return sorted_list
