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
                logger.debug(f"Paddle line detection fallback: {exc}")
                lines = []

        # Method 2: If detector found fewer than 5 lines, use morphological line grouping
        if len(lines) < 5 and HAS_CV2 and cv2 is not None:
            morph_lines = self._segment_via_morphology(image, page_number)
            if len(morph_lines) >= len(lines):
                lines = morph_lines

        # Method 3: Fallback uniform horizontal slice grid if image has text but segmentation failed
        if not lines:
            lines = self._fallback_slice_grid(image, page_number)

        # Sort lines in natural reading order (detect columns, top to bottom within column)
        sorted_lines = self._sort_in_reading_order(lines, image_width=w, image_height=h)
        return sorted_lines

    def _segment_via_paddle(self, image: Image.Image, page_number: int) -> List[LineCrop]:
        """Discovers text line bounding boxes using Paddle OCR model and classifies printed vs handwritten."""
        import os
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=False, lang="ka", enable_mkldnn=False, show_log=False)
        arr = np.array(image.convert("RGB"))
        res = ocr.ocr(arr, cls=False, rec=True)
        if not res or not res[0]:
            return []

        boxes = res[0]
        raw_crops: List[LineCrop] = []
        w, h = image.size

        for idx, item in enumerate(boxes):
            # item format: [box_pts, (rec_text, rec_conf)] or box_pts if rec=False
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], (list, tuple)):
                b = item[0]
                rec_text, rec_conf = str(item[1][0]).strip(), float(item[1][1])
            else:
                b = item
                rec_text, rec_conf = "", 0.0

            xs = [pt[0] for pt in b]
            ys = [pt[1] for pt in b]
            x1 = max(0, int(min(xs)) - self.padding_px)
            y1 = max(0, int(min(ys)) - self.padding_px)
            x2 = min(w, int(max(xs)) + self.padding_px)
            y2 = min(h, int(max(ys)) + self.padding_px)

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < self.min_line_width or box_h < self.min_line_height:
                continue

            crop_img = image.crop((x1, y1, x2, y2))
            bbox = BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)

            # Classify printed English vs printed Kannada vs handwriting
            has_latin = any(c.isascii() and c.isalpha() for c in rec_text)
            is_printed_english = has_latin and rec_conf > 0.40
            is_printed_kannada = (not has_latin) and (rec_conf > 0.88)

            if is_printed_english:
                is_handwritten = False
                language = "english"
                script = "Latin"
                meta = {
                    "source": "paddle_ocr",
                    "pre_recognized_text": rec_text,
                    "pre_confidence": rec_conf,
                    "page_number": page_number,
                }
            elif is_printed_kannada:
                is_handwritten = False
                language = "kannada"
                script = "Kannada"
                meta = {
                    "source": "paddle_ocr",
                    "pre_recognized_text": rec_text,
                    "pre_confidence": rec_conf,
                    "page_number": page_number,
                }
            else:
                is_handwritten = True
                language = "kannada"
                script = "Kannada"
                meta = {
                    "source": "paddle_det",
                    "paddle_hint": rec_text if rec_text else None,
                    "paddle_hint_conf": rec_conf if rec_conf > 0 else None,
                    "page_number": page_number,
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

            crop_img = image.crop((x1, y1, x2, y2))
            bbox = BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)

            lines.append(
                LineCrop(
                    line_id=f"morph_line_{idx+1:03d}",
                    bbox=bbox,
                    image_crop=crop_img,
                    reading_order_index=idx,
                    is_handwritten=True,
                    language="kannada",
                    script="Kannada",
                    metadata={"source": "morphology", "page_number": page_number},
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
