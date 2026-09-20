"""PaddleOCR-based Text Detection & Geometric Spacing Reconstruction.

Discovers line and word bounding boxes on both handwritten and printed documents,
clusters tokens into natural baselines, and reconstructs authentic inter-word spacing
from physical bounding box geometry.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from schemas import BoundingBox

logger = logging.getLogger(__name__)

# Soft import PaddleOCR & EasyOCR
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    PaddleOCR = None
    HAS_PADDLEOCR = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    easyocr = None
    HAS_EASYOCR = False


@dataclass
class DetectedWordBox:
    """Represents an individual detected word/token box."""
    bbox: BoundingBox
    confidence: float = 0.90
    crop: Optional[Image.Image] = None


@dataclass
class DetectedLineGroup:
    """Represents a horizontal text line grouping with geometric token gaps."""
    line_index: int
    line_bbox: BoundingBox
    word_boxes: List[DetectedWordBox] = field(default_factory=list)
    line_crop: Optional[Image.Image] = None

    def reconstruct_line_text(
        self,
        word_texts: List[str],
        space_gap_ratio: float = 0.22,
    ) -> str:
        """Reconstructs line text with genuine spaces derived from inter-box geometry.

        Args:
            word_texts: List of recognized string tokens corresponding to word_boxes.
            space_gap_ratio: Minimum fraction of line height required to declare a space.

        Returns:
            Space-delimited line string reflecting authentic physical layout.
        """
        if not word_texts:
            return ""
        if len(word_texts) == 1:
            return word_texts[0].strip()

        result_tokens = [word_texts[0].strip()]
        line_height = max(10.0, self.line_bbox.height)

        for i in range(1, len(word_texts)):
            prev_box = self.word_boxes[i - 1].bbox
            curr_box = self.word_boxes[i].bbox

            gap = curr_box.x_min - prev_box.x_max
            threshold = space_gap_ratio * line_height

            curr_text = word_texts[i].strip()
            if gap >= threshold:
                result_tokens.append(" " + curr_text)
            else:
                result_tokens.append(curr_text)

        return "".join(result_tokens).strip()


class DocumentTextDetector:
    """Detects text regions and reconstructs physical spacing using DBNet geometry."""

    def __init__(
        self,
        paddle_engine: Optional[Any] = None,
        min_box_size: int = 8,
        space_gap_ratio: float = 0.22,
    ):
        self.min_box_size = min_box_size
        self.space_gap_ratio = space_gap_ratio
        self._paddle_engine = paddle_engine

    def _get_engine(self) -> Optional[Any]:
        """Lazily initializes PaddleOCR detector."""
        if self._paddle_engine is not None:
            return self._paddle_engine
        if HAS_PADDLEOCR and PaddleOCR is not None:
            try:
                self._paddle_engine = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    lang="en",  # Latin DBNet generalizes across all scripts for detection
                    enable_mkldnn=False,
                )
            except (TypeError, ValueError):
                try:
                    self._paddle_engine = PaddleOCR(
                        use_textline_orientation=False,
                        lang="en",
                        enable_mkldnn=False,
                    )
                except Exception as e:
                    logger.warning(f"PaddleOCR detector init notice: {e}")
                    self._paddle_engine = None
            except Exception as e:
                logger.warning(f"PaddleOCR detector init notice: {e}")
                self._paddle_engine = None
        return self._paddle_engine

    def detect_lines_and_words(
        self,
        image: Image.Image,
    ) -> List[DetectedLineGroup]:
        """Discovers text boxes and clusters them into geometric line groupings."""
        w, h = image.size
        engine = self._get_engine()
        raw_boxes: List[BoundingBox] = []

        if engine is not None:
            try:
                arr = np.array(image.convert("RGB"))
                # Check if PaddleX direct text_det_model is available
                paddlex_det = getattr(getattr(getattr(engine, "paddlex_pipeline", None), "_pipeline", None), "text_det_model", None)
                polys = []
                if paddlex_det is not None:
                    p_res = list(paddlex_det.predict(arr))
                    if p_res and len(p_res) > 0:
                        polys = p_res[0].get("dt_polys", [])
                else:
                    try:
                        det_res = engine.ocr(arr, rec=False)
                    except (TypeError, ValueError):
                        det_res = engine.ocr(arr)
                    if det_res and len(det_res) > 0:
                        first_item = det_res[0] if isinstance(det_res, list) else det_res
                        if isinstance(first_item, list):
                            polys = first_item

                for poly in polys:
                            if isinstance(poly, list) and len(poly) >= 4:
                                xs = [pt[0] for pt in poly]
                                ys = [pt[1] for pt in poly]
                                x_min, x_max = max(0, min(xs)), min(w, max(xs))
                                y_min, y_max = max(0, min(ys)), min(h, max(ys))
                                if (x_max - x_min) >= self.min_box_size and (y_max - y_min) >= self.min_box_size:
                                    raw_boxes.append(BoundingBox(
                                        x_min=float(x_min),
                                        y_min=float(y_min),
                                        x_max=float(x_max),
                                        y_max=float(y_max),
                                    ))
            except Exception as exc:
                logger.warning(f"Paddle text detection exception: {exc}")
                raw_boxes = []

        # Real CRAFT detector fallback if Paddle is unavailable
        if not raw_boxes and HAS_EASYOCR and easyocr is not None:
            try:
                if not hasattr(self, "_easyocr_reader") or self._easyocr_reader is None:
                    self._easyocr_reader = easyocr.Reader(['kn', 'en'], gpu=False, download_enabled=False)
                arr = np.array(image.convert("RGB"))
                bboxes, _ = self._easyocr_reader.detect(arr)
                if bboxes and len(bboxes[0]) > 0:
                    for b in bboxes[0]:
                        x_min, x_max, y_min, y_max = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                        if (x_max - x_min) >= self.min_box_size and (y_max - y_min) >= self.min_box_size:
                            raw_boxes.append(BoundingBox(
                                x_min=max(0.0, x_min),
                                y_min=max(0.0, y_min),
                                x_max=min(float(w), x_max),
                                y_max=min(float(h), y_max),
                            ))
            except Exception as ez_exc:
                logger.warning(f"EasyOCR text detection notice: {ez_exc}")

        # Fallback: if detection yielded zero boxes, treat full image as single region
        if not raw_boxes:
            raw_boxes = [BoundingBox(x_min=0.0, y_min=0.0, x_max=float(w), y_max=float(h))]

        # Cluster word boxes into line groups by vertical baseline overlap
        return self._cluster_into_lines(image, raw_boxes)

    def _cluster_into_lines(
        self,
        image: Image.Image,
        boxes: List[BoundingBox],
    ) -> List[DetectedLineGroup]:
        """Clusters bounding boxes into lines and sorts each line horizontally."""
        if not boxes:
            return []

        # Sort primarily by top Y coordinate
        sorted_boxes = sorted(boxes, key=lambda b: (b.y_min, b.x_min))

        line_clusters: List[List[BoundingBox]] = []
        for box in sorted_boxes:
            box_h = max(1.0, box.height)
            box_cy = (box.y_min + box.y_max) / 2.0
            matched = False

            for cluster in line_clusters:
                # Compare center Y with cluster average center Y
                cluster_cy = sum((b.y_min + b.y_max) / 2.0 for b in cluster) / len(cluster)
                cluster_h = sum(b.height for b in cluster) / len(cluster)
                tolerance = max(12.0, cluster_h * 0.45)

                if abs(box_cy - cluster_cy) <= tolerance:
                    cluster.append(box)
                    matched = True
                    break

            if not matched:
                line_clusters.append([box])

        # Convert each cluster into DetectedLineGroup
        line_groups: List[DetectedLineGroup] = []
        for line_idx, cluster in enumerate(line_clusters):
            # Sort boxes horizontally left-to-right within the line
            cluster_sorted = sorted(cluster, key=lambda b: b.x_min)

            l_xmin = min(b.x_min for b in cluster_sorted)
            l_ymin = min(b.y_min for b in cluster_sorted)
            l_xmax = max(b.x_max for b in cluster_sorted)
            l_ymax = max(b.y_max for b in cluster_sorted)

            line_bbox = BoundingBox(x_min=l_xmin, y_min=l_ymin, x_max=l_xmax, y_max=l_ymax)

            # Crop line image
            line_crop = image.crop((
                int(max(0, l_xmin)),
                int(max(0, l_ymin)),
                int(min(image.width, l_xmax)),
                int(min(image.height, l_ymax)),
            ))

            word_boxes = [
                DetectedWordBox(
                    bbox=b,
                    crop=image.crop((
                        int(max(0, b.x_min)),
                        int(max(0, b.y_min)),
                        int(min(image.width, b.x_max)),
                        int(min(image.height, b.y_max)),
                    )),
                )
                for b in cluster_sorted
            ]

            line_groups.append(DetectedLineGroup(
                line_index=line_idx + 1,
                line_bbox=line_bbox,
                word_boxes=word_boxes,
                line_crop=line_crop,
            ))

        return line_groups
