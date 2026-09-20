"""Handwritten Line Word Decomposition & Reading-Order Reconstruction.

Decomposes complex multi-word handwritten Kannada line crops into candidate
word bounding boxes using morphological connected-component clustering,
vertical projection profiles, and detector fallbacks.

Features:
- Kannada-adapted morphological dilation (bridges intra-word akshara/matra gaps while preserving inter-word spacing).
- Left-to-right reading order sorting.
- Boundary filtering and plausibility verification (detects zero words, excessive overlaps, monolithic single boxes).
- Dual storage of original line crop, detected word boxes, and per-word OCR predictions.
- Debug visualization mode drawing bounding boxes and labels (word_1, word_2, ...).
- Transparent failure tracking without forced recognition.
"""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from schemas import BoundingBox
from src.preprocessing.image_enhancement import load_image_as_pil

logger = logging.getLogger("line_word_decomposer")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


@dataclass
class WordBoxCandidate:
    """Represents a segmented word candidate within a line crop."""
    box_index: int
    bbox: BoundingBox
    crop: Image.Image
    confidence: float = 0.90
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineDecompositionResult:
    """Structured result of line-level word decomposition."""
    line_id: str
    original_line_crop: Image.Image
    word_boxes: List[BoundingBox] = field(default_factory=list)
    word_candidates: List[WordBoxCandidate] = field(default_factory=list)
    is_valid: bool = True
    needs_review: bool = False
    failure_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class LineWordDecomposer:
    """Decomposes handwritten Kannada line images into discrete word boxes."""

    def __init__(
        self,
        min_word_width: int = 8,
        min_word_height: int = 10,
        max_overlap_iou: float = 0.40,
        aspect_ratio_line_threshold: float = 3.0,
        enable_debug_viz: bool = False,
        debug_output_dir: Optional[Union[str, Path]] = None,
    ):
        self.min_word_width = min_word_width
        self.min_word_height = min_word_height
        self.max_overlap_iou = max_overlap_iou
        self.aspect_ratio_line_threshold = aspect_ratio_line_threshold
        self.enable_debug_viz = enable_debug_viz
        self.debug_output_dir = Path(debug_output_dir) if debug_output_dir else None

    @staticmethod
    def _binarize(img_arr: np.ndarray) -> np.ndarray:
        """Binarizes grayscale image to binary mask (255 = ink foreground, 0 = background)."""
        if len(img_arr.shape) == 3:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_arr.copy()

        # Check if text is dark on light background (standard paper)
        # If mean intensity is high, background is light, text is dark
        if np.mean(gray) > 127:
            # Invert so ink is white (255)
            inverted = cv2.bitwise_not(gray)
        else:
            inverted = gray

        # Otsu thresholding
        _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def decompose(
        self,
        line_image: Union[Image.Image, np.ndarray, str, Path],
        line_id: str = "line_0",
        save_debug: Optional[bool] = None,
    ) -> LineDecompositionResult:
        """Decomposes a single line image into candidate word boxes in left-to-right reading order.

        Args:
            line_image: PIL Image or array of the line crop.
            line_id: Identifier for the line crop.
            save_debug: Optional override to save debug annotated image.

        Returns:
            LineDecompositionResult containing sorted word boxes, crops, and plausibility status.
        """
        pil_img = load_image_as_pil(line_image).convert("RGB")
        w, h = pil_img.size
        aspect_ratio = float(w) / max(1.0, float(h))

        # Case 1: Extremely small or degenerated image
        if w < 10 or h < 8:
            return LineDecompositionResult(
                line_id=line_id,
                original_line_crop=pil_img,
                word_boxes=[],
                is_valid=False,
                needs_review=True,
                failure_reason="Line crop dimensions too small to decompose (<10x8 px)",
                diagnostics={"width": w, "height": h, "aspect_ratio": aspect_ratio},
            )

        # Case 2: Short aspect ratio (likely a single word)
        if aspect_ratio < 2.0:
            box = BoundingBox(x_min=0.0, y_min=0.0, x_max=float(w), y_max=float(h))
            candidate = WordBoxCandidate(box_index=0, bbox=box, crop=pil_img.copy())
            return LineDecompositionResult(
                line_id=line_id,
                original_line_crop=pil_img,
                word_boxes=[box],
                word_candidates=[candidate],
                is_valid=True,
                needs_review=False,
                failure_reason=None,
                diagnostics={
                    "strategy": "single_word_direct",
                    "word_count": 1,
                    "aspect_ratio": aspect_ratio,
                },
            )

        if not HAS_CV2 or cv2 is None:
            # Fallback: cannot run morphological segmentation without OpenCV
            box = BoundingBox(x_min=0.0, y_min=0.0, x_max=float(w), y_max=float(h))
            candidate = WordBoxCandidate(box_index=0, bbox=box, crop=pil_img.copy())
            return LineDecompositionResult(
                line_id=line_id,
                original_line_crop=pil_img,
                word_boxes=[box],
                word_candidates=[candidate],
                is_valid=False,
                needs_review=True,
                failure_reason="OpenCV not available for morphological line decomposition",
                diagnostics={"strategy": "opencv_missing_fallback"},
            )

        # Multi-strategy segmentation:
        # Strategy A: Morphological Connected-Component Analysis (CCA)
        arr = np.array(pil_img)
        binary = self._binarize(arr)

        boxes_cca, diag_cca = self._segment_morphological_cca(binary, w, h)
        
        # Validate candidate boxes
        valid_boxes, failure_reason = self._validate_and_filter_boxes(boxes_cca, w, h, aspect_ratio)

        # If CCA failed or produced single monolithic box on very wide line, try Vertical Projection Valleys
        strategy_used = "morphological_cca"
        if not valid_boxes or (len(valid_boxes) == 1 and aspect_ratio > 4.0):
            boxes_proj, diag_proj = self._segment_projection_valleys(binary, w, h)
            alt_boxes, alt_reason = self._validate_and_filter_boxes(boxes_proj, w, h, aspect_ratio)
            if alt_boxes and (len(alt_boxes) > len(valid_boxes) or not valid_boxes):
                valid_boxes = alt_boxes
                failure_reason = alt_reason
                strategy_used = "projection_valleys"

        # If both CCA and projection valleys failed on multi-word line, try component gap clustering
        if not valid_boxes and aspect_ratio >= 2.5:
            boxes_gaps, diag_gaps = self._segment_component_gaps(binary, w, h)
            alt_boxes, alt_reason = self._validate_and_filter_boxes(boxes_gaps, w, h, aspect_ratio)
            if alt_boxes:
                valid_boxes = alt_boxes
                failure_reason = alt_reason
                strategy_used = "component_gap_clustering"

        # Final Plausibility Checks
        if not valid_boxes:
            fail_msg = failure_reason or "Zero valid word boundaries detected in line crop"
            logger.info("Line %s decomposition rejected: %s", line_id, fail_msg)
            return LineDecompositionResult(
                line_id=line_id,
                original_line_crop=pil_img,
                word_boxes=[],
                word_candidates=[],
                is_valid=False,
                needs_review=True,
                failure_reason=fail_msg,
                diagnostics={
                    "strategy": strategy_used,
                    "aspect_ratio": aspect_ratio,
                    "failure_reason": fail_msg,
                },
            )

        # Sort strictly Left-to-Right
        sorted_boxes = sorted(valid_boxes, key=lambda b: b.x_min)

        # Build WordBoxCandidates with safe crop boundary padding
        candidates: List[WordBoxCandidate] = []
        for idx, bbox in enumerate(sorted_boxes):
            # Clamp padding (2px horizontal, 2px vertical)
            x1 = max(0, int(bbox.x_min) - 2)
            y1 = max(0, int(bbox.y_min) - 2)
            x2 = min(w, int(bbox.x_max) + 2)
            y2 = min(h, int(bbox.y_max) + 2)
            crop = pil_img.crop((x1, y1, x2, y2))
            
            # Sub-box coordinates normalized to original line
            sub_box = BoundingBox(
                x_min=float(x1),
                y_min=float(y1),
                x_max=float(x2),
                y_max=float(y2),
            )
            candidates.append(
                WordBoxCandidate(
                    box_index=idx,
                    bbox=sub_box,
                    crop=crop,
                    confidence=0.90,
                )
            )

        result = LineDecompositionResult(
            line_id=line_id,
            original_line_crop=pil_img,
            word_boxes=[c.bbox for c in candidates],
            word_candidates=candidates,
            is_valid=True,
            needs_review=False,
            failure_reason=None,
            diagnostics={
                "strategy": strategy_used,
                "word_count": len(candidates),
                "aspect_ratio": aspect_ratio,
                "original_dimensions": {"width": w, "height": h},
            },
        )

        # Optional Debug Visualization
        should_debug = save_debug if save_debug is not None else self.enable_debug_viz
        if should_debug and self.debug_output_dir:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)
            viz_path = self.debug_output_dir / f"{line_id}_words_debug.png"
            labels = [f"word_{i+1}" for i in range(len(candidates))]
            self.visualize_line_decomposition(pil_img, [c.bbox for c in candidates], labels, viz_path)

        return result

    def _preprocess_binary_mask(self, binary: np.ndarray, width: int, height: int) -> np.ndarray:
        """Removes thin horizontal ruled table lines and margin edge bleed while preserving text glyphs."""
        cleaned = binary.copy()
        # Ruled lines are thin (height 1-3px) and wide (>= 20% of line width or >= 40px)
        h_line_w = max(40, int(width * 0.20))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_line_w, 1))
        h_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, h_kernel)

        # Verify that detected line is actually thin vertically (not a block of text)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
        thick_areas = cv2.morphologyEx(h_lines, cv2.MORPH_OPEN, v_kernel)
        thin_lines = cv2.subtract(h_lines, thick_areas)
        cleaned = cv2.subtract(cleaned, thin_lines)

        # Suppress border bleed (3% top/bottom and 1% left/right)
        margin_y = max(1, int(height * 0.03))
        margin_x = max(1, int(width * 0.005))
        cleaned[:margin_y, :] = 0
        cleaned[-margin_y:, :] = 0
        cleaned[:, :margin_x] = 0
        cleaned[:, -margin_x:] = 0
        return cleaned

    def _segment_morphological_cca(
        self,
        binary: np.ndarray,
        width: int,
        height: int,
    ) -> Tuple[List[BoundingBox], Dict[str, Any]]:
        """Extracts word clusters using morphological closing and connected components."""
        cleaned = self._preprocess_binary_mask(binary, width, height)

        # Kernel size dynamically proportional to line height:
        # Kannada intra-character gap is small (1-3px), intra-word gap between aksharas is ~4-8px,
        # while inter-word whitespace is > 14-25px.
        k_w = max(4, min(14, int(height * 0.14)))
        k_h = max(2, min(4, int(height * 0.04)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))

        # Close horizontally to fuse letters of same word
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        # Find external contours of connected ink blobs
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            # Filter noise
            if bw < self.min_word_width or bh < self.min_word_height:
                continue
            # Filter full-image frame artifacts
            if bw >= width * 0.98 and bh >= height * 0.98:
                continue
            raw_boxes.append(BoundingBox(
                x_min=float(x),
                y_min=float(y),
                x_max=float(x + bw),
                y_max=float(y + bh),
            ))

        return raw_boxes, {"raw_contour_count": len(contours)}

    def _segment_projection_valleys(
        self,
        binary: np.ndarray,
        width: int,
        height: int,
    ) -> Tuple[List[BoundingBox], Dict[str, Any]]:
        """Segments line into words using vertical projection profile whitespace valleys."""
        cleaned = self._preprocess_binary_mask(binary, width, height)
        # Focus projection on middle 80% vertical strip to avoid ascender/descender horizontal touching
        mid_y1 = int(height * 0.10)
        mid_y2 = int(height * 0.90)
        mid_strip = cleaned[mid_y1:mid_y2, :] if mid_y2 > mid_y1 else cleaned

        # Sum foreground ink per column
        proj = np.sum(mid_strip > 0, axis=0)

        # Background threshold: <= 3% of line height containing ink
        gap_threshold = max(1, int(height * 0.03))
        is_whitespace = proj <= gap_threshold

        # Find continuous ink segments
        in_word = False
        start_x = 0
        word_spans: List[Tuple[int, int]] = []
        min_gap_len = max(6, int(height * 0.16))  # Minimum gap width to declare word break

        whitespace_count = 0
        for x in range(width):
            if not is_whitespace[x]:
                if not in_word:
                    in_word = True
                    start_x = max(0, x - whitespace_count // 2 if whitespace_count < min_gap_len else x)
                whitespace_count = 0
            else:
                whitespace_count += 1
                if in_word and whitespace_count >= min_gap_len:
                    end_x = x - whitespace_count
                    if (end_x - start_x) >= self.min_word_width:
                        word_spans.append((start_x, end_x))
                    in_word = False

        if in_word:
            end_x = width
            if (end_x - start_x) >= self.min_word_width:
                word_spans.append((start_x, end_x))

        boxes = []
        for sx, ex in word_spans:
            sub_mask = cleaned[:, sx:ex]
            ink_ys = np.where(np.sum(sub_mask, axis=1) > 0)[0]
            if len(ink_ys) > 0:
                y_min = float(ink_ys[0])
                y_max = float(ink_ys[-1] + 1)
            else:
                y_min = 0.0
                y_max = float(height)

            boxes.append(BoundingBox(
                x_min=float(sx),
                y_min=y_min,
                x_max=float(ex),
                y_max=y_max,
            ))

        return boxes, {"span_count": len(word_spans)}

    def _segment_component_gaps(
        self,
        binary: np.ndarray,
        width: int,
        height: int,
    ) -> Tuple[List[BoundingBox], Dict[str, Any]]:
        """Clusters individual character/glyph connected components by inter-word whitespace gaps."""
        cleaned = self._preprocess_binary_mask(binary, width, height)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
        valid_comps = []
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            if area >= 12 and bw >= 3 and bh >= 5:
                if bw < width * 0.90 and bh < height * 0.95:
                    valid_comps.append((x, y, x + bw, y + bh))

        if not valid_comps:
            return [], {"reason": "no_valid_components"}

        valid_comps.sort(key=lambda b: b[0])
        med_h = float(np.median([b[3] - b[1] for b in valid_comps]))
        gap_threshold = max(10, int(med_h * 0.35))

        words = []
        curr = list(valid_comps[0])
        for b in valid_comps[1:]:
            gap = b[0] - curr[2]
            if gap <= gap_threshold:
                curr[0] = min(curr[0], b[0])
                curr[1] = min(curr[1], b[1])
                curr[2] = max(curr[2], b[2])
                curr[3] = max(curr[3], b[3])
            else:
                words.append(curr)
                curr = list(b)
        words.append(curr)

        boxes = [
            BoundingBox(x_min=float(w[0]), y_min=float(w[1]), x_max=float(w[2]), y_max=float(w[3]))
            for w in words
            if (w[2] - w[0]) >= self.min_word_width
        ]
        return boxes, {"gap_threshold": gap_threshold, "comp_count": len(valid_comps)}

    def _validate_and_filter_boxes(
        self,
        boxes: List[BoundingBox],
        width: int,
        height: int,
        aspect_ratio: float,
    ) -> Tuple[List[BoundingBox], Optional[str]]:
        """Validates candidate boxes against plausibility criteria.

        Returns:
            (filtered_boxes, failure_reason)
        """
        if not boxes:
            return [], "Zero word candidates detected"

        # 1. Merge vertically overlapping or nested fragments
        # Sort left-to-right
        sorted_b = sorted(boxes, key=lambda b: (b.x_min, b.x_max))
        merged: List[BoundingBox] = []

        for b in sorted_b:
            if not merged:
                merged.append(b)
                continue

            last = merged[-1]
            # Check horizontal overlap
            overlap_x = max(0.0, min(last.x_max, b.x_max) - max(last.x_min, b.x_min))
            min_w = min(last.width, b.width)

            # If overlap is high (> 60% of smaller box) or one is completely nested inside another
            if overlap_x > 0.60 * min_w or (b.x_min >= last.x_min and b.x_max <= last.x_max):
                # Union the two boxes
                new_box = BoundingBox(
                    x_min=min(last.x_min, b.x_min),
                    y_min=min(last.y_min, b.y_min),
                    x_max=max(last.x_max, b.x_max),
                    y_max=max(last.y_max, b.y_max),
                )
                merged[-1] = new_box
            else:
                merged.append(b)

        # 2. Check for single monolithic box or dominant box spanning whole line on multi-word line
        if aspect_ratio >= 3.0:
            if len(merged) == 1:
                single = merged[0]
                if (single.x_max - single.x_min) >= width * 0.85:
                    return [], "Candidate box covers almost the entire line width as one object without decomposing words"
            for b in merged:
                if (b.x_max - b.x_min) >= width * 0.92:
                    return [], "Candidate box covers almost the entire line width as one object"

        # 3. Check for excessive box overlaps across non-merged list
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                b1, b2 = merged[i], merged[j]
                inter_w = max(0.0, min(b1.x_max, b2.x_max) - max(b1.x_min, b2.x_min))
                inter_h = max(0.0, min(b1.y_max, b2.y_max) - max(b1.y_min, b2.y_min))
                inter_area = inter_w * inter_h
                union_area = (b1.area + b2.area) - inter_area
                iou = inter_area / max(1.0, union_area)
                if iou > self.max_overlap_iou:
                    return [], f"Excessive bounding box overlap detected (IoU={iou:.2f} > {self.max_overlap_iou:.2f})"

        return merged, None

    @staticmethod
    def visualize_line_decomposition(
        line_image: Image.Image,
        word_boxes: List[BoundingBox],
        labels: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        """Draws annotated word boxes over line image for visual engineering verification."""
        viz = line_image.copy().convert("RGB")
        draw = ImageDraw.Draw(viz)

        for i, box in enumerate(word_boxes):
            lbl = labels[i] if labels and i < len(labels) else f"word_{i+1}"
            x1, y1, x2, y2 = int(box.x_min), int(box.y_min), int(box.x_max), int(box.y_max)
            # Draw distinct cyan rectangle
            draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 255), width=2)
            # Draw label tag
            tag_text = f" {lbl} "
            tag_y = max(0, y1 - 12)
            draw.rectangle([x1, tag_y, x1 + len(tag_text) * 7, tag_y + 12], fill=(0, 100, 180))
            draw.text((x1, tag_y), tag_text, fill=(255, 255, 255))

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            viz.save(out_p)

        return viz
