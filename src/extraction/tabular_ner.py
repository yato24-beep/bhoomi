"""Layout-Aware 2D Tabular Field Extractor for Land Record Forms.

Extracts structured fields from RTC/Pahani/7-12 forms where field labels and
their values occupy different rows or columns (2D spatial relationship).

Architecture:
  1. Accept EasyOCR raw output (list of (poly, text, conf) tuples with bboxes)
  2. Cluster tokens into rows by y-coordinate
  3. Cluster rows into columns by x-coordinate (column detection)
  4. Match field-label anchors to the nearest value token spatially
     (downward or rightward proximity — not line sequence)
  5. Report OCR typos as-is; do not correct them

This is intentionally separate from LandRecordFieldExtractor (1D scanner).
Both can be run and their outputs merged.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fuzzy anchor matching vocabulary
# Maps normalized anchor variations (including OCR typos) to canonical fields
# ---------------------------------------------------------------------------
TABULAR_ANCHORS: Dict[str, List[str]] = {
    "survey_number": [
        # Kannada
        "ಸರ್ವೆ ನಂಬರ್", "ಸರ್ವೆ ನಂ", "ಸರ್ವೇ ನಂ", "ಸರ್ವೆ",
        # English (clean + common OCR typos)
        "survey number", "survey no", "survey", "surv", "svy no",
        "surveyno", "srv no",
    ],
    "hissa_number": [
        "ಹಿಸ್ಸಾ ನಂ", "ಹಿಸ್ಸಾ",
        "hissa", "hissd", "hissa no", "hissano",
    ],
    "owner_name": [
        "ಮಾಲೀಕರ ಹೆಸರು", "ಖಾತೆದಾರರ ಹೆಸರು", "ಖಾತೆದಾರ",
        "khatadar name", "khatadar", "khuthedur", "khuthedur nme",
        "owner name", "owner", "occupant", "occupunt",
        "name", "ryot name", "ryot",
    ],
    "extent_area": [
        "ವಿಸ್ತೀರ್ಣ", "ಒಟ್ಟು ವಿಸ್ತೀರ್ಣ",
        "total extent", "extent", "area", "ared", "tutdl extent",
        "area (hec)", "area hec", "ared (hec)",
    ],
    "village": [
        "ಗ್ರಾಮ", "ಗ್ರಾಮದ ಹೆಸರು",
        "village", "villgge", "vill", "grama",
    ],
    "taluk": [
        "ತಾಲೂಕು", "ತಾಲ್ಲೂಕು",
        "taluk", "taluka", "tqluk", "tehsil",
    ],
    "district": [
        "ಜಿಲ್ಲೆ", "ಜಿಲ್ಲಾ",
        "district", "distt", "dist",
    ],
    "khata_number": [
        "ಖಾತಾ ಸಂಖ್ಯೆ", "ಖಾತೆ ನಂ",
        "khata number", "khata no", "khata",
    ],
    "assessment": [
        "ಕಂದಾಯ",
        "assessment", "land revenue", "revenue",
    ],
}


@dataclass
class TabularToken:
    """A single OCR-detected token with its spatial bounding box."""
    text: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def cx(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def cy(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass
class TabularField:
    """A structured field extracted via 2D layout analysis."""
    field_name: str
    anchor_text: str           # The verbatim OCR text that matched the anchor
    anchor_normalized: str     # Normalized anchor key matched
    value_text: str            # Verbatim OCR value (NOT corrected)
    anchor_bbox: Tuple[float, float, float, float]   # x_min, y_min, x_max, y_max
    value_bbox: Tuple[float, float, float, float]
    confidence: float
    spatial_relationship: str  # "below", "right", "inline"
    requires_review: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "anchor_text": self.anchor_text,
            "anchor_normalized": self.anchor_normalized,
            "value_text": self.value_text,
            "anchor_bbox": self.anchor_bbox,
            "value_bbox": self.value_bbox,
            "confidence": round(self.confidence, 4),
            "spatial_relationship": self.spatial_relationship,
            "requires_review": self.requires_review,
        }


class TabularLayoutExtractor:
    """Layout-aware 2D field extractor for tabular land record documents.

    Input: raw EasyOCR readtext() output — list of (poly, text, conf) tuples.
    Output: dict mapping field_name -> TabularField.
    """

    def __init__(
        self,
        row_tolerance_px: float = 12.0,
        col_tolerance_px: float = 30.0,
        value_search_radius_y: float = 120.0,
        value_search_radius_x: float = 200.0,
        confidence_threshold: float = 0.25,
    ):
        self.row_tolerance_px = row_tolerance_px
        self.col_tolerance_px = col_tolerance_px
        self.value_search_radius_y = value_search_radius_y
        self.value_search_radius_x = value_search_radius_x
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def extract_from_easyocr_result(
        self,
        raw_easyocr: List[Tuple[List, str, float]],
    ) -> Dict[str, TabularField]:
        """Extract structured fields from raw EasyOCR readtext() output."""
        tokens = self._parse_tokens(raw_easyocr)
        return self.extract_from_tokens(tokens)

    def extract_from_regions(
        self,
        regions: Sequence[Any],
    ) -> Dict[str, TabularField]:
        """Extract structured fields from RecognizedRegionResult or OCRResult sequence."""
        tokens: List[TabularToken] = []
        for r in regions:
            text = getattr(r, "normalized_text", None) or getattr(r, "raw_text", None) or getattr(r, "text", "")
            text = str(text).strip()
            if not text:
                continue
            conf = getattr(r, "confidence", None)
            conf_val = float(conf) if conf is not None else 1.0
            bbox = getattr(r, "bbox", None)
            if bbox is not None:
                x_min = float(getattr(bbox, "x_min", 0.0))
                y_min = float(getattr(bbox, "y_min", 0.0))
                x_max = float(getattr(bbox, "x_max", 0.0))
                y_max = float(getattr(bbox, "y_max", 0.0))
            else:
                x_min, y_min, x_max, y_max = 0.0, 0.0, 0.0, 0.0
            tokens.append(TabularToken(
                text=text,
                confidence=conf_val,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            ))
        return self.extract_from_tokens(tokens)

    def extract_from_tokens(
        self,
        tokens: List[TabularToken],
    ) -> Dict[str, TabularField]:
        """Extract structured fields from parsed TabularToken objects using 2D layout analysis."""
        if not tokens:
            return {}

        rows = self._cluster_rows(tokens)
        extracted: Dict[str, TabularField] = {}

        for token in tokens:
            anchor_field, anchor_key = self._match_anchor(token.text)
            if anchor_field is None:
                continue
            if anchor_field in extracted:
                continue  # Take first / best match

            # Priority 1: Inline value after colon/separator (e.g. "Village: Shirur")
            inline = self._extract_inline_value(token.text, anchor_key)
            if inline:
                extracted[anchor_field] = TabularField(
                    field_name=anchor_field,
                    anchor_text=token.text,
                    anchor_normalized=anchor_key,
                    value_text=inline,
                    anchor_bbox=(token.x_min, token.y_min, token.x_max, token.y_max),
                    value_bbox=(token.x_min, token.y_min, token.x_max, token.y_max),
                    confidence=token.confidence,
                    spatial_relationship="inline",
                    requires_review=token.confidence < self.confidence_threshold,
                )
                continue

            # Priority 2: Nearest token spatially below the anchor
            value_token = self._find_value_below(token, tokens, rows)
            relationship = "below"
            if value_token is None:
                # Priority 3: Nearest token to the right on the same row
                value_token = self._find_value_right(token, tokens)
                relationship = "right"

            if value_token is not None:
                extracted[anchor_field] = TabularField(
                    field_name=anchor_field,
                    anchor_text=token.text,
                    anchor_normalized=anchor_key,
                    value_text=value_token.text,
                    anchor_bbox=(token.x_min, token.y_min, token.x_max, token.y_max),
                    value_bbox=(value_token.x_min, value_token.y_min, value_token.x_max, value_token.y_max),
                    confidence=min(token.confidence, value_token.confidence),
                    spatial_relationship=relationship,
                    requires_review=(
                        min(token.confidence, value_token.confidence) < self.confidence_threshold
                    ),
                )

        return extracted

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _parse_tokens(self, raw: List[Tuple]) -> List[TabularToken]:
        tokens = []
        for item in raw:
            if len(item) < 3:
                continue
            poly, text, conf = item[0], item[1], item[2]
            text = str(text).strip()
            if not text:
                continue
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            tokens.append(TabularToken(
                text=text,
                confidence=float(conf),
                x_min=float(min(xs)),
                y_min=float(min(ys)),
                x_max=float(max(xs)),
                y_max=float(max(ys)),
            ))
        return tokens

    def _cluster_rows(self, tokens: List[TabularToken]) -> List[List[TabularToken]]:
        """Groups tokens into horizontal rows by y-center proximity."""
        if not tokens:
            return []
        sorted_tokens = sorted(tokens, key=lambda t: t.cy)
        rows: List[List[TabularToken]] = []
        current_row = [sorted_tokens[0]]
        current_cy = sorted_tokens[0].cy

        for tok in sorted_tokens[1:]:
            if abs(tok.cy - current_cy) <= self.row_tolerance_px:
                current_row.append(tok)
            else:
                rows.append(sorted(current_row, key=lambda t: t.cx))
                current_row = [tok]
                current_cy = tok.cy

        if current_row:
            rows.append(sorted(current_row, key=lambda t: t.cx))
        return rows

    def _match_anchor(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Returns (field_name, matched_anchor) or (None, None)."""
        text_lower = text.lower().strip()
        text_norm = re.sub(r"[:\s]+", " ", text_lower).strip()

        for field_name, anchors in TABULAR_ANCHORS.items():
            for anchor in anchors:
                anchor_lower = anchor.lower()
                # Exact or starts-with match
                if text_norm == anchor_lower or text_norm.startswith(anchor_lower):
                    return field_name, anchor
                # Substring match for long anchors (>5 chars)
                if len(anchor_lower) > 5 and anchor_lower in text_norm:
                    return field_name, anchor
        return None, None

    def _find_value_below(
        self,
        anchor: TabularToken,
        all_tokens: List[TabularToken],
        rows: List[List[TabularToken]],
    ) -> Optional[TabularToken]:
        """Find the nearest non-anchor token spatially below the anchor."""
        candidates = []
        for tok in all_tokens:
            if tok is anchor:
                continue
            # Must be below anchor
            dy = tok.y_min - anchor.y_max
            if dy < 0 or dy > self.value_search_radius_y:
                continue
            # X center must overlap with anchor's x range (±tolerance)
            x_overlap = (
                tok.cx >= anchor.x_min - self.col_tolerance_px and
                tok.cx <= anchor.x_max + self.col_tolerance_px
            )
            if not x_overlap:
                continue
            # Must not itself be an anchor
            if self._match_anchor(tok.text)[0] is not None:
                continue
            candidates.append((dy, tok))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _find_value_right(
        self,
        anchor: TabularToken,
        all_tokens: List[TabularToken],
    ) -> Optional[TabularToken]:
        """Find the nearest non-anchor token to the right of the anchor on the same row."""
        candidates = []
        for tok in all_tokens:
            if tok is anchor:
                continue
            # Must be on approximately same row
            dy = abs(tok.cy - anchor.cy)
            if dy > self.row_tolerance_px * 2:
                continue
            # Must be to the right
            dx = tok.x_min - anchor.x_max
            if dx < 0 or dx > self.value_search_radius_x:
                continue
            if self._match_anchor(tok.text)[0] is not None:
                continue
            candidates.append((dx, tok))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _extract_inline_value(self, text: str, anchor_key: str) -> Optional[str]:
        """Extracts value after ':' or '=' on the same token text."""
        for sep in [":", "=", "-"]:
            if sep in text:
                parts = text.split(sep, 1)
                val = parts[1].strip()
                if val and len(val) > 0:
                    return val
        return None
