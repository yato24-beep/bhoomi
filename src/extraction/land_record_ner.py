"""Gazetteer & Label-Anchor Structured Field Extractor for Karnataka Land Records.

Extracts structured revenue/cadastral fields (Owner Name, Survey Number, Khata Number,
Taluk, Village, Hobli, District, Area/Extent) from line-segmented OCR text using
domain gazetteers, anchor token matching, and per-field confidence scoring.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Indic numerals conversion table
INDIC_DIGITS = {
    '೦': '0', '೧': '1', '೨': '2', '೩': '3', '೪': '4',
    '೫': '5', '೬': '6', '೭': '7', '೮': '8', '೯': '9',
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}


def normalize_digits(text: str) -> str:
    """Normalizes Kannada/Indic numerals to standard ASCII digits."""
    return "".join(INDIC_DIGITS.get(c, c) for c in text)


# Canonical RTC/Bhoomi Field Label Gazetteer Anchors
FIELD_ANCHORS: Dict[str, List[str]] = {
    "owner_name": [
        "ಮಾಲೀಕರ ಹೆಸರು", "ಖಾತೆದಾರರ ಹೆಸರು", "ಖಾತೆದಾರರ ವಿವರ", "ಹಿಡುವಳಿದಾರರ ಹೆಸರು",
        "ಖಾತೆದಾರರು", "ಖಾತೆದಾರ", "ಮಾಲೀಕರು", "ಹೆಸರು", "owner name", "khatadar name",
    ],
    "survey_number": [
        "ಸರ್ವೆ ನಂಬರ್", "ಸರ್ವೆ ನಂ", "ಸರ್ವೇ ನಂಬರ್", "ಸರ್ವೇ ನಂ", "ಸರ್ವೆ", "ಸರ್ವೇ",
        "ಹಿಸ್ಸಾ ನಂ", "ಹಿಸ್ಸಾ", "survey number", "survey no",
    ],
    "khata_number": [
        "ಖಾತಾ ಸಂಖ್ಯೆ", "ಖಾತೆ ಸಂಖ್ಯೆ", "ಖಾತಾ ನಂ", "ಖಾತೆ ನಂ", "ಖಾತಾ", "ಖಾತೆ",
        "khata number", "khata no",
    ],
    "taluk": [
        "ತಾಲೂಕು", "ತಾಲ್ಲೂಕು", "ತಾಲೂಕ", "taluk", "taluka", "tehsil",
    ],
    "village": [
        "ಗ್ರಾಮದ ಹೆಸರು", "ದಖಲೆ ಗ್ರಾಮ", "ಹಳ್ಳಿಯ ಹೆಸರು", "ಗ್ರಾಮ", "village",
    ],
    "hobli": [
        "ಹೋಬಳಿ", "ಹೋಬಲಿ", "hobli",
    ],
    "district": [
        "ಜಿಲ್ಲೆ", "ಜಿಲ್ಲಾ", "district",
    ],
    "extent_area": [
        "ವಿಸ್ತೀರ್ಣ", "ವಿಸ್ತಾರ", "ಒಟ್ಟು ವಿಸ್ತೀರ್ಣ", "extent", "area", "ಹಿಡುವಳಿ ವಿಸ್ತೀರ್ಣ",
    ],
}


@dataclass
class ExtractedRecordField:
    """Represents a single structured field extracted from the land record."""
    field_name: str
    raw_value: str
    normalized_value: str
    confidence: float
    anchor_matched: Optional[str] = None
    line_index: int = 1
    requires_human_review: bool = False
    evidence_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "confidence": round(self.confidence, 4),
            "anchor_matched": self.anchor_matched,
            "line_index": self.line_index,
            "requires_human_review": self.requires_human_review,
            "evidence_text": self.evidence_text,
        }


class LandRecordFieldExtractor:
    """Gazetteer and label-anchor NLU/NER extractor for land records."""

    def __init__(self, confidence_threshold: float = 0.60):
        self.confidence_threshold = confidence_threshold

    def extract_fields(
        self,
        lines: List[str],
        line_confidences: Optional[List[float]] = None,
    ) -> Dict[str, ExtractedRecordField]:
        """Extracts all recognized canonical fields from spaced document lines.

        Args:
            lines: List of clean, spaced text lines in document reading order.
            line_confidences: Optional per-line OCR confidence scores.

        Returns:
            Dictionary mapping field_name -> ExtractedRecordField.
        """
        extracted: Dict[str, ExtractedRecordField] = {}
        if not lines:
            return extracted

        num_lines = len(lines)
        has_ocr_conf = bool(line_confidences and len(line_confidences) == num_lines)

        # Track which fields have been successfully extracted
        extracted_fields = set()

        for field_name, anchors in FIELD_ANCHORS.items():
            best_match: Optional[ExtractedRecordField] = None

            for idx, line in enumerate(lines):
                line_clean = line.strip()
                if not line_clean:
                    continue

                for anchor in anchors:
                    # Check if anchor is in line
                    if anchor in line_clean:
                        # Extract the adjacent value span
                        val_span, anchor_found = self._extract_value_after_anchor(line_clean, anchor)

                        # If value span is empty, check if value is on the immediate subsequent line
                        if not val_span and (idx + 1) < num_lines:
                            next_line = lines[idx + 1].strip()
                            # Ensure next line does not contain an anchor itself
                            if not any(a in next_line for a_list in FIELD_ANCHORS.values() for a in a_list):
                                val_span = next_line

                        if val_span:
                            norm_val = self._normalize_field_value(field_name, val_span)
                            match_conf = 0.95 if len(anchor) > 4 else 0.85
                            if has_ocr_conf:
                                ocr_conf = line_confidences[idx]
                                composite_conf = min(0.99, (ocr_conf * 0.6) + (match_conf * 0.4))
                            else:
                                composite_conf = match_conf
                            needs_review = composite_conf < self.confidence_threshold or len(norm_val) < 2

                            candidate = ExtractedRecordField(
                                field_name=field_name,
                                raw_value=val_span,
                                normalized_value=norm_val,
                                confidence=composite_conf,
                                anchor_matched=anchor_found,
                                line_index=idx + 1,
                                requires_human_review=needs_review,
                                evidence_text=line_clean,
                            )

                            if best_match is None or candidate.confidence > best_match.confidence:
                                best_match = candidate
                            break

                if best_match is not None and not best_match.requires_human_review:
                    break

            if best_match is not None:
                extracted[field_name] = best_match

        return extracted

    def _extract_value_after_anchor(self, line: str, anchor: str) -> Tuple[str, str]:
        """Extracts the value token string immediately following an anchor in a line."""
        idx = line.find(anchor)
        if idx == -1:
            return "", ""

        remainder = line[idx + len(anchor):].strip()
        # Strip common punctuation delimiters
        remainder = re.sub(r"^[\s:;=\-—\.]+", "", remainder).strip()
        # If remainder contains another distinct anchor at word boundary, truncate at that anchor
        for a_list in FIELD_ANCHORS.values():
            for a in a_list:
                if a == anchor:
                    continue
                # Require preceding space or boundary delimiter before subsequent anchor
                match = re.search(r"(?:\s+|^)" + re.escape(a) + r"[\s:;=\-—\.]", remainder)
                if match and match.start() > 0:
                    remainder = remainder[:match.start()].strip()

        return remainder, anchor

    def _normalize_field_value(self, field_name: str, raw_val: str) -> str:
        """Normalizes field value based on domain constraints."""
        val = raw_val.strip()
        if field_name in ("survey_number", "khata_number"):
            val = normalize_digits(val)
            # Retain only digits and slashes/hyphens for numbers
            val = re.sub(r"[^\d/\-]", " ", val).strip()
            val = re.sub(r"\s+", "", val)
        elif field_name == "extent_area":
            val = normalize_digits(val)
        return val
