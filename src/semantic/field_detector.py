"""Field Label and Anchor Detector for Karnataka Land Records.

Identifies canonical land record field label anchors in Kannada and English,
such as Survey Number (ಸರ್ವೆ ನಂ), Owner Name (ಖಾತೆದಾರರ ಹೆಸರು / ಮಾಲೀಕರ ಹೆಸರು),
Village (ಗ್ರಾಮ), Taluk (ತಾಲೂಕು), District (ಜಿಲ್ಲೆ), Khata Number (ಖಾತಾ ಸಂಖ್ಯೆ),
Extent/Area (ವಿಸ್ತೀರ್ಣ), Mutation Number (ಮ್ಯುಟೇಶನ್ ಸಂಖ್ಯೆ / ಎಂ.ಆರ್. ನಂ), etc.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

from schemas import BoundingBox

# Canonical Anchor Definitions with Kannada and English patterns
FIELD_LABEL_PATTERNS: Dict[str, List[str]] = {
    "survey_number": [
        r"ಸರ್ವೆ\s*ನಂ[ಬರ\.]*", r"ಸರ್ವೇ\s*ನಂ[ಬರ\.]*", r"ಸರ್ವೆ\s*ಸಂಖ್ಯೆ", r"ಸರ್ವೇ\s*ಸಂಖ್ಯೆ",
        r"ಸರ್ವೆ\s*ನಂಬರ್", r"ಸರ್ವೇ\s*ನಂಬರ್", r"ಸರ್ವೆ", r"ಸರ್ವೇ",
        r"survey\s*no[\.]*", r"survey\s*number", r"sy\s*no[\.]*",
    ],
    "owner_name": [
        r"ಖಾತೆದಾರರ\s*ಹೆಸರು", r"ಮಾಲೀಕರ\s*ಹೆಸರು", r"ಹಿಡುವಳಿದಾರರ\s*ಹೆಸರು", r"ಅನುಭೋಗದಾರರ\s*ಹೆಸರು",
        r"ಖಾತೆದಾರರು", r"ಮಾಲೀಕರು", r"ಹೆಸರು",
        r"khatadar\s*name", r"owner\s*name", r"occupant\s*name", r"name",
    ],
    "hissa_number": [
        r"ಹಿಸ್ಸಾ\s*ನಂ[ಬರ\.]*", r"ಹಿಸ್ಸೆ\s*ನಂ[ಬರ\.]*", r"ಹಿಸ್ಸಾ\s*ಸಂಖ್ಯೆ", r"ಹಿಸ್ಸಾ",
        r"hissa\s*no[\.]*", r"hissa",
    ],
    "khata_number": [
        r"ಖಾತಾ\s*ಸಂಖ್ಯೆ", r"ಖಾತೆ\s*ಸಂಖ್ಯೆ", r"ಖಾತಾ\s*ನಂ[ಬರ\.]*", r"ಖಾತೆ\s*ನಂ[ಬರ\.]*",
        r"ಖಾತಾ\b", r"ಖಾತೆ\b", r"khata\s*no[\.]*", r"khata\s*number", r"\bkhata\b",
    ],
    "father_name": [
        r"ತಂದೆಯ\s*ಹೆಸರು", r"ಗಂಡನ\s*ಹೆಸರು", r"ತಂದೆ", r"ಗಂಡ",
        r"father['\s]*s\s*name", r"husband['\s]*s\s*name", r"s/o", r"w/o", r"d/o",
    ],
    "district": [
        r"ಜಿಲ್ಲೆ", r"ಜಿಲ್ಲಾ", r"district", r"dist[\.]*",
    ],
    "taluk": [
        r"ತಾಲೂಕು", r"ತಾಲ್ಲೂಕು", r"ತಾಲೂಕ", r"taluk", r"taluka", r"tehsil",
    ],
    "hobli": [
        r"ಹೋಬಳಿ", r"ಹೋಬಲಿ", r"hobli", r"circle",
    ],
    "village": [
        r"ಗ್ರಾಮದ\s*ಹೆಸರು", r"ದಖಲೆ\s*ಗ್ರಾಮ", r"ಹಳ್ಳಿಯ\s*ಹೆಸರು", r"ಗ್ರಾಮ", r"village",
    ],
    "extent": [
        r"ಒಟ್ಟು\s*ವಿಸ್ತೀರ್ಣ", r"ಹಿಡುವಳಿ\s*ವಿಸ್ತೀರ್ಣ", r"ವಿಸ್ತೀರ್ಣ", r"ವಿಸ್ತಾರ",
        r"total\s*extent", r"extent", r"area",
    ],
    "land_type": [
        r"ಜಮೀನಿನ\s*ವಿವರ", r"ಭೂಮಿಯ\s*ವಿಧ", r"ಖುಷ್ಕಿ", r"ತರಿ", r"ಬಾಗಾಯ್ತು",
        r"land\s*nature", r"land\s*type", r"dry", r"wet", r"garden",
    ],
    "assessment": [
        r"ಕಂದಾಯ", r"ಆಕಾರಬಂದ್", r"ಕಂದಾಯದ\s*ವಿವರ", r"assessment", r"land\s*revenue",
    ],
    "mutation_number": [
        r"ಮ್ಯುಟೇಶನ್\s*ನಂ[ಬರ\.]*", r"ಮ್ಯುಟೇಶನ್\s*ಸಂಖ್ಯೆ", r"ಎಂ\s*ಆರ್\s*ನಂ", r"ಎಂ\.ಆರ್\.",
        r"mutation\s*no[\.]*", r"mr\s*no[\.]*",
    ],
    "document_number": [
        r"ದಾಖಲೆ\s*ಸಂಖ್ಯೆ", r"ನೋಂದಣಿ\s*ಸಂಖ್ಯೆ", r"ದಸ್ತಾವೇಜು\s*ಸಂಖ್ಯೆ",
        r"document\s*no[\.]*", r"registration\s*no[\.]*",
    ],
    "registration_date": [
        r"ದಿನಾಂಕ", r"ನೋಂದಣಿ\s*ದಿನಾಂಕ", r"ಆದೇಶದ\s*ದಿನಾಂಕ", r"date", r"registration\s*date",
    ],
    "boundaries": [
        r"ಚಕ್ಕುಬಂದಿ", r"ಚತುಸ್ಸೀಮೆ", r"ಗಡಿಗಳು", r"boundaries",
    ],
    "remarks": [
        r"ಷರಾ", r"ವಿಶೇಷ\s*ಷರಾ", r"ಟಿಪ್ಪಣಿ", r"remarks", r"notes",
    ],
}


@dataclass
class DetectedAnchor:
    """Represents a detected field label anchor on the document."""
    field_name: str
    matched_label: str
    region_id: str
    bbox: Optional[BoundingBox]
    text: str
    is_inline_key_value: bool = False
    inline_value: Optional[str] = None


class FieldDetector:
    """Detects field label anchors and inline key-value pairs from text regions."""

    def __init__(self):
        # Compile regular expressions for anchor matching
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for fname, raw_pats in FIELD_LABEL_PATTERNS.items():
            self.compiled_patterns[fname] = [
                re.compile(pat, re.IGNORECASE) for pat in raw_pats
            ]

    def detect_anchors(self, regions: List[Any]) -> List[DetectedAnchor]:
        """Scans document regions and detects field label anchors.

        Args:
            regions: List of RecognizedRegionResult or OCRRegion objects.

        Returns:
            List of DetectedAnchor instances with spatial coordinates and inline values if present.
        """
        anchors: List[DetectedAnchor] = []

        for reg in regions:
            text = getattr(reg, "normalized_text", None) or getattr(reg, "text", "")
            raw_text = getattr(reg, "raw_text", text)
            reg_id = getattr(reg, "region_id", "unknown_reg")
            bbox = getattr(reg, "bbox", None)

            if not text or not text.strip():
                continue

            # Check each field definition
            for field_name, patterns in self.compiled_patterns.items():
                for pat in patterns:
                    match = pat.search(text)
                    if match:
                        matched_str = match.group(0)
                        
                        # Check if text contains inline value (e.g. "ಸರ್ವೆ ನಂ: 125/1" or "ಹೆಸರು - ರಾಮಯ್ಯ")
                        inline_val = None
                        is_inline = False
                        
                        # Check delimiter after anchor
                        remainder = text[match.end():].strip()
                        # Strip separator colons, dashes, dots
                        sep_match = re.match(r"^[:\-\.=\s]+(.*)$", remainder)
                        if sep_match:
                            val_candidate = sep_match.group(1).strip()
                            if len(val_candidate) > 0:
                                is_inline = True
                                inline_val = val_candidate

                        anchors.append(
                            DetectedAnchor(
                                field_name=field_name,
                                matched_label=matched_str,
                                region_id=reg_id,
                                bbox=bbox,
                                text=raw_text,
                                is_inline_key_value=is_inline,
                                inline_value=inline_val,
                            )
                        )
                        break  # Match highest priority pattern for this field

        return anchors

    def detect_anchor(self, text: str) -> Optional[str]:
        """Convenience method to detect the canonical field name for a single text label."""
        if not text:
            return None
        for field_name, patterns in self.compiled_patterns.items():
            for pat in patterns:
                if pat.search(text):
                    return field_name
        return None


FieldAnchorDetector = FieldDetector
