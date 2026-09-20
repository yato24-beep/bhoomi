"""Controlled Non-Destructive OCR Normalization Stage.

Performs deterministic, conservative Unicode normalization:
- NFC canonical normalization
- Safe whitespace collapsing (strips leading/trailing, normalizes internal spaces)
- Canonical punctuation normalization (e.g. smart quotes, em-dashes to hyphens)
- Optional Indic numeral mapping to standard ASCII digits

STRICT POLICY:
- Strictly prohibits speculative dictionary spell-correction
- Never mutates ambiguous characters
- Preserves both raw_ocr_text and normalized_ocr_text
"""

from dataclasses import dataclass
import re
import unicodedata
from typing import Dict, Optional, Tuple

KANNADA_DIGITS: Dict[str, str] = {
    '೦': '0', '೧': '1', '೨': '2', '೩': '3', '೪': '4',
    '೫': '5', '೬': '6', '೭': '7', '೮': '8', '೯': '9',
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}


@dataclass
class NormalizedTextOutput:
    """Pair of raw and normalized text representations."""
    raw_ocr_text: str
    normalized_ocr_text: str
    has_changes: bool

    @property
    def raw_text(self) -> str:
        return self.raw_ocr_text

    @property
    def normalized_text(self) -> str:
        return self.normalized_ocr_text


class SemanticNormalizer:
    """Controlled OCR normalizer preserving raw evidence."""

    def __init__(self, convert_digits: bool = True):
        self.convert_digits = convert_digits

    def normalize(self, text: str) -> NormalizedTextOutput:
        """Applies conservative NFC and punctuation normalization.

        Args:
            text: Raw OCR string.

        Returns:
            NormalizedTextOutput containing verbatim raw and safely normalized strings.
        """
        if not text:
            return NormalizedTextOutput(raw_ocr_text="", normalized_ocr_text="", has_changes=False)

        raw = text

        # 1. Unicode NFC normalization (composes base characters and combining marks)
        norm = unicodedata.normalize("NFC", raw)

        # 2. Punctuation normalization (unambiguous forms only)
        norm = norm.replace("“", '"').replace("”", '"')
        norm = norm.replace("‘", "'").replace("’", "'")
        norm = norm.replace("—", "-").replace("–", "-")
        norm = norm.replace("…", "...")

        # 3. Safe whitespace normalization
        norm = re.sub(r"[\t\r\f\v]+", " ", norm)
        norm = re.sub(r" +", " ", norm)
        norm = norm.strip()

        # 4. Optional digit normalization
        if self.convert_digits:
            norm = "".join(KANNADA_DIGITS.get(ch, ch) for ch in norm)

        return NormalizedTextOutput(
            raw_ocr_text=raw,
            normalized_ocr_text=norm,
            has_changes=(raw != norm),
        )


def normalize_kannada_numerals(text: str) -> str:
    """Converts Kannada and Devanagari numerals to standard ASCII Hindu-Arabic digits."""
    if not text:
        return ""
    return "".join(KANNADA_DIGITS.get(ch, ch) for ch in text)


ControlledOCRNormalizer = SemanticNormalizer
