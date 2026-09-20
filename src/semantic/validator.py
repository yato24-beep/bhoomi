"""Field-Specific Deterministic Validators for Land Records.

Validates extracted semantic field values according to Karnataka revenue formats:
- Survey Number: Cadastral patterns (e.g. 125, 125/1, 125/1A, 125/P1, 125/1+2)
- Khata Number: Positive numeric or alphanumeric registration identifiers
- Date: Real calendar dates (DD/MM/YYYY, YYYY-MM-DD), rejecting impossible calendar dates (e.g. month > 12, day > 31)
- Extent: Standard Karnataka land area structures (Acre-Gunta-Anna, 1-15, 0-04, hectares)
- Localities / Names: Syntax and structure verification without aggressive dictionary mutation

CRITICAL MANDATE:
Validation MUST NOT silently change OCR text. It returns:
(value, validation_status, validation_reason, source_region_ids)
"""

from datetime import datetime
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.semantic.schema import SemanticFieldItem, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)

# Cadastral Survey Number Regex: e.g. 125, 125/1, 125/1A, 125/P1, 125/1+2, 125-1
CADASTRAL_SURVEY_REGEX = re.compile(
    r"^\s*([1-9]\d{0,4}(?:[/_\-\.][0-9A-Za-z\+\-_]+)*)\s*$"
)

# Khata Number Regex: e.g. 45, 102, K-12, 102/A
KHATA_REGEX = re.compile(
    r"^\s*([1-9]\d{0,5}[A-Za-z]?|[A-Za-z]{1,3}[-_/]?\d{1,5})\s*$"
)

# Extent Land Area Regex: e.g. "1-20" (1 acre 20 guntas), "0-04", "2.5", "1-20-08"
EXTENT_REGEX = re.compile(
    r"^\s*(\d{1,4}[-\s\.:]\d{1,2}(?:[-\s\.:]\d{1,2})?|\d{1,4}(?:\.\d{1,4})?|\d{1,4}\s*(?:ಎಕರೆ|acre[s]?)\s*\d{1,2}\s*(?:ಗುಂಟೆ|gunta[s]?))\s*(?:ಎಕರೆ|ಗುಂಟೆ|acre[s]?|gunta[s]?|hec|ha)?\s*$",
    re.IGNORECASE,
)

# Supported Date Patterns
DATE_PATTERNS = [
    (re.compile(r"^\s*(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})\s*$"), ("day", "month", "year")),
    (re.compile(r"^\s*(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})\s*$"), ("year", "month", "day")),
]


class FieldValidator:
    """Validates semantic fields against domain revenue rules without mutating OCR evidence."""

    @staticmethod
    def validate_survey_number(raw_val: str) -> Tuple[ValidationStatus, Optional[str]]:
        """Validates cadastral survey number formats."""
        if not raw_val or not raw_val.strip():
            return ValidationStatus.INVALID, "Survey number value is empty"

        cleaned = raw_val.strip()
        if CADASTRAL_SURVEY_REGEX.match(cleaned):
            return ValidationStatus.VALID, "Valid cadastral survey number format"

        # Check if digits exist at all
        if not re.search(r"\d", cleaned):
            return ValidationStatus.INVALID, "Survey number contains no numeric digits"

        return ValidationStatus.WARNING, "Non-standard cadastral survey number format; officer inspection recommended"

    @staticmethod
    def validate_khata_number(raw_val: str) -> Tuple[ValidationStatus, Optional[str]]:
        """Validates khata account number formats."""
        if not raw_val or not raw_val.strip():
            return ValidationStatus.INVALID, "Khata number value is empty"

        cleaned = raw_val.strip()
        if KHATA_REGEX.match(cleaned):
            return ValidationStatus.VALID, "Valid khata number format"

        if re.search(r"\d", cleaned):
            return ValidationStatus.WARNING, "Non-standard khata number format"

        return ValidationStatus.INVALID, "Khata number contains no digits"

    @staticmethod
    def validate_date(raw_val: str) -> Tuple[ValidationStatus, Optional[str]]:
        """Validates real calendar dates and rejects impossible dates."""
        if not raw_val or not raw_val.strip():
            return ValidationStatus.INVALID, "Date value is empty"

        cleaned = raw_val.strip()
        for pat, order in DATE_PATTERNS:
            match = pat.match(cleaned)
            if match:
                g1, g2, g3 = int(match.group(1)), int(match.group(2)), int(match.group(3))
                if order == ("day", "month", "year"):
                    d, m, y = g1, g2, g3
                else:
                    y, m, d = g1, g2, g3

                # Check year bounds
                if not (1800 <= y <= 2100):
                    return ValidationStatus.INVALID, f"Date year {y} is outside plausible range [1800, 2100]"

                # Check month bounds
                if not (1 <= m <= 12):
                    return ValidationStatus.INVALID, f"Impossible calendar month: {m} (must be 1-12)"

                # Check actual calendar day valid in calendar
                try:
                    datetime(year=y, month=m, day=d)
                    return ValidationStatus.VALID, f"Valid calendar date ({y:04d}-{m:02d}-{d:02d})"
                except ValueError as e:
                    return ValidationStatus.INVALID, f"Impossible calendar date: {e}"

        return ValidationStatus.WARNING, "Unrecognized date format; manual verification required"

    @staticmethod
    def validate_extent(raw_val: str) -> Tuple[ValidationStatus, Optional[str]]:
        """Validates land measure extent (Acre-Gunta-Anna)."""
        if not raw_val or not raw_val.strip():
            return ValidationStatus.INVALID, "Extent value is empty"

        cleaned = raw_val.strip()
        if EXTENT_REGEX.match(cleaned):
            # Check gunta bounds if in acre-gunta format (e.g. 1-25)
            sub_parts = re.split(r"[-\s\.:]", cleaned)
            if len(sub_parts) >= 2:
                try:
                    guntas = int(sub_parts[1])
                    if guntas >= 40:
                        return ValidationStatus.WARNING, f"Gunta value {guntas} >= 40 (1 acre = 40 guntas)"
                except ValueError:
                    pass
            return ValidationStatus.VALID, "Valid land measurement extent format"

        return ValidationStatus.WARNING, "Non-standard extent measurement format"

    @staticmethod
    def validate_name_or_locality(raw_val: str, field_type: str = "name") -> Tuple[ValidationStatus, Optional[str]]:
        """Validates names or administrative divisions without destructive dictionary overwrites."""
        if not raw_val or not raw_val.strip():
            return ValidationStatus.INVALID, f"{field_type.replace('_', ' ').title()} is empty"

        cleaned = raw_val.strip()
        # Must have at least 2 characters and not consist purely of punctuation or symbols
        letters = [ch for ch in cleaned if ch.isalpha()]
        if len(letters) < 2:
            return ValidationStatus.INVALID, f"{field_type.replace('_', ' ').title()} contains insufficient alphabetic characters"

        return ValidationStatus.VALID, f"{field_type.replace('_', ' ').title()} format plausible"

    @staticmethod
    def validate_hissa_number(raw_val: str) -> Tuple[ValidationStatus, Optional[str]]:
        """Validates cadastral hissa sub-division number formats."""
        if not raw_val or not raw_val.strip():
            return ValidationStatus.INVALID, "Hissa number value is empty"

        cleaned = raw_val.strip()
        if re.match(r"^\s*([0-9A-Za-z\+\-_/]+)\s*$", cleaned):
            return ValidationStatus.VALID, "Valid hissa number format"
        return ValidationStatus.WARNING, "Non-standard hissa number format"

    @staticmethod
    def validate_mutation_number(raw_val: str) -> Tuple[ValidationStatus, Optional[str]]:
        """Validates mutation register reference (MR number)."""
        if not raw_val or not raw_val.strip():
            return ValidationStatus.INVALID, "Mutation number value is empty"

        cleaned = raw_val.strip()
        if re.search(r"\d+", cleaned):
            return ValidationStatus.VALID, "Valid mutation record identifier"
        return ValidationStatus.WARNING, "Mutation number lacks numeric digits"

    @staticmethod
    def verify_ocr_grounding(
        field_item: SemanticFieldItem,
        ocr_texts: Sequence[str],
    ) -> Tuple[bool, Optional[str]]:
        """Strictly verifies that the raw extracted value exists in recognized OCR tokens.

        CRITICAL SAFETY RULE: Prohibits hallucinations. If the model generated text
        that is absent from the OCR evidence, flags as ungrounded/hallucinated.
        """
        raw_val = field_item.raw_value.strip() if field_item.raw_value else ""
        if not raw_val:
            return False, "Value is empty"

        # Check in field's own provenance first
        if field_item.provenance and field_item.provenance.raw_ocr_text:
            prov_text = field_item.provenance.raw_ocr_text.strip()
            if raw_val in prov_text:
                return True, "Exact substring match in provenance OCR text"

        # Check across all recognized OCR texts
        for text in ocr_texts:
            if not text:
                continue
            if raw_val in text:
                return True, "Value found in OCR line evidence"

        # Token-based subset check (e.g. multi-word names or extent tokens)
        val_tokens = set(raw_val.split())
        all_ocr_tokens = set(" ".join(ocr_texts).split())
        if val_tokens and val_tokens.issubset(all_ocr_tokens):
            return True, "All tokens found in OCR vocabulary"

        return False, f"Value '{raw_val}' not found in OCR evidence (potential hallucination)"

    def validate_field(
        self,
        field_item: Union[SemanticFieldItem, str],
        value: Optional[str] = None,
        ocr_texts: Optional[Sequence[str]] = None,
    ) -> SemanticFieldItem:
        """Validates an extracted field and populates validation status and reason without mutating text."""
        if isinstance(field_item, str):
            val_str = value or ""
            target_item = SemanticFieldItem(
                field_name=field_item,
                value=val_str,
                raw_value=val_str,
            )
        else:
            target_item = field_item

        fname = target_item.field_name
        val = target_item.value

        # Field-specific deterministic syntax validation
        if fname == "survey_number":
            status, reason = self.validate_survey_number(val)
        elif fname == "hissa_number":
            status, reason = self.validate_hissa_number(val)
        elif fname == "khata_number":
            status, reason = self.validate_khata_number(val)
        elif fname in ("date", "registration_date", "document_date", "record_date"):
            status, reason = self.validate_date(val)
        elif fname == "extent":
            status, reason = self.validate_extent(val)
        elif fname in ("owner_name", "cultivator_name", "father_name"):
            status, reason = self.validate_name_or_locality(val, fname)
        elif fname in ("village", "hobli", "taluk", "district"):
            status, reason = self.validate_name_or_locality(val, fname)
        elif fname in ("mutation_number", "registration_number", "document_number"):
            status, reason = self.validate_mutation_number(val)
        else:
            status = ValidationStatus.VALID if val and len(val.strip()) > 0 else ValidationStatus.INVALID
            reason = "Non-empty string check passed" if status == ValidationStatus.VALID else "Field is empty"

        # Check evidence grounding if OCR texts are provided
        if ocr_texts is not None and status == ValidationStatus.VALID:
            is_grounded, ground_reason = self.verify_ocr_grounding(target_item, ocr_texts)
            if not is_grounded:
                status = ValidationStatus.INVALID
                reason = ground_reason

        # Provenance requirement check: populated fields must have valid source region
        if status == ValidationStatus.VALID and target_item.provenance:
            if not target_item.provenance.region_id or not target_item.provenance.region_id.strip():
                status = ValidationStatus.WARNING
                reason = "Provenance missing valid region_id"

        target_item.validation_status = status
        target_item.validation_reason = reason
        return target_item

