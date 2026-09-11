"""
src/extraction/extractor.py
State-configuration driven regex and layout-based structured field extractor.
Preserves raw values, exact bounding boxes, pages, and provenance evidence.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from schemas import (
    BoundingBox,
    DocumentOCRResult,
    ExtractedField,
    ExtractionMethod,
    FieldEvidence,
    HandwritingResult,
    TableStructure,
    ValidationStatus,
)


# Translation map for Indic numerals (Devanagari, Kannada, Tamil, Gujarati, Telugu, Bengali)
INDIC_DIGITS = {
    # Devanagari (Hindi, Marathi, Nepali)
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
    # Kannada
    '೦': '0', '೧': '1', '೨': '2', '೩': '3', '೪': '4',
    '೫': '5', '೬': '6', '೭': '7', '೮': '8', '೯': '9',
    # Tamil
    '௦': '0', '௧': '1', '௨': '2', '௩': '3', '௪': '4',
    '௫': '5', '௬': '6', '௭': '7', '௮': '8', '௯': '9',
    # Telugu
    '౦': '0', '౧': '1', '౨': '2', '౩': '3', '౪': '4',
    '౫': '5', '౬': '6', '౭': '7', '౮': '8', '౯': '9',
    # Bengali
    '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
    '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9',
}


def convert_indic_numerals(text: str) -> str:
    """Converts Devanagari, Kannada, Tamil, Telugu, Bengali numerals to standard digits (0-9)."""
    return "".join(INDIC_DIGITS.get(ch, ch) for ch in text)


# Backwards compatible alias
convert_devanagari_numerals = convert_indic_numerals


def clean_ocr_text(text: str) -> str:
    """
    Normalizes whitespace and Unicode representation without modifying
    visible content.

    FIX 1: Tesseract's Tamil (and some other Indic) language models insert
    invisible zero-width joiner/non-joiner characters (U+200C ZWNJ,
    U+200D ZWJ) after many words — e.g. "மாவட்டம்\u200c:" instead of
    "மாவட்டம்:". These are invisible when printed/viewed but sit right
    where extraction regex patterns expect a label to end, breaking
    nearly every field match while looking completely normal in print
    statements. Stripping them here fixes extraction without touching
    any visible text.

    FIX 2: EasyOCR's (and sometimes Tesseract's) Devanagari/Marathi/Hindi
    models systematically misread the ASCII colon ":" immediately after
    a label word as the Devanagari visarga character "ः" (U+0903) —
    e.g. "जिल्हाः पुणे" instead of "जिल्हा: पुणे". Left alone, this both
    breaks label/value regex matching (khasra_number pattern requires a
    literal ":" to skip past) and corrupts extracted values with a
    stray leading "ः" character (owner_name, village, tehsil, district
    all showed this in testing). We conservatively convert "ः" back to
    ":" ONLY when it's immediately followed by whitespace or a digit —
    i.e. acting as a label terminator — since genuine word-internal
    visarga is virtually always followed by more letters, not a space
    or digit. This is a documented, tested OCR-artifact correction, not
    a guess.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")  # ZWNJ, ZWJ
    text = text.replace("\ufeff", "")  # BOM, occasionally injected too
    text = re.sub(r"ः(?=\s|\d)", ":", text)  # misread colon -> real colon
    return re.sub(r"[ \t]+", " ", text).strip()


class FieldExtractor:
    """
    Extracts structured land record fields from OCR text and tables based on state configuration.
    """

    def __init__(self, state_config: Dict[str, Any]):
        self.config = state_config
        self.fields_def = state_config.get("fields", {})

    def extract_all(
        self,
        ocr_result: DocumentOCRResult,
        handwriting_result: Optional[HandwritingResult] = None
    ) -> Dict[str, ExtractedField]:
        """
        Main extraction entry point.
        Processes OCR lines, full text, and tables to extract all state-configured fields.
        """
        extracted_fields: Dict[str, ExtractedField] = {}
        full_text = ocr_result.raw_full_text or "\n".join(line.text for line in ocr_result.text_lines)
        full_text = clean_ocr_text(full_text)  # same ZWNJ/ZWJ fix applied to the full-text fallback pass

        for field_name, field_spec in self.fields_def.items():
            field_obj = self._extract_single_field(
                field_name=field_name,
                field_spec=field_spec,
                ocr_result=ocr_result,
                full_text=full_text,
                handwriting_result=handwriting_result,
            )
            if field_obj is not None:
                extracted_fields[field_name] = field_obj

        # Also attempt extraction from detected tables (e.g. parcel listings)
        table_fields = self._extract_from_tables(ocr_result.tables)
        for k, v in table_fields.items():
            if k not in extracted_fields:
                extracted_fields[k] = v

        return extracted_fields

    def _extract_single_field(
        self,
        field_name: str,
        field_spec: Dict[str, Any],
        ocr_result: DocumentOCRResult,
        full_text: str,
        handwriting_result: Optional[HandwritingResult] = None,
    ) -> Optional[ExtractedField]:
        """Extracts one field using regex patterns defined in state config."""
        patterns = field_spec.get("patterns", [])
        
        # 1. First attempt matching per-line to get precise BoundingBox and Page
        for pattern_str in patterns:
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE | re.UNICODE)
            except re.error:
                continue

            for line_idx, line in enumerate(ocr_result.text_lines):
                line_text = clean_ocr_text(line.text)
                match = compiled.search(line_text)
                if match:
                    raw_val = match.group(1) if match.groups() else match.group(0)
                    raw_val = raw_val.strip()
                    if not raw_val:
                        continue

                    # Extract unit if captured as group 2
                    raw_unit = match.group(2).strip() if len(match.groups()) >= 2 and match.group(2) else None

                    evidence = FieldEvidence(
                        page_number=line.page_number,
                        bbox=line.bbox,
                        raw_ocr_text=line.text,
                        ocr_engine=str(line.engine),
                        extraction_rule_id=f"regex_{field_name}",
                        source_record=f"ocr_line_{line_idx}",
                    )

                    return ExtractedField(
                        field_name=field_name,
                        raw_value=raw_val,
                        normalized_value=raw_val,  # Will be normalized by normalization module
                        raw_unit=raw_unit,
                        confidence=line.confidence,
                        page=line.page_number,
                        bbox=line.bbox,
                        evidence=evidence,
                        extraction_method=ExtractionMethod.REGEX,
                        validation_status=ValidationStatus.UNVERIFIED,
                    )

        # 2. If not found line-by-line, search across the concatenated full_text (for multi-line patterns)
        for pattern_str in patterns:
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE | re.UNICODE | re.MULTILINE)
            except re.error:
                continue

            match = compiled.search(full_text)
            if match:
                raw_val = match.group(1) if match.groups() else match.group(0)
                raw_val = raw_val.strip()
                if not raw_val:
                    continue

                raw_unit = match.group(2).strip() if len(match.groups()) >= 2 and match.group(2) else None

                # Locate closest line bounding box if possible
                matched_line, line_num = self._find_matching_line(raw_val, ocr_result.text_lines)
                bbox = matched_line.bbox if matched_line else BoundingBox(x_min=0, y_min=0, x_max=0, y_max=0)
                page = matched_line.page_number if matched_line else 1
                line_conf = matched_line.confidence if matched_line else 0.80

                evidence = FieldEvidence(
                    page_number=page,
                    bbox=bbox,
                    raw_ocr_text=matched_line.text if matched_line else raw_val,
                    ocr_engine=str(matched_line.engine) if matched_line else "paddleocr",
                    extraction_rule_id=f"regex_multiline_{field_name}",
                    source_record=f"full_text_match",
                )

                return ExtractedField(
                    field_name=field_name,
                    raw_value=raw_val,
                    normalized_value=raw_val,
                    raw_unit=raw_unit,
                    confidence=line_conf,
                    page=page,
                    bbox=bbox,
                    evidence=evidence,
                    extraction_method=ExtractionMethod.REGEX,
                    validation_status=ValidationStatus.UNVERIFIED,
                )

        return None

    def _find_matching_line(self, snippet: str, lines: List[Any]) -> Tuple[Optional[Any], int]:
        """Finds the OCRTextLine containing the matched snippet."""
        snippet_clean = snippet.strip()
        for idx, line in enumerate(lines):
            if snippet_clean in line.text or any(part in line.text for part in snippet_clean.split() if len(part) > 2):
                return line, idx
        return None, -1

    def _extract_from_tables(self, tables: List[TableStructure]) -> Dict[str, ExtractedField]:
        """Extracts structured values from detected tables where column headers match terminology."""
        extracted = {}
        for table in tables:
            headers_lower = [h.lower() for h in table.headers]
            for r_idx, row in enumerate(table.rows):
                for c_idx, cell_value in enumerate(row):
                    if not cell_value or c_idx >= len(headers_lower):
                        continue
                    header = headers_lower[c_idx]

                    # Check for Khasra in table headers
                    if any(k in header for k in ["khasra", "खसरा", "गाटा", "survey", "plot"]):
                        if "khasra_number" not in extracted and cell_value.strip():
                            extracted["khasra_number"] = ExtractedField(
                                field_name="khasra_number",
                                raw_value=cell_value.strip(),
                                normalized_value=cell_value.strip(),
                                confidence=table.confidence,
                                page=table.page_number,
                                bbox=table.bbox,
                                evidence=FieldEvidence(
                                    page_number=table.page_number,
                                    bbox=table.bbox,
                                    raw_ocr_text=cell_value,
                                    ocr_engine="pp_structure",
                                    extraction_rule_id="table_lookup_khasra",
                                    source_record=f"table_{table.table_id}_row_{r_idx}_col_{c_idx}",
                                ),
                                extraction_method=ExtractionMethod.TABLE_LOOKUP,
                                validation_status=ValidationStatus.UNVERIFIED,
                            )

                    # Check for Area in table headers
                    if any(k in header for k in ["area", "रकबा", "क्षेत्रफल", "rakba", "क्षेत्र"]):
                        if "land_area" not in extracted and cell_value.strip():
                            extracted["land_area"] = ExtractedField(
                                field_name="land_area",
                                raw_value=cell_value.strip(),
                                normalized_value=cell_value.strip(),
                                confidence=table.confidence,
                                page=table.page_number,
                                bbox=table.bbox,
                                evidence=FieldEvidence(
                                    page_number=table.page_number,
                                    bbox=table.bbox,
                                    raw_ocr_text=cell_value,
                                    ocr_engine="pp_structure",
                                    extraction_rule_id="table_lookup_area",
                                    source_record=f"table_{table.table_id}_row_{r_idx}_col_{c_idx}",
                                ),
                                extraction_method=ExtractionMethod.TABLE_LOOKUP,
                                validation_status=ValidationStatus.UNVERIFIED,
                            )
        return extracted
