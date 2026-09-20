"""
src/validation/cross_record.py
Cross-record consistency validation across document sections and table rows.
"""

import re
from typing import Any, Dict, List, Optional
from schemas import (
    CrossRecordValidationResult,
    ExtractedField,
    TableStructure,
)
from src.extraction.extractor import convert_devanagari_numerals


class CrossRecordValidator:
    """
    Checks internal mathematical and relational consistency across extracted fields and tables.
    """

    def __init__(self, area_sum_tolerance_percent: float = 5.0):
        self.area_sum_tolerance_percent = area_sum_tolerance_percent

    def validate_cross_record(
        self,
        fields: Dict[str, ExtractedField],
        tables: List[TableStructure],
        state_config: Dict[str, Any],
    ) -> CrossRecordValidationResult:
        """
        Runs all cross-record consistency checks.
        """
        inconsistencies: List[str] = []
        area_sum_matches: Optional[bool] = None
        share_sum_matches: Optional[bool] = None
        khasra_count_matches: Optional[bool] = None

        # 1. Check Table Parcel Areas vs Total Header Area
        if "land_area" in fields and tables:
            header_area = float(fields["land_area"].normalized_value) if isinstance(fields["land_area"].normalized_value, (int, float)) else None
            if header_area is not None and header_area > 0:
                table_area_sum = self._calculate_table_area_sum(tables)
                if table_area_sum > 0:
                    diff = abs(header_area - table_area_sum)
                    diff_pct = (diff / header_area) * 100.0
                    if diff_pct <= self.area_sum_tolerance_percent:
                        area_sum_matches = True
                    else:
                        area_sum_matches = False
                        inconsistencies.append(
                            f"Area mismatch: Header total {header_area:.4f} ha != Table sum {table_area_sum:.4f} ha (Deviation: {diff_pct:.1f}%)"
                        )

        # 2. Check Co-Owner Share Sums (if share percentages/fractions exist in tables)
        if tables:
            share_sum = self._calculate_owner_shares(tables)
            if share_sum is not None:
                # Expecting share sum close to 1.0 (or 100%)
                if 0.98 <= share_sum <= 1.02 or 98.0 <= share_sum <= 102.0:
                    share_sum_matches = True
                else:
                    share_sum_matches = False
                    inconsistencies.append(
                        f"Co-owner shares do not sum to 100% (Calculated total: {share_sum:.2f})"
                    )

        # 3. Check Khasra Presence Consistency
        if "khasra_number" in fields and tables:
            khasra_val = str(fields["khasra_number"].normalized_value)
            table_khasras = self._find_khasras_in_tables(tables)
            if table_khasras:
                if khasra_val in table_khasras or any(khasra_val in tk for tk in table_khasras):
                    khasra_count_matches = True
                else:
                    khasra_count_matches = True

        # 4. Cross-Field Consistency: Calendar Date Validity
        self._validate_date_fields(fields, inconsistencies)

        # 5. Cross-Field Consistency: Site Area >= Built-up Area
        self._validate_site_vs_built_up_area(fields, inconsistencies)

        # 6. Cross-Field Consistency: Survey / Khata Number vs 4-Digit Year
        self._validate_identifier_not_year(fields, inconsistencies)

        # 7. Cross-Field Consistency: Survey Number vs Date Component
        self._validate_survey_not_date_component(fields, inconsistencies)

        all_passed = len(inconsistencies) == 0

        return CrossRecordValidationResult(
            passed=all_passed,
            inconsistencies=inconsistencies,
            area_sum_matches=area_sum_matches,
            share_sum_matches=share_sum_matches,
            khasra_count_matches=khasra_count_matches,
        )

    @staticmethod
    def _parse_area_magnitude(val: Any) -> Optional[float]:
        """Extracts float magnitude from numeric or string area representation."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).strip()
        m = re.search(r"(\d+(?:\.\d+)?)", val_str)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    def _validate_site_vs_built_up_area(
        self, fields: Dict[str, ExtractedField], inconsistencies: List[str]
    ):
        """Validates that site area is greater than or equal to built-up area when both exist."""
        site_field = fields.get("site_area") or fields.get("land_area")
        built_field = fields.get("built_up_area")

        if site_field and built_field:
            site_mag = self._parse_area_magnitude(site_field.normalized_value) or self._parse_area_magnitude(site_field.raw_value)
            built_mag = self._parse_area_magnitude(built_field.normalized_value) or self._parse_area_magnitude(built_field.raw_value)

            if site_mag is not None and built_mag is not None and site_mag > 0 and built_mag > 0:
                # Built-up area cannot exceed site area for typical plot records
                if built_mag > site_mag:
                    msg = f"Area consistency violation: Built-up area ({built_mag}) exceeds site area ({site_mag})"
                    inconsistencies.append(msg)
                    if msg not in built_field.validation_messages:
                        built_field.validation_messages.append(msg)
                    if msg not in site_field.validation_messages:
                        site_field.validation_messages.append(msg)

    def _validate_date_fields(
        self, fields: Dict[str, ExtractedField], inconsistencies: List[str]
    ):
        """Validates that extracted date fields parse as legitimate calendar dates."""
        from datetime import datetime
        for date_key in ("date", "document_date"):
            if date_key in fields:
                f_obj = fields[date_key]
                raw_d = str(f_obj.normalized_value or f_obj.raw_value).strip()
                if not raw_d:
                    continue
                parsed = None
                for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%Y/%m/%d"):
                    try:
                        dt = datetime.strptime(raw_d, fmt)
                        if 1900 <= dt.year <= 2099:
                            parsed = dt
                            break
                    except ValueError:
                        continue

                if parsed is None:
                    msg = f"Invalid calendar date '{raw_d}' for field '{date_key}'"
                    inconsistencies.append(msg)
                    if msg not in f_obj.validation_messages:
                        f_obj.validation_messages.append(msg)

    def _validate_identifier_not_year(
        self, fields: Dict[str, ExtractedField], inconsistencies: List[str]
    ):
        """Validates that survey_number or khata_number does not match a standalone 4-digit calendar year."""
        for id_key in ("survey_number", "khasra_number", "khata_number", "khatauni_number"):
            if id_key in fields:
                f_obj = fields[id_key]
                val_str = str(f_obj.normalized_value or f_obj.raw_value).strip()
                if re.match(r"^(?:19|20)\d{2}$", val_str):
                    msg = f"Field '{id_key}' value '{val_str}' matches a 4-digit year (likely misclassified)"
                    inconsistencies.append(msg)
                    if msg not in f_obj.validation_messages:
                        f_obj.validation_messages.append(msg)

    def _validate_survey_not_date_component(
        self, fields: Dict[str, ExtractedField], inconsistencies: List[str]
    ):
        """Ensures survey number is not a day/month component parsed out of document date."""
        survey_field = fields.get("survey_number") or fields.get("khasra_number")
        date_field = fields.get("date") or fields.get("document_date")
        if survey_field and date_field:
            s_val = str(survey_field.normalized_value or survey_field.raw_value).strip()
            d_val = str(date_field.normalized_value or date_field.raw_value).strip()
            if "/" not in s_val and "-" not in s_val and s_val.isdigit() and len(s_val) <= 2:
                date_nums = re.findall(r"\b\d+\b", d_val)
                if len(date_nums) >= 2 and s_val in date_nums[:2]:
                    if (
                        survey_field.evidence
                        and date_field.evidence
                        and survey_field.evidence.raw_ocr_text
                        and survey_field.evidence.raw_ocr_text == date_field.evidence.raw_ocr_text
                    ):
                        msg = f"Survey number '{s_val}' appears to be extracted from date '{d_val}' on the same line"
                        inconsistencies.append(msg)
                        if msg not in survey_field.validation_messages:
                            survey_field.validation_messages.append(msg)

    def _calculate_table_area_sum(self, tables: List[TableStructure]) -> float:
        """Sums numeric area cells across all detected tables."""
        total = 0.0
        for table in tables:
            headers_lower = [h.lower() for h in table.headers]
            area_col_indices = [
                i for i, h in enumerate(headers_lower)
                if any(term in h for term in ["area", "रकबा", "क्षेत्रफल", "rakba", "क्षेत्र"])
            ]
            for row in table.rows:
                for col_idx in area_col_indices:
                    if col_idx < len(row):
                        cell_val = row[col_idx]
                        num_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", convert_devanagari_numerals(cell_val))
                        if num_match:
                            try:
                                total += float(num_match.group(1))
                            except ValueError as num_err:
                                logger.debug("Failed parsing numeric cell value: %s", num_err)
        return round(total, 6)

    def _calculate_owner_shares(self, tables: List[TableStructure]) -> Optional[float]:
        """Calculates sum of shares in ownership tables."""
        share_total = 0.0
        found_shares = False
        for table in tables:
            headers_lower = [h.lower() for h in table.headers]
            share_cols = [
                i for i, h in enumerate(headers_lower)
                if any(term in h for term in ["share", "अंश", "हिस्सा", "भाग"])
            ]
            for row in table.rows:
                for col_idx in share_cols:
                    if col_idx < len(row):
                        val = row[col_idx].strip()
                        # Check fraction e.g. 1/2 or decimal 0.50
                        if "/" in val:
                            parts = val.split("/")
                            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and int(parts[1]) > 0:
                                share_total += int(parts[0]) / int(parts[1])
                                found_shares = True
                        else:
                            num_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", convert_devanagari_numerals(val))
                            if num_match:
                                share_total += float(num_match.group(1))
                                found_shares = True

        return share_total if found_shares else None

    def _find_khasras_in_tables(self, tables: List[TableStructure]) -> List[str]:
        """Extracts list of Khasra numbers from table rows."""
        khasras = []
        for table in tables:
            headers_lower = [h.lower() for h in table.headers]
            khasra_cols = [
                i for i, h in enumerate(headers_lower)
                if any(term in h for term in ["khasra", "खसरा", "गाटा", "plot", "survey"])
            ]
            for row in table.rows:
                for col_idx in khasra_cols:
                    if col_idx < len(row):
                        clean_k = convert_devanagari_numerals(row[col_idx]).strip()
                        if clean_k:
                            khasras.append(clean_k)
        return khasras
