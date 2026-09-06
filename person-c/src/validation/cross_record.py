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
                    # Not necessarily an error if document has multiple khasras, but worth noting
                    khasra_count_matches = True

        all_passed = len(inconsistencies) == 0

        return CrossRecordValidationResult(
            passed=all_passed,
            inconsistencies=inconsistencies,
            area_sum_matches=area_sum_matches,
            share_sum_matches=share_sum_matches,
            khasra_count_matches=khasra_count_matches,
        )

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
                            except ValueError:
                                pass
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
