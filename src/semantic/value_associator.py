"""Deterministic 2D Spatial Key/Value Associator for Land Records.

Associates detected field label anchors with their corresponding value text regions using:
- Inline key-value parsing (e.g. "ಸರ್ವೆ ನಂ: 125/1" -> value="125/1")
- Rightward horizontal proximity (same row / reading line)
- Downward vertical proximity (tabular or form column format)
- Cadastral pattern matching rules without speculative hallucination
"""

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from schemas import BoundingBox
from src.semantic.field_detector import DetectedAnchor
from src.semantic.schema import FieldProvenance, SemanticFieldItem, ValidationStatus

logger = logging.getLogger(__name__)

# Standalone cadastral and value regex patterns
STANDALONE_PATTERNS: Dict[str, re.Pattern] = {
    "survey_number": re.compile(r"\b(\d{1,4}(?:/[0-9A-Za-z\-_]+)?)\b"),
    "registration_date": re.compile(r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b"),
    "extent": re.compile(r"\b(\d{1,3}[-\s]\d{1,2}(?:[-\s]\d{1,2})?|\d{1,3}\.\d{1,4})\b"),
}


class ValueAssociator:
    """Associates field anchors with value regions using geometric layout relationships."""

    def __init__(
        self,
        max_horizontal_gap: float = 350.0,
        max_vertical_gap: float = 120.0,
    ):
        self.max_horizontal_gap = max_horizontal_gap
        self.max_vertical_gap = max_vertical_gap

    def associate_fields(
        self,
        anchors: List[DetectedAnchor],
        regions: List[Any],
        document_id: Optional[str] = None,
        page_number: int = 1,
    ) -> Dict[str, SemanticFieldItem]:
        """Maps detected field anchors to value text regions.

        Returns:
            Dictionary mapping canonical field names to SemanticFieldItem instances with provenance.
        """
        extracted_fields: Dict[str, SemanticFieldItem] = {}
        used_region_ids = set()

        # Phase 1: Inline Key-Value Associations (Highest precision)
        for anchor in anchors:
            if anchor.is_inline_key_value and anchor.inline_value:
                val = anchor.inline_value.strip()
                if val:
                    prov = FieldProvenance(
                        document_id=document_id,
                        page_number=page_number,
                        region_id=anchor.region_id,
                        bbox=anchor.bbox,
                        raw_ocr_text=anchor.text,
                        normalized_value=val,
                        extraction_method="inline_delimiter",
                    )
                    extracted_fields[anchor.field_name] = SemanticFieldItem(
                        field_name=anchor.field_name,
                        value=val,
                        raw_value=val,
                        validation_status=ValidationStatus.UNVERIFIED,
                        source_region_ids=[anchor.region_id],
                        provenance=prov,
                    )
                    used_region_ids.add(anchor.region_id)

        # Build lookup table of candidate value regions
        anchor_region_ids = {a.region_id for a in anchors}
        candidate_regions = [
            r for r in regions
            if getattr(r, "region_id", None) not in used_region_ids
        ]

        # Phase 2: Rightward & Downward 2D Proximity Association
        for anchor in anchors:
            if anchor.field_name in extracted_fields:
                continue
            if not anchor.bbox:
                continue

            best_match_reg = None
            best_dist = float("inf")
            best_method = "spatial"

            a_box = anchor.bbox
            a_mid_y = (a_box.y_min + a_box.y_max) / 2.0
            a_mid_x = (a_box.x_min + a_box.x_max) / 2.0

            for cand in candidate_regions:
                c_id = getattr(cand, "region_id", None)
                if c_id in used_region_ids or c_id in anchor_region_ids:
                    continue

                c_box = getattr(cand, "bbox", None)
                if not c_box:
                    continue

                c_text = (getattr(cand, "normalized_text", None) or getattr(cand, "text", "")).strip()
                if not c_text:
                    continue

                # Case A: Rightward horizontal proximity (same text row)
                # Check vertical alignment overlap
                overlap_y = max(0.0, min(a_box.y_max, c_box.y_max) - max(a_box.y_min, c_box.y_min))
                min_h = min(a_box.height, c_box.height)
                is_same_row = (overlap_y > 0.3 * min_h) or (abs(a_mid_y - (c_box.y_min + c_box.y_max) / 2.0) < 15.0)

                if is_same_row and c_box.x_min >= a_box.x_min - 5.0:
                    dist_x = c_box.x_min - a_box.x_max
                    if 0 <= dist_x <= self.max_horizontal_gap:
                        if dist_x < best_dist:
                            best_dist = dist_x
                            best_match_reg = cand
                            best_method = "rightward_horizontal"

                # Case B: Downward vertical proximity (table column header to cell)
                # Check horizontal alignment overlap
                overlap_x = max(0.0, min(a_box.x_max, c_box.x_max) - max(a_box.x_min, c_box.x_min))
                min_w = min(a_box.width, c_box.width)
                is_same_col = (overlap_x > 0.3 * min_w) or (abs(a_mid_x - (c_box.x_min + c_box.x_max) / 2.0) < 25.0)

                if is_same_col and c_box.y_min >= a_box.y_max - 5.0:
                    dist_y = c_box.y_min - a_box.y_max
                    if 0 <= dist_y <= self.max_vertical_gap:
                        if dist_y < best_dist:
                            best_dist = dist_y
                            best_match_reg = cand
                            best_method = "downward_vertical"

            if best_match_reg:
                cand_text = (getattr(best_match_reg, "normalized_text", None) or getattr(best_match_reg, "text", "")).strip()
                cand_raw = getattr(best_match_reg, "raw_text", cand_text)
                cand_id = getattr(best_match_reg, "region_id", "unknown_reg")
                cand_box = getattr(best_match_reg, "bbox", None)

                prov = FieldProvenance(
                    document_id=document_id,
                    page_number=page_number,
                    region_id=cand_id,
                    bbox=cand_box,
                    raw_ocr_text=cand_raw,
                    normalized_value=cand_text,
                    extraction_method=best_method,
                )
                extracted_fields[anchor.field_name] = SemanticFieldItem(
                    field_name=anchor.field_name,
                    value=cand_text,
                    raw_value=cand_raw,
                    validation_status=ValidationStatus.UNVERIFIED,
                    source_region_ids=[anchor.region_id, cand_id],
                    provenance=prov,
                )
                used_region_ids.add(cand_id)

        # Phase 3: Standalone Pattern Fallbacks (when explicit anchors were omitted on clear records)
        for field_name, pattern in STANDALONE_PATTERNS.items():
            if field_name not in extracted_fields:
                for cand in candidate_regions:
                    cand_id = getattr(cand, "region_id", None)
                    if cand_id in used_region_ids:
                        continue
                    text = (getattr(cand, "normalized_text", None) or getattr(cand, "text", "")).strip()
                    match = pattern.search(text)
                    if match:
                        matched_val = match.group(1).strip()
                        cand_raw = getattr(cand, "raw_text", text)
                        cand_box = getattr(cand, "bbox", None)
                        prov = FieldProvenance(
                            document_id=document_id,
                            page_number=page_number,
                            region_id=cand_id,
                            bbox=cand_box,
                            raw_ocr_text=cand_raw,
                            normalized_value=matched_val,
                            extraction_method="standalone_pattern",
                        )
                        extracted_fields[field_name] = SemanticFieldItem(
                            field_name=field_name,
                            value=matched_val,
                            raw_value=matched_val,
                            validation_status=ValidationStatus.UNVERIFIED,
                            source_region_ids=[cand_id],
                            provenance=prov,
                        )
                        used_region_ids.add(cand_id)
                        break

        return extracted_fields

    def associate_regions(
        self,
        regions: List[Any],
        document_id: Optional[str] = None,
        page_number: int = 1,
    ) -> Dict[str, SemanticFieldItem]:
        """Convenience method that runs anchor detection then association."""
        from src.semantic.field_detector import FieldDetector
        detector = FieldDetector()
        anchors = detector.detect_anchors(regions)
        return self.associate_fields(
            anchors=anchors,
            regions=regions,
            document_id=document_id,
            page_number=page_number,
        )


SpatialValueAssociator = ValueAssociator
