"""Evidence-Based Engineering Confidence Calculator for Land Record Semantic Extraction.

Calculates multi-factor confidence without assuming LLM self-reported confidence
is a calibrated probability.

Components evaluated:
1. Grounding Strength (exact match, token substring overlap)
2. Spatial / Layout Proximity (table cell alignment, proximity to labels)
3. Deterministic Format Validation (cadastral format, calendar date, extent patterns)
4. Provenance Completeness (page, region_id, bounding box, raw text)
5. Candidate Conflict Penalty (penalizes ambiguous/competing values)
"""

import logging
import re
from typing import Any, Dict, List, Optional

from src.semantic.schema import FieldProvenance, ValidationStatus

logger = logging.getLogger(__name__)


class EvidenceConfidenceCalculator:
    """Calculates evidence-grounded engineering confidence for semantic fields."""

    def __init__(
        self,
        weight_grounding: float = 0.35,
        weight_validation: float = 0.30,
        weight_provenance: float = 0.20,
        weight_spatial: float = 0.15,
        conflict_penalty: float = 0.40,
    ):
        self.weight_grounding = weight_grounding
        self.weight_validation = weight_validation
        self.weight_provenance = weight_provenance
        self.weight_spatial = weight_spatial
        self.conflict_penalty = conflict_penalty

    def calculate_field_confidence(
        self,
        field_name: str,
        raw_value: str,
        normalized_value: str,
        provenance: Optional[FieldProvenance],
        validation_status: ValidationStatus,
        has_conflicts: bool = False,
        is_table_cell: bool = False,
        model_reported_confidence: Optional[float] = None,
    ) -> float:
        """Computes engineering confidence score in range [0.0, 1.0]."""
        if not raw_value or not raw_value.strip():
            return 0.0

        raw_val_clean = raw_value.strip()

        # 1. Grounding score: check provenance raw OCR text vs raw_value
        grounding_score = 0.0
        if provenance and provenance.raw_ocr_text:
            source_text = provenance.raw_ocr_text.strip()
            if raw_val_clean in source_text:
                grounding_score = 1.0
            else:
                # Check token overlap
                raw_tokens = set(raw_val_clean.split())
                source_tokens = set(source_text.split())
                if raw_tokens and raw_tokens.issubset(source_tokens):
                    grounding_score = 0.9
                elif raw_tokens and (raw_tokens & source_tokens):
                    grounding_score = 0.6
                else:
                    # Not found in source text! Serious grounding deficiency
                    grounding_score = 0.0
        elif provenance:
            grounding_score = 0.5

        # 2. Validation score: based on deterministic validator status
        if validation_status == ValidationStatus.VALID:
            val_score = 1.0
        elif validation_status == ValidationStatus.WARNING:
            val_score = 0.6
        elif validation_status == ValidationStatus.UNVERIFIED:
            val_score = 0.5
        else:  # INVALID
            val_score = 0.0

        # 3. Provenance completeness
        prov_score = 0.0
        if provenance:
            has_reg = bool(provenance.region_id and provenance.region_id.strip())
            has_box = bool(provenance.bbox is not None)
            has_raw = bool(provenance.raw_ocr_text and provenance.raw_ocr_text.strip())
            prov_score = (0.4 if has_reg else 0.0) + (0.3 if has_box else 0.0) + (0.3 if has_raw else 0.0)

        # 4. Spatial / Layout score
        spatial_score = 1.0 if is_table_cell else (0.85 if provenance and provenance.bbox else 0.5)

        # Weighted combination
        base_score = (
            self.weight_grounding * grounding_score
            + self.weight_validation * val_score
            + self.weight_provenance * prov_score
            + self.weight_spatial * spatial_score
        )

        # Apply conflict penalty if contradictory values were detected
        if has_conflicts:
            base_score = max(0.0, base_score - self.conflict_penalty)

        # If validation completely failed or grounding is zero, clamp to low score
        if validation_status == ValidationStatus.INVALID or grounding_score == 0.0:
            base_score = min(base_score, 0.25)

        return round(float(base_score), 4)
