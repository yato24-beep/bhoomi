"""
src/extraction/disagreement.py
OCR Consensus and Disagreement Resolution Engine.
Compares printed OCR, handwriting recognition, and validation evidence without silently overwriting.
"""

from typing import Any, Dict, List, Optional, Tuple
from schemas import (
    BoundingBox,
    ExtractedField,
    ExtractionMethod,
    FieldEvidence,
    HandwritingResult,
    ValidationStatus,
)


class DisagreementResolver:
    """
    Detects and resolves spatial and textual conflicts between printed OCR and handwriting recognition.
    """

    def __init__(self, iou_overlap_threshold: float = 0.25):
        self.iou_overlap_threshold = iou_overlap_threshold

    def resolve_conflicts(
        self,
        extracted_fields: Dict[str, ExtractedField],
        handwriting_result: Optional[HandwritingResult],
        state_config: Dict[str, Any],
    ) -> Tuple[Dict[str, ExtractedField], List[Dict[str, Any]]]:
        """
        Detects if any extracted field overlaps spatially with a handwritten region.
        Compares printed vs handwritten interpretations.
        """
        if not handwriting_result or not handwriting_result.regions:
            return extracted_fields, []

        resolved_fields = dict(extracted_fields)
        disagreement_log: List[Dict[str, Any]] = []

        for field_name, field_obj in list(resolved_fields.items()):
            if field_obj.bbox is None:
                continue

            for hw_region in handwriting_result.regions:
                # Check page match and bounding box spatial overlap
                if hw_region.page_number != field_obj.page:
                    continue

                if field_obj.bbox.overlaps_with(hw_region.bbox, self.iou_overlap_threshold):
                    # Spatial overlap detected! Check for text disagreement
                    printed_val = field_obj.raw_value.strip()
                    hw_val = hw_region.text.strip()

                    if printed_val.lower() != hw_val.lower():
                        # Conflict identified
                        resolution_record = self._handle_disagreement(
                            field_name=field_name,
                            field_obj=field_obj,
                            hw_region=hw_region,
                            state_config=state_config,
                        )
                        disagreement_log.append(resolution_record)

                        # Update field with fused evidence and penalized confidence
                        field_obj.evidence.handwriting_text = hw_val
                        field_obj.evidence.validation_notes.append(
                            f"OCR Disagreement: Printed '{printed_val}' vs Handwritten '{hw_val}' (Resolved as: {field_obj.raw_value})"
                        )

        return resolved_fields, disagreement_log

    def _handle_disagreement(
        self,
        field_name: str,
        field_obj: ExtractedField,
        hw_region: Any,
        state_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Rule-based resolution:
        1. In fill-in-the-blank forms, handwritten text often overrides pre-printed template background.
        2. If handwriting confidence is high and printed text is boilerplate label, choose handwriting.
        3. If both confidences are close, flag for review and apply confidence penalty.
        """
        printed_val = field_obj.raw_value
        printed_conf = field_obj.confidence
        hw_val = hw_region.text
        hw_conf = hw_region.confidence

        chosen_value = printed_val
        chosen_source = "printed_ocr"
        resolution_strategy = "printed_priority"
        penalty = 0.15

        # If handwriting is significantly more confident and has non-empty text
        if hw_conf > printed_conf + 0.15 and len(hw_val) > 0:
            chosen_value = hw_val
            chosen_source = "handwriting_ocr"
            resolution_strategy = "handwriting_high_confidence"
            field_obj.raw_value = hw_val
            field_obj.extraction_method = ExtractionMethod.HANDWRITING_FUSION
            penalty = 0.10
        elif printed_conf > hw_conf + 0.20:
            chosen_value = printed_val
            chosen_source = "printed_ocr"
            resolution_strategy = "printed_high_confidence"
            penalty = 0.10
        else:
            # Unclear conflict - keep printed but mark conflict status and heavier penalty
            field_obj.validation_status = ValidationStatus.CONFLICT
            field_obj.validation_messages.append(
                f"Unresolved conflict between printed '{printed_val}' (conf: {printed_conf:.2f}) and handwritten '{hw_val}' (conf: {hw_conf:.2f})"
            )
            penalty = 0.30
            resolution_strategy = "unresolved_ambiguity"

        # Apply confidence adjustment
        field_obj.confidence = max(0.05, field_obj.confidence - penalty)

        return {
            "field_name": field_name,
            "printed_value": printed_val,
            "printed_confidence": printed_conf,
            "handwritten_value": hw_val,
            "handwritten_confidence": hw_conf,
            "chosen_value": chosen_value,
            "chosen_source": chosen_source,
            "strategy": resolution_strategy,
            "confidence_penalty_applied": penalty,
        }
