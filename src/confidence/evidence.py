"""
src/confidence/evidence.py
Provenance and evidence tracking for extracted fields.
Ensures full auditability answering: 'Where did this value come from?'
"""

from typing import Any, Dict, List, Optional
from schemas import (
    BoundingBox,
    ExtractedField,
    FieldEvidence,
)


class EvidenceTracker:
    """
    Builds and enriches audit evidence trails for all document fields.
    """

    @staticmethod
    def create_evidence(
        page_number: int,
        bbox: Optional[BoundingBox],
        raw_ocr_text: str,
        ocr_engine: str = "paddleocr",
        extraction_rule_id: str = "regex_default",
        source_record: Optional[str] = None,
        handwriting_text: Optional[str] = None,
        validation_notes: Optional[List[str]] = None,
    ) -> FieldEvidence:
        """Constructs a comprehensive FieldEvidence object."""
        return FieldEvidence(
            page_number=page_number,
            bbox=bbox,
            raw_ocr_text=raw_ocr_text,
            handwriting_text=handwriting_text,
            ocr_engine=ocr_engine,
            extraction_rule_id=extraction_rule_id,
            source_record=source_record,
            validation_notes=validation_notes or [],
        )

    @staticmethod
    def append_validation_note(field_obj: ExtractedField, note: str):
        """Appends an audit note to the field's evidence object."""
        if note not in field_obj.evidence.validation_notes:
            field_obj.evidence.validation_notes.append(note)
