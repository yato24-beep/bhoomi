"""Land Record Information Extraction Service.

Extracts structured revenue/cadastral fields (Owner Name, Survey Number, Khata Number,
Taluk, Village, Hobli, District, Area/Extent) from normalized OCR text.
Designed with a Pan-India multi-state extensible architecture (Karnataka-first).
"""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.services.translation_service import TranslationService

logger = logging.getLogger(__name__)


class ExtractedFieldRecord(BaseModel):
    """Structured field extracted from document text."""
    field_name: str
    raw_value: str
    normalized_value: str
    english_value: Optional[str] = None
    confidence: float = 0.85
    validation_status: str = "valid"  # valid | needs_review
    translation_status: Optional[str] = None


class ExtractionServiceResult(BaseModel):
    """Result of land-record field extraction across document."""
    state: str = "karnataka"
    document_type: str = "Land Record / RTC"
    fields: Dict[str, ExtractedFieldRecord] = Field(default_factory=dict)
    bilingual_fields: Dict[str, Dict[str, Optional[str]]] = Field(default_factory=dict)
    overall_confidence: float = 0.85
    requires_human_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)


class ExtractionService:
    """Service orchestrating land record cadastral field extraction."""

    @staticmethod
    def extract_fields(
        text: str,
        state: str = "karnataka",
        translate_fields: bool = True,
    ) -> ExtractionServiceResult:
        """Extract structured fields from normalized OCR text according to state schema."""
        if not text or not text.strip():
            return ExtractionServiceResult(state=state)

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            lines = [text.strip()]

        fields: Dict[str, ExtractedFieldRecord] = {}
        bilingual: Dict[str, Dict[str, Optional[str]]] = {}
        warnings: List[str] = []

        try:
            from src.extraction.land_record_ner import LandRecordFieldExtractor
            extractor = LandRecordFieldExtractor()
            records = extractor.extract_fields(lines)

            for fname, rec in records.items():
                eng_val: Optional[str] = None
                trans_status: Optional[str] = None

                if translate_fields and rec.normalized_value:
                    t_res = TranslationService.translate(
                        text=rec.normalized_value,
                        source_lang="kn" if state.lower() == "karnataka" else "auto",
                        target_lang="en",
                    )
                    if t_res.is_successful:
                        eng_val = t_res.translated_text
                        trans_status = "TRANSLATED"

                is_valid = not rec.requires_human_review
                field_record = ExtractedFieldRecord(
                    field_name=fname,
                    raw_value=rec.raw_value,
                    normalized_value=rec.normalized_value,
                    english_value=eng_val,
                    confidence=round(rec.confidence, 4),
                    validation_status="valid" if is_valid else "needs_review",
                    translation_status=trans_status,
                )
                fields[fname] = field_record
                bilingual[fname] = {
                    "kannada": rec.normalized_value,
                    "english": eng_val,
                }

                if not is_valid:
                    warnings.append(f"Low confidence extraction for field '{fname}' ({rec.confidence:.2f})")

        except Exception as exc:
            logger.warning(f"Field extraction pipeline error: {exc}")
            warnings.append(f"Extraction exception: {exc}")

        # Ensure fundamental baseline field exists if text is present
        if "handwritten_kannada_text" not in fields and text.strip():
            t_res = TranslationService.translate(text=text, source_lang="kn", target_lang="en")
            eng_val = t_res.translated_text if t_res.is_successful else None
            fields["handwritten_kannada_text"] = ExtractedFieldRecord(
                field_name="handwritten_kannada_text",
                raw_value=text,
                normalized_value=text,
                english_value=eng_val,
                confidence=0.85,
                validation_status="valid",
                translation_status="TRANSLATED" if eng_val else None,
            )

        confidences = [f.confidence for f in fields.values()]
        avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.85

        return ExtractionServiceResult(
            state=state,
            document_type="Karnataka Land Record (RTC / Bhoomi)" if state.lower() == "karnataka" else f"{state.title()} Land Record",
            fields=fields,
            bilingual_fields=bilingual,
            overall_confidence=avg_conf,
            requires_human_review=bool(warnings),
            review_reasons=warnings,
        )
