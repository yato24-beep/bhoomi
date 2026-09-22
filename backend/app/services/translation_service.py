"""Translation Service Boundary.

Provides a clean separation between source language OCR (Kannada) and target translation (English).
Enforces the strict rule: English translation must NEVER be represented as original Kannada text.
If translation fails or cannot be computed, upstream OCR is preserved and an explicit status is set.
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TranslationServiceResult(BaseModel):
    """Result of a translation operation with explicit status and engine provenance."""
    original_text: str
    translated_text: Optional[str] = None
    source_lang: str = "kn"
    target_lang: str = "en"
    status: str = "COMPLETED"  # COMPLETED | FAILED | SKIPPED
    engine: str = "domain_glossary_v1"
    error_message: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        return self.status == "COMPLETED" and bool(self.translated_text and self.translated_text.strip())


class TranslationService:
    """Service providing robust land-record and document text translation."""

    @staticmethod
    def translate(
        text: str,
        source_lang: str = "kn",
        target_lang: str = "en",
    ) -> TranslationServiceResult:
        """Translate document text while strictly preserving source/target boundary."""
        if not text or not text.strip():
            return TranslationServiceResult(
                original_text=text or "",
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                status="SKIPPED",
                engine="noop",
            )

        cleaned_text = text.strip()
        try:
            from src.translation.translator import translate_bidirectional
            translated = translate_bidirectional(
                text=cleaned_text,
                source_lang=source_lang,
                target_lang=target_lang,
            )

            if translated and translated.strip():
                return TranslationServiceResult(
                    original_text=cleaned_text,
                    translated_text=translated.strip(),
                    source_lang=source_lang,
                    target_lang=target_lang,
                    status="COMPLETED",
                    engine="bidirectional_pipeline",
                )
            else:
                return TranslationServiceResult(
                    original_text=cleaned_text,
                    translated_text=None,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    status="COMPLETED",
                    engine="bidirectional_pipeline",
                )
        except Exception as exc:
            logger.warning(f"TranslationService failure for text length {len(cleaned_text)}: {exc}")
            return TranslationServiceResult(
                original_text=cleaned_text,
                translated_text=None,
                source_lang=source_lang,
                target_lang=target_lang,
                status="FAILED",
                engine="error",
                error_message=str(exc),
            )
