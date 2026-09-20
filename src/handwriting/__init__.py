"""Handwritten text recognition modules for Land Record Digitization (Person B).

Public API exports:
- BaseHandwritingRecognizer: Abstract interface for handwriting OCR backends.
- PaddleKannadaRecognizer: Baseline PaddleOCR backend for Kannada and Indic script OCR.
- LanguageScriptRouter: Configurable router for regional languages (Kannada, Telugu, Tamil, Hindi, etc.).
- HandwritingOCRService: Service layer using recognizers via dependency injection.
- Confidence utilities: Functions for auditable, non-fabricated confidence evaluation.
"""

from src.handwriting.confidence import (
    build_confidence_audit_trail,
    classify_confidence_tier,
    compute_min_token_confidence,
    compute_token_geometric_mean_confidence,
    compute_token_mean_confidence,
)
from src.handwriting.recognizer import (
    BaseHandwritingRecognizer,
    ImageInput,
)


def __getattr__(name: str):
    if name == "EasyOCRKannadaRecognizer":
        from src.handwriting.easyocr_recognizer import EasyOCRKannadaRecognizer
        return EasyOCRKannadaRecognizer
    if name == "PaddleKannadaRecognizer":
        from src.handwriting.paddle_recognizer import PaddleKannadaRecognizer
        return PaddleKannadaRecognizer
    if name == "TrocrHandwritingRecognizer":
        from src.handwriting.trocr_recognizer import TrocrHandwritingRecognizer
        return TrocrHandwritingRecognizer
    if name in ("LanguageScriptRouter", "LANGUAGE_ALIASES"):
        import src.handwriting.router as _router
        return getattr(_router, name)
    if name == "HandwritingOCRService":
        from src.handwriting.service import HandwritingOCRService
        return HandwritingOCRService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseHandwritingRecognizer",
    "ImageInput",
    "EasyOCRKannadaRecognizer",
    "PaddleKannadaRecognizer",
    "TrocrHandwritingRecognizer",
    "LanguageScriptRouter",
    "LANGUAGE_ALIASES",
    "HandwritingOCRService",
    "compute_token_mean_confidence",
    "compute_token_geometric_mean_confidence",
    "compute_min_token_confidence",
    "classify_confidence_tier",
    "build_confidence_audit_trail",
]

