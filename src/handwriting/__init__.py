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
from src.handwriting.paddle_recognizer import PaddleKannadaRecognizer
from src.handwriting.recognizer import (
    BaseHandwritingRecognizer,
    ImageInput,
)
from src.handwriting.router import (
    LANGUAGE_ALIASES,
    LanguageScriptRouter,
)
from src.handwriting.service import HandwritingOCRService
from src.handwriting.trocr_recognizer import TrocrHandwritingRecognizer

__all__ = [
    "BaseHandwritingRecognizer",
    "ImageInput",
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
