"""person-a/src/ocr/__init__.py"""
from .paddle_ocr import PaddleOCREngine
from .consensus import (
    OCRConsensusCandidate,
    OCRConsensusResult,
    OCRConsensusEngine,
    normalized_levenshtein_distance,
)

__all__ = [
    "PaddleOCREngine",
    "OCRConsensusCandidate",
    "OCRConsensusResult",
    "OCRConsensusEngine",
    "normalized_levenshtein_distance",
]
