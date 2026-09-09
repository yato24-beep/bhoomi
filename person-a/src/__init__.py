"""person-a/src/__init__.py
Person A Vision, Multilingual OCR, Layout, Tables, and Preprocessing Engine.
"""

from .pipeline import PersonAPipeline, process_document, process_document_with_handwriting
from .schemas import (
    BlockType,
    BoundingBox,
    ClassificationResult,
    DocumentInput,
    ImageQualityAssessment,
    LanguageDetectionResult,
    OCRBlock,
    OCREngineType,
    OCRLine,
    OCROutput,
    OCRPageResult,
    OCRWord,
    TableCell,
    TableStructure,
)
from .utils.hashing import compute_document_hash

__all__ = [
    "PersonAPipeline",
    "process_document",
    "process_document_with_handwriting",
    "compute_document_hash",
    "BlockType",
    "BoundingBox",
    "ClassificationResult",
    "DocumentInput",
    "ImageQualityAssessment",
    "LanguageDetectionResult",
    "OCRBlock",
    "OCREngineType",
    "OCRLine",
    "OCROutput",
    "OCRPageResult",
    "OCRWord",
    "TableCell",
    "TableStructure",
]
