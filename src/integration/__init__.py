"""Integration package for Land Record Digitization OCR & Handwriting Pipeline."""

from src.integration.person_a_adapter import PersonAAdapter
from src.integration.person_c_adapter import PersonCAdapter
from src.integration.person_c_service import extract_and_validate
from src.integration.schemas import (
    DocumentProcessingRequest,
    DocumentProcessingResponse,
    ProcessingStatus,
    RecognizedRegionResult,
    RegionRequest,
    RegionType,
)


def get_default_pipeline():
    from src.integration.document_pipeline import get_default_pipeline as _gdp
    return _gdp()


def process_document(*args, **kwargs):
    from src.integration.document_pipeline import process_document as _pd
    return _pd(*args, **kwargs)


def __getattr__(name: str):
    if name in ("DocumentProcessingPipeline", "DocumentProcessingResult"):
        import src.integration.document_pipeline as _dp
        return getattr(_dp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DocumentProcessingPipeline",
    "DocumentProcessingResult",
    "DocumentProcessingRequest",
    "DocumentProcessingResponse",
    "RegionRequest",
    "RecognizedRegionResult",
    "RegionType",
    "ProcessingStatus",
    "PersonAAdapter",
    "PersonCAdapter",
    "extract_and_validate",
    "get_default_pipeline",
    "process_document",
]
