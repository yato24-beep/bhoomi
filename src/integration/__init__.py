"""Integration package for Land Record Digitization OCR & Handwriting Pipeline."""

from src.integration.document_pipeline import (
    DocumentProcessingPipeline,
    DocumentProcessingResult,
    get_default_pipeline,
    process_document,
)
from src.integration.person_a_adapter import PersonAAdapter
from src.integration.schemas import (
    DocumentProcessingRequest,
    DocumentProcessingResponse,
    ProcessingStatus,
    RecognizedRegionResult,
    RegionRequest,
    RegionType,
)

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
    "get_default_pipeline",
    "process_document",
]
