"""Document processing pipeline interface and implementations."""
from app.pipeline.processor import (
    BaseDocumentProcessor,
    MockDocumentProcessor,
    ProcessingResult,
    get_document_processor,
)

__all__ = [
    "BaseDocumentProcessor",
    "MockDocumentProcessor",
    "ProcessingResult",
    "get_document_processor",
]
