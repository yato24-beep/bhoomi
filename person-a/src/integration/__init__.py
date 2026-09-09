"""person-a/src/integration/__init__.py"""
from .person_c_adapter import convert_person_a_to_document_ocr_result
from .backend_adapter import process_backend_stream_to_result

__all__ = [
    "convert_person_a_to_document_ocr_result",
    "process_backend_stream_to_result",
]
