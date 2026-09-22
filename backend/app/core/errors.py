"""Standardized Application Error Codes & Exceptions.

Enforces machine-readable error codes, user-readable messages, and isolated internal context
without leaking sensitive stack traces or internal implementation paths to clients.
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class AppErrorCode:
    """Standardized machine-readable error codes."""
    INVALID_FILE = "INVALID_FILE"
    FILE_UPLOAD_FAILED = "FILE_UPLOAD_FAILED"
    IMAGE_STORAGE_FAILED = "IMAGE_STORAGE_FAILED"
    OCR_FAILED = "OCR_FAILED"
    OCR_INSUFFICIENT_CONFIDENCE = "OCR_INSUFFICIENT_CONFIDENCE"
    TRANSLATION_FAILED = "TRANSLATION_FAILED"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"


class StandardizedAppException(HTTPException):
    """Base exception for all domain errors."""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        user_message: str,
        diagnostic_context: Optional[Dict[str, Any]] = None,
    ):
        detail = {
            "error_code": error_code,
            "message": user_message,
        }
        if diagnostic_context:
            detail["context"] = diagnostic_context
        super().__init__(status_code=status_code, detail=detail)


class DocumentNotFoundException(StandardizedAppException):
    def __init__(self, document_id: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=AppErrorCode.DOCUMENT_NOT_FOUND,
            user_message=f"Document with ID {document_id} was not found.",
            diagnostic_context={"document_id": document_id},
        )


class PermissionDeniedException(StandardizedAppException):
    def __init__(self, required_role: str = "ADMIN"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_code=AppErrorCode.PERMISSION_DENIED,
            user_message="You do not have sufficient permissions to perform this operation.",
            diagnostic_context={"required_role": required_role},
        )


class StorageException(StandardizedAppException):
    def __init__(self, message: str = "Failed to store document in storage."):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code=AppErrorCode.IMAGE_STORAGE_FAILED,
            user_message=message,
        )
