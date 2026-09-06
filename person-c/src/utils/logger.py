"""
src/utils/logger.py
Structured logger for pipeline tracing with document IDs and stage milestones.
"""

import sys
from typing import Any, Dict, Optional
from loguru import logger

# Remove default logger and configure structured format
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)


class PipelineLogger:
    """Helper wrapper to log document lifecycle stages with document_id context."""

    def __init__(self, document_id: str = "SYSTEM"):
        self.document_id = document_id

    def _format_msg(self, stage: str, message: str, extra: Optional[Dict[str, Any]] = None) -> str:
        base = f"[{self.document_id}] [{stage}] {message}"
        if extra:
            sanitized = {k: v for k, v in extra.items() if "password" not in k.lower() and "token" not in k.lower()}
            base += f" | {sanitized}"
        return base

    def info(self, stage: str, message: str, **kwargs):
        logger.info(self._format_msg(stage, message, kwargs))

    def debug(self, stage: str, message: str, **kwargs):
        logger.debug(self._format_msg(stage, message, kwargs))

    def warning(self, stage: str, message: str, **kwargs):
        logger.warning(self._format_msg(stage, message, kwargs))

    def error(self, stage: str, message: str, **kwargs):
        logger.error(self._format_msg(stage, message, kwargs))

    # Standard Pipeline Milestones
    def doc_received(self, sha256: str, state: Optional[str] = None):
        self.info("DOCUMENT_RECEIVED", f"New document incoming. Hash: {sha256[:12]}... State: {state or 'AUTO'}")

    def preprocessing_started(self):
        self.info("PREPROCESSING_STARTED", "Person A preprocessing started")

    def preprocessing_completed(self, meta: Dict[str, Any]):
        self.info("PREPROCESSING_COMPLETED", "Preprocessing finished", **meta)

    def ocr_started(self, engine: str = "PaddleOCR"):
        self.info("OCR_STARTED", f"Printed OCR started using {engine}")

    def ocr_completed(self, lines_count: int, tables_count: int):
        self.info("OCR_COMPLETED", f"OCR finished with {lines_count} text lines and {tables_count} tables")

    def layout_detection_completed(self, table_count: int, block_count: int):
        self.info("LAYOUT_COMPLETED", f"Layout detection found {table_count} tables, {block_count} blocks")

    def classification_completed(self, doc_type: str, state: str, confidence: float):
        self.info("CLASSIFICATION_COMPLETED", f"Classified as {doc_type} in {state} (conf: {confidence:.2f})")

    def handwriting_started(self):
        self.info("HANDWRITING_STARTED", "Person B TrOCR handwriting recognition started")

    def handwriting_completed(self, regions_count: int):
        self.info("HANDWRITING_COMPLETED", f"Recognized {regions_count} handwritten crops")

    def extraction_completed(self, field_count: int):
        self.info("EXTRACTION_COMPLETED", f"Person C extracted {field_count} structured fields")

    def validation_completed(self, status: str, error_count: int):
        self.info("VALIDATION_COMPLETED", f"Validation finished with status: {status}, errors: {error_count}")

    def confidence_calculated(self, overall_conf: float, needs_review: bool):
        self.info("CONFIDENCE_CALCULATED", f"Overall confidence: {overall_conf:.3f}, Needs review: {needs_review}")

    def final_output_generated(self, execution_time_ms: float):
        self.info("FINAL_OUTPUT_GENERATED", f"Document processing finalized in {execution_time_ms:.1f}ms")
