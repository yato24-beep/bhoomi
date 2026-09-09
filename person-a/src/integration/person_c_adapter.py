"""person-a/src/integration/person_c_adapter.py
Integration Adapter converting Person A OCROutput into canonical Person C DocumentOCRResult.
Ensures zero schema competition and 100% interoperability with Person C's extract_and_validate(...) interface.
"""

from typing import Any, Dict, List, Optional
from ..schemas import BlockType, BoundingBox, OCROutput


def convert_person_a_to_document_ocr_result(ocr_output: OCROutput) -> Dict[str, Any]:
    """Convert Person A's OCROutput into the dictionary payload conforming to Person C DocumentOCRResult."""
    # Convert text lines
    text_lines = []
    tables = []

    for page in ocr_output.pages:
        for block in page.blocks:
            for line in block.lines:
                text_lines.append({
                    "text": line.text,
                    "confidence": line.confidence,
                    "bbox": {
                        "x_min": line.bbox.x_min,
                        "y_min": line.bbox.y_min,
                        "x_max": line.bbox.x_max,
                        "y_max": line.bbox.y_max,
                        "normalized": line.bbox.normalized,
                    },
                    "page_number": page.page_number,
                    "language": line.language or "en",
                    "engine": ocr_output.ocr_engine,
                    "source_region": block.block_type.value,
                })

        for table in page.tables:
            cells = []
            for c in table.cells:
                cells.append({
                    "row_index": c.row_index,
                    "col_index": c.col_index,
                    "row_span": c.row_span,
                    "col_span": c.col_span,
                    "text": c.text,
                    "bbox": {
                        "x_min": c.bbox.x_min,
                        "y_min": c.bbox.y_min,
                        "x_max": c.bbox.x_max,
                        "y_max": c.bbox.y_max,
                        "normalized": c.bbox.normalized,
                    } if c.bbox else None,
                    "confidence": c.confidence,
                })

            tables.append({
                "table_id": table.table_id,
                "page_number": table.page_number,
                "bbox": {
                    "x_min": table.bbox.x_min,
                    "y_min": table.bbox.y_min,
                    "x_max": table.bbox.x_max,
                    "y_max": table.bbox.y_max,
                    "normalized": table.bbox.normalized,
                },
                "headers": table.headers,
                "rows": table.rows_data,
                "cells": cells,
                "confidence": table.confidence,
            })

    first_page_quality = ocr_output.pages[0].quality if ocr_output.pages else None

    payload = {
        "document_id": ocr_output.document_id,
        "sha256_hash": ocr_output.sha256_hash,
        "preprocessing": {
            "deskew_angle": first_page_quality.skew_angle if first_page_quality else 0.0,
            "super_resolution_applied": any("super_resolution" in p.quality.preprocessing_applied for p in ocr_output.pages),
            "denoised": any("bilateral_denoise" in p.quality.preprocessing_applied for p in ocr_output.pages),
            "contrast_enhanced": any("clahe" in p.quality.preprocessing_applied for p in ocr_output.pages),
            "original_resolution": (ocr_output.pages[0].width, ocr_output.pages[0].height) if ocr_output.pages else (0, 0),
            "processed_resolution": (ocr_output.pages[0].width, ocr_output.pages[0].height) if ocr_output.pages else (0, 0),
            "rotation_degrees": first_page_quality.rotation_needed if first_page_quality else 0,
            "page_count": len(ocr_output.pages),
        },
        "classification": {
            "document_type": ocr_output.classification.predicted_type,
            "state": ocr_output.classification.state or "UNKNOWN",
            "confidence": ocr_output.classification.confidence,
            "signals_matched": ocr_output.classification.matched_signals,
            "language": ocr_output.pages[0].language_info.primary_language if (ocr_output.pages and ocr_output.pages[0].language_info) else "en",
        },
        "text_lines": text_lines,
        "tables": tables,
        "raw_full_text": ocr_output.full_text,
        "pages_processed": len(ocr_output.pages),
        "processing_time_ms": ocr_output.processing_time_ms,
    }

    return payload
