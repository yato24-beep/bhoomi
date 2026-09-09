"""person-a/src/integration/backend_adapter.py
Backend adapter converting Person A OCROutput into the backend's ProcessingResult format.
Enables immediate Celery worker integration and PostgreSQL database persistence.
"""

import time
from typing import Any, BinaryIO, Dict, List, Optional
from ..pipeline import process_document
from ..schemas import DocumentInput, OCROutput


def process_backend_stream_to_result(
    file_stream: BinaryIO,
    filename: str,
    content_type: Optional[str] = None,
    state_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute Person A pipeline on a backend stream and format into ProcessingResult dictionary."""
    start_time = time.perf_counter()

    file_stream.seek(0)
    raw_bytes = file_stream.read()

    doc_id = filename.rsplit(".", 1)[0] if "." in filename else filename
    doc_input = DocumentInput(
        document_id=doc_id,
        file_bytes=raw_bytes,
        state_hint=state_hint,
    )

    ocr_output: OCROutput = process_document(doc_input)

    # Flatten fields from OCR blocks & tables with normalized bounding boxes
    fields: List[Dict[str, Any]] = []
    total_w = ocr_output.pages[0].width if ocr_output.pages else 1
    total_h = ocr_output.pages[0].height if ocr_output.pages else 1

    for page in ocr_output.pages:
        for block in page.blocks:
            for line in block.lines:
                norm_bbox = {
                    "x_min": round(line.bbox.x_min / total_w, 4),
                    "y_min": round(line.bbox.y_min / total_h, 4),
                    "x_max": round(line.bbox.x_max / total_w, 4),
                    "y_max": round(line.bbox.y_max / total_h, 4),
                    "unit": "normalized",
                }
                fields.append({
                    "field_name": f"{block.block_type.value}_line",
                    "original_value": line.text,
                    "normalized_value": line.text.strip(),
                    "confidence_score": line.confidence,
                    "source_page": page.page_number,
                    "bounding_box": norm_bbox,
                })

    structured_data = {
        "document_type": ocr_output.classification.predicted_type,
        "state": ocr_output.classification.state or "UNKNOWN",
        "full_text": ocr_output.full_text,
        "pages_count": len(ocr_output.pages),
        "tables_count": sum(len(p.tables) for p in ocr_output.pages),
        "sha256": ocr_output.sha256_hash,
    }

    duration_ms = int((time.perf_counter() - start_time) * 1000)
    avg_conf = ocr_output.overall_confidence

    return {
        "extracted_data": structured_data,
        "fields": fields,
        "confidence_score": round(avg_conf, 2),
        "is_valid": len(ocr_output.pages) > 0,
        "validation_info": {
            "checks_passed": [f"ingestion_verified: {len(ocr_output.pages)} page(s)"],
            "errors": [] if len(ocr_output.pages) > 0 else ["No readable pages extracted"],
            "rules_evaluated_count": 1,
        },
        "processing_time_ms": max(duration_ms, int(ocr_output.processing_time_ms)),
    }
