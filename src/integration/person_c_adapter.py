"""Person C Integration Adapter.

Provides seamless conversion and bridging between:
- Person B's multimodal OCR output (`DocumentProcessingResponse`, `RecognizedRegionResult`)
- Person C's extraction & validation engine (`extract_and_validate`, `FinalDocumentResult`)
- Unified API responses containing both comprehensive OCR evidence and structured land record data.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from sqlalchemy.orm import Session

from schemas import (
    BoundingBox,
    DocumentClassificationResult,
    DocumentOCRResult,
    DocumentType,
    ExtractedField,
    FinalDocumentResult,
    HandwritingRegionResult,
    HandwritingResult,
    OCREngineType,
    OCRTextLine,
    PreprocessingMetadata,
    TableStructure,
    ValidationStatus,
)
from src.integration.schemas import DocumentProcessingResponse, RecognizedRegionResult
from src.integration.person_c_service import extract_and_validate
from src.utils.config_loader import ConfigLoader


class PersonCAdapter:
    """Adapter facilitating data translation and pipeline execution between Person B and Person C."""

    @staticmethod
    def _convert_bbox(bbox_obj: Optional[Any]) -> BoundingBox:
        """Converts any BoundingBox representation into Person C standard BoundingBox."""
        if bbox_obj is None:
            return BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)
        
        x_min = getattr(bbox_obj, "x_min", 0.0)
        y_min = getattr(bbox_obj, "y_min", 0.0)
        x_max = getattr(bbox_obj, "x_max", 100.0)
        y_max = getattr(bbox_obj, "y_max", 100.0)
        normalized = getattr(bbox_obj, "normalized", False)

        return BoundingBox(
            x_min=float(x_min),
            y_min=float(y_min),
            x_max=float(x_max),
            y_max=float(y_max),
            normalized=bool(normalized),
        )

    @classmethod
    def to_person_c_input(
        cls,
        b_response: DocumentProcessingResponse,
        file_bytes: Optional[bytes] = None,
        selected_state: Optional[str] = None,
    ) -> Tuple[DocumentOCRResult, Optional[HandwritingResult]]:
        """Transforms Person B's DocumentProcessingResponse into Person C's expected inputs:

        1. DocumentOCRResult (printed lines, tables, layout metadata, full text)
        2. HandwritingResult (handwritten regions recognized via TrOCR)
        """
        doc_id = b_response.document_id or f"doc_{int(time.time() * 1000)}"

        # Compute SHA-256 hash
        if file_bytes:
            sha256_hash = hashlib.sha256(file_bytes).hexdigest()
        else:
            sha256_hash = hashlib.sha256(b_response.merged_text.encode("utf-8", errors="ignore")).hexdigest()

        # Extract preprocessing metadata
        first_region = b_response.ordered_regions[0] if b_response.ordered_regions else None
        prep_info = getattr(first_region, "preprocessing_metadata", {}) if first_region else {}
        
        orig_dims = prep_info.get("original_size", {})
        deskew_info = prep_info.get("deskew", {})
        
        preprocessing = PreprocessingMetadata(
            deskew_angle=float(deskew_info.get("detected_skew_angle", 0.0)),
            super_resolution_applied=False,
            denoised=bool(prep_info.get("denoise")),
            contrast_enhanced=bool(prep_info.get("contrast")),
            original_resolution=(orig_dims.get("width", 1800), orig_dims.get("height", 2400)) if orig_dims else None,
            processed_resolution=None,
            rotation_degrees=int(round(deskew_info.get("rotation_applied_deg", 0.0))),
            page_count=1,
        )

        # Detect jurisdiction / state and classification
        raw_full_text = b_response.merged_text or ""
        loader = ConfigLoader()
        detected_state = selected_state or loader.detect_state_from_text(raw_full_text)
        state_upper = (detected_state or "KA").upper()

        # Intelligent Content-Based Document Classification
        from src.extraction.document_classifier import classify_land_document
        class_res = classify_land_document(raw_full_text)
        doc_type = class_res["document_type"]

        classification = DocumentClassificationResult(
            document_type=doc_type,
            state=detected_state or "KA",
            confidence=float(class_res.get("confidence", 0.92)),
            signals_matched=["content_based_classifier", class_res.get("document_type_label", "Land Record")],
            language="kn" if state_upper in ("KA", "KARNATAKA") or any('\u0c80' <= c <= '\u0cff' for c in raw_full_text) else "en",
        )

        # Partition regions into printed OCR lines and handwritten regions
        text_lines: List[OCRTextLine] = []
        hw_regions: List[HandwritingRegionResult] = []

        for idx, reg in enumerate(b_response.ordered_regions):
            bbox = cls._convert_bbox(reg.bbox)
            conf = float(reg.confidence) if reg.confidence is not None else 0.85
            text_val = reg.normalized_text or reg.raw_text or ""

            if not text_val.strip():
                continue

            # Check if region was identified as handwritten
            if reg.is_handwritten is True or (reg.model_name and "trocr" in reg.model_name.lower()):
                hw_regions.append(
                    HandwritingRegionResult(
                        region_id=reg.region_id or f"hw_{idx}",
                        text=text_val,
                        confidence=conf,
                        page_number=reg.page_number or b_response.page_number,
                        bbox=bbox,
                        model_version=reg.model_version or reg.model_name or "trocr-kannada-handwritten",
                        source_region="handwriting_fill",
                    )
                )

            # Printed line representation (also include all text lines for complete field extraction)
            text_lines.append(
                OCRTextLine(
                    text=text_val,
                    confidence=conf,
                    bbox=bbox,
                    page_number=reg.page_number or b_response.page_number,
                    language=reg.language or "kn",
                    engine=OCREngineType.TROCR if reg.is_handwritten else OCREngineType.PADDLE_OCR,
                    source_region="table_cell" if getattr(reg, "status", None) == "table_cell" else "body",
                )
            )

        ocr_result = DocumentOCRResult(
            document_id=doc_id,
            sha256_hash=sha256_hash,
            preprocessing=preprocessing,
            classification=classification,
            text_lines=text_lines,
            tables=[],
            raw_full_text=raw_full_text,
            pages_processed=1,
            processing_time_ms=b_response.processing_time_ms,
        )

        handwriting_result = (
            HandwritingResult(
                document_id=doc_id,
                regions=hw_regions,
                model_version="trocr-kannada-v1",
                processing_time_ms=0.0,
            )
            if hw_regions
            else None
        )

        return ocr_result, handwriting_result

    @classmethod
    def execute_person_c(
        cls,
        b_response: DocumentProcessingResponse,
        file_bytes: Optional[bytes] = None,
        selected_state: Optional[str] = None,
        db_session: Optional[Session] = None,
    ) -> FinalDocumentResult:
        """Translates Person B output and invokes Person C extract_and_validate pipeline."""
        ocr_result, handwriting_result = cls.to_person_c_input(
            b_response=b_response,
            file_bytes=file_bytes,
            selected_state=selected_state,
        )

        return extract_and_validate(
            ocr_result=ocr_result,
            handwriting_result=handwriting_result,
            selected_state=selected_state or ocr_result.classification.state,
            db_session=db_session,
            file_bytes=file_bytes,
        )

    @classmethod
    def to_unified_response(
        cls,
        b_response: DocumentProcessingResponse,
        c_result: FinalDocumentResult,
    ) -> Dict[str, Any]:
        """Synthesizes Person B's OCR output and Person C's FinalDocumentResult into a single,
        backwards-compatible dictionary response.
        """
        # Formulate structured fields dictionary
        fields_dict: Dict[str, Any] = {}
        for fname, fval in c_result.fields.items():
            fields_dict[fname] = {
                "field_name": fval.field_name,
                "raw_value": fval.raw_value,
                "normalized_value": fval.normalized_value,
                "raw_unit": fval.raw_unit,
                "normalized_unit": fval.normalized_unit,
                "confidence": round(fval.confidence, 4),
                "page": fval.page,
                "bbox": fval.bbox.model_dump() if fval.bbox else None,
                "validation_status": fval.validation_status.value,
                "validation_messages": fval.validation_messages,
                "evidence": fval.evidence.model_dump() if fval.evidence else None,
            }

        # Provide aliases for Karnataka / South Indian and standard frontend field names
        if "khasra_number" in fields_dict and "survey_number" not in fields_dict:
            survey_copy = dict(fields_dict["khasra_number"])
            survey_copy["field_name"] = "survey_number"
            fields_dict["survey_number"] = survey_copy

        if "khatauni_number" in fields_dict and "property_number" not in fields_dict:
            prop_copy = dict(fields_dict["khatauni_number"])
            prop_copy["field_name"] = "property_number"
            fields_dict["property_number"] = prop_copy

        if "tehsil" in fields_dict and "taluk" not in fields_dict:
            taluk_copy = dict(fields_dict["tehsil"])
            taluk_copy["field_name"] = "taluk"
            fields_dict["taluk"] = taluk_copy

        if "document_date" in fields_dict and "date" not in fields_dict:
            date_copy = dict(fields_dict["document_date"])
            date_copy["field_name"] = "date"
            fields_dict["date"] = date_copy


        combined_review = bool(b_response.requires_human_review or c_result.requires_human_review)
        
        # Combine review reasons uniquely
        seen_reasons = set()
        merged_reasons = []
        for r in list(b_response.warnings) + list(c_result.review_reasons):
            if r and r not in seen_reasons:
                seen_reasons.add(r)
                merged_reasons.append(r)

        return {
            # Core Document Identifiers
            "document_id": c_result.document_id,
            "sha256_hash": c_result.sha256_hash,
            "state": c_result.state,
            "document_type": c_result.document_type.value,
            "page_number": b_response.page_number,
            "image_path": b_response.image_path,

            # Person C Structured Extractions
            "extracted_fields": fields_dict,
            "tables": [t.model_dump() for t in c_result.tables],

            # Person C Validation & Quality Reports
            "validation": {
                "status": c_result.validation_status.value,
                "errors": [e.model_dump() for e in c_result.validation_errors],
                "cross_record": c_result.cross_record_validation.model_dump(),
            },
            "gis_validation": c_result.gis_validation.model_dump(),
            "duplicate_analysis": c_result.duplicate_analysis.model_dump(),

            # Quality & Confidence
            "overall_confidence": round(float(c_result.overall_confidence), 4),
            "requires_human_review": combined_review,
            "review_reasons": merged_reasons,
            "pipeline_stages_completed": c_result.pipeline_stages_completed,

            # Person B OCR Evidence Preservation (Full Backward Compatibility)
            "ocr": {
                "merged_text": b_response.merged_text,
                "document_confidence": b_response.document_confidence,
                "ordered_regions": [r.model_dump() for r in b_response.ordered_regions],
                "status": b_response.status,
                "warnings": b_response.warnings,
                "engine_breakdown": b_response.engine_breakdown,
                "processing_time_ms": b_response.processing_time_ms,
            },

            # Direct backward-compatible top-level keys
            "merged_text": b_response.merged_text,
            "document_confidence": b_response.document_confidence,
            "ordered_regions": [r.model_dump() for r in b_response.ordered_regions],
            "status": c_result.validation_status.value if not combined_review else "flagged_for_review",
            "warnings": merged_reasons,
            "processing_time_ms": round(b_response.processing_time_ms + c_result.processing_time_total_ms, 2),
        }
