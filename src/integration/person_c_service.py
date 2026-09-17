"""
src/integration/person_c_service.py
Primary interface for Person C: Extraction, Validation, GIS, Duplicates, and Confidence.
Integrates outputs from Person A (DocumentOCRResult) and Person B (HandwritingResult).
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from sqlalchemy.orm import Session

from schemas import (
    DocumentOCRResult,
    DocumentType,
    ExtractedField,
    FinalDocumentResult,
    HandwritingResult,
    ValidationStatus,
)
from src.confidence.scorer import ConfidenceScorer
from src.database.duplicates import DuplicateDetector
from src.database.gis import GISValidator
from src.extraction.disagreement import DisagreementResolver
from src.extraction.extractor import FieldExtractor
from src.utils.config_loader import ConfigLoader
from src.utils.logger import PipelineLogger
from src.validation.cross_record import CrossRecordValidator
from src.validation.normalization import FieldNormalizer
from src.validation.rules import RuleValidator


def extract_and_validate(
    ocr_result: DocumentOCRResult,
    handwriting_result: Optional[HandwritingResult] = None,
    selected_state: Optional[str] = None,
    gis_validator: Optional[GISValidator] = None,
    duplicate_detector: Optional[DuplicateDetector] = None,
    config_loader: Optional[ConfigLoader] = None,
    db_session: Optional[Session] = None,
    file_bytes: Optional[bytes] = None,
) -> FinalDocumentResult:
    """
    Person C Main Interface.
    Accepts OCR and Handwriting outputs, performs extraction, normalization, consensus resolution,
    rule & cross-record validation, GIS cadastral verification, duplicate analysis, and field-level confidence scoring.

    If db_session is passed and bound to a real PostgreSQL/PostGIS+pgvector database,
    GIS validation and duplicate detection use real ST_Contains/pgvector SQL queries.
    Otherwise (no session, or a SQLite dev database) both fall back to the equivalent
    in-Python Shapely/TF-IDF logic - same results shape, not a live spatial-DB query.
    """
    start_time = time.perf_counter()
    doc_id = ocr_result.document_id
    pipeline_logger = PipelineLogger(doc_id)
    stages_completed: List[str] = ["ocr_received"]

    # 1. Load State & Pipeline Configurations
    loader = config_loader or ConfigLoader()
    pipeline_cfg = loader.get_pipeline_config()

    # Determine State: explicit selection -> classification state -> auto-detection from text
    target_state = selected_state
    if not target_state and ocr_result.classification and ocr_result.classification.state != "UNKNOWN":
        target_state = ocr_result.classification.state
    if not target_state:
        target_state = loader.detect_state_from_text(ocr_result.raw_full_text)

    state_config = loader.get_state_config(target_state)
    actual_state_code = state_config.get("state_code", target_state or "DEFAULT")

    # Guard: If document is not a land record, halt extraction pipeline immediately
    if ocr_result.classification and (
        ocr_result.classification.document_type == DocumentType.NOT_LAND_RECORD
        or getattr(ocr_result.classification.document_type, "value", "") == "not_land_record"
    ):
        return FinalDocumentResult(
            document_id=doc_id,
            sha256_hash=ocr_result.sha256_hash or "not_land_record",
            state=actual_state_code,
            document_type=DocumentType.NOT_LAND_RECORD,
            fields={},
            overall_confidence=0.95,
            validation_status=ValidationStatus.INVALID,
            review_reasons=["This document does not appear to be a land record."],
            requires_human_review=True,
            pipeline_stages_completed=["classification"],
            processing_time_total_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
        )

    # 2. Semantic Understanding & Structured Field Extraction
    from src.extraction.semantic_understanding import semantic_understand, fuse_semantic_and_regex_fields

    raw_text_content = ocr_result.raw_full_text or "\n".join(l.text for l in ocr_result.text_lines)
    semantic_result = semantic_understand(
        ordered_ocr_text=raw_text_content,
        regions=ocr_result.text_lines,
        document_context={
            "document_id": doc_id,
            "state": actual_state_code,
            "classification": ocr_result.classification.model_dump() if ocr_result.classification else {},
        },
        image_bytes=file_bytes,
    )
    stages_completed.append("semantic_understanding")

    # Step 2b: Run Existing Person C Rule & Regex Extraction
    extractor = FieldExtractor(state_config)
    regex_fields = extractor.extract_all(ocr_result, handwriting_result)

    # Step 2c: Fuse Semantic & Rule Extraction Candidates
    extracted_fields = fuse_semantic_and_regex_fields(
        semantic_result=semantic_result,
        regex_fields=regex_fields,
        ocr_result=ocr_result,
        state_config=state_config,
    )
    stages_completed.append("extraction")
    pipeline_logger.extraction_completed(len(extracted_fields))

    # 3. Field Normalization (Names, Dates, Land Units, Khasra format)
    normalizer = FieldNormalizer(state_config)
    for field_name, field_obj in extracted_fields.items():
        if field_name in ("owner_name", "father_or_husband_name"):
            field_obj.normalized_value = normalizer.normalize_name(field_obj.raw_value)
        elif field_name in ("khasra_number", "survey_number", "khata_number", "property_number", "khatauni_number"):
            field_obj.normalized_value = normalizer.normalize_khasra(field_obj.raw_value)
        elif field_name in ("land_area", "site_area", "built_up_area"):
            norm_area, raw_unit, norm_unit = normalizer.normalize_land_area(
                field_obj.raw_value, field_obj.raw_unit
            )
            field_obj.raw_unit = raw_unit or field_obj.raw_unit or "Sq Ft"
            field_obj.normalized_unit = norm_unit
            # If raw value represents urban plot/construction with Sq Ft, preserve readable string
            raw_str = str(field_obj.raw_value).strip()
            if "sq" in raw_str.lower() or "ft" in raw_str.lower() or "ಅಡಿ" in raw_str:
                field_obj.normalized_value = raw_str
            elif ("site" in field_name or "built" in field_name) and re.match(r"^\d+(?:\.\d+)?$", raw_str):
                field_obj.normalized_value = f"{raw_str} Sq Ft"
            else:
                field_obj.normalized_value = norm_area
        elif field_name in ("locality", "village"):
            field_obj.normalized_value = normalizer.normalize_locality(field_obj.raw_value)
        elif field_name in ("taluk", "tehsil", "sub_division"):
            field_obj.normalized_value = normalizer.normalize_taluk(field_obj.raw_value)
        elif field_name == "district":
            field_obj.normalized_value = normalizer.normalize_district(field_obj.raw_value)
        elif field_name in ("document_date", "date"):
            norm_date = normalizer.normalize_date(field_obj.raw_value)
            field_obj.normalized_value = norm_date or field_obj.raw_value.strip()
        elif field_name == "fasli_year":
            norm_fasli, _ = normalizer.normalize_fasli_year(field_obj.raw_value)
            field_obj.normalized_value = norm_fasli
        else:
            field_obj.normalized_value = field_obj.raw_value.strip()

    stages_completed.append("normalization")

    # 4. OCR Disagreement Resolution (Printed OCR vs Handwriting OCR)
    disagreement_resolver = DisagreementResolver()
    extracted_fields, disagreement_log = disagreement_resolver.resolve_conflicts(
        extracted_fields=extracted_fields,
        handwriting_result=handwriting_result,
        state_config=state_config,
    )
    if disagreement_log:
        stages_completed.append("disagreement_resolution")

    # 5. Single-Field Rule Validation
    rule_validator = RuleValidator(state_config)
    validation_items, aggregate_rule_status = rule_validator.validate_fields(extracted_fields)
    stages_completed.append("rule_validation")

    # 6. Cross-Record Validation (Area sums, owner shares, table consistency)
    cross_validator = CrossRecordValidator()
    cross_record_result = cross_validator.validate_cross_record(
        fields=extracted_fields,
        tables=ocr_result.tables,
        state_config=state_config,
    )
    stages_completed.append("cross_record_validation")

    # 7. Cadastral GIS Spatial Validation
    gis = gis_validator or GISValidator(
        area_mismatch_threshold_percent=pipeline_cfg.get("gis_settings", {}).get("area_mismatch_threshold_percent", 10.0)
    )
    gis_result = gis.validate_gis(extracted_fields, state_code=actual_state_code, db_session=db_session)
    stages_completed.append("gis_validation")

    # 8. Duplicate Detection (SHA-256 and Semantic Vector Similarity)
    dup_detector = duplicate_detector or DuplicateDetector(
        vector_similarity_threshold=pipeline_cfg.get("duplicate_settings", {}).get("vector_similarity_threshold", 0.85)
    )
    dup_result = dup_detector.check_duplicate(
        document_id=doc_id,
        sha256_hash=ocr_result.sha256_hash,
        full_text=ocr_result.raw_full_text,
        fields=extracted_fields,
        db_session=db_session,
    )
    # Register document for future duplicate checks
    dup_detector.register_document(
        document_id=doc_id,
        sha256_hash=ocr_result.sha256_hash,
        full_text=ocr_result.raw_full_text,
        fields=extracted_fields,
        db_session=db_session,
    )
    stages_completed.append("duplicate_analysis")

    # 9. Field-Level & Document-Level Confidence Scoring
    confidence_scorer = ConfidenceScorer(
        weights=pipeline_cfg.get("confidence_weights"),
        high_threshold=pipeline_cfg.get("confidence_thresholds", {}).get("high_confidence_auto_approve", 0.85),
        medium_threshold=pipeline_cfg.get("confidence_thresholds", {}).get("medium_confidence_flag", 0.65),
    )
    overall_conf, requires_review, review_reasons = confidence_scorer.evaluate_document_confidence(
        fields=extracted_fields,
        validation_items=validation_items,
        cross_record=cross_record_result,
        gis_result=gis_result,
        duplicate_result=dup_result,
    )
    stages_completed.append("confidence_scoring")

    # Determine final document classification type
    doc_type = ocr_result.classification.document_type if ocr_result.classification else DocumentType.UNKNOWN

    # Determine overall document validation status
    if aggregate_rule_status == ValidationStatus.INVALID or gis_result.has_mismatch:
        final_val_status = ValidationStatus.INVALID
    elif aggregate_rule_status == ValidationStatus.WARNING or not cross_record_result.passed or dup_result.is_duplicate:
        final_val_status = ValidationStatus.WARNING
    else:
        final_val_status = ValidationStatus.VALID

    pipeline_logger.validation_completed(final_val_status.value, len(validation_items))
    pipeline_logger.confidence_calculated(overall_conf, requires_review)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    pipeline_logger.final_output_generated(elapsed_ms)

    return FinalDocumentResult(
        document_id=doc_id,
        sha256_hash=ocr_result.sha256_hash,
        state=actual_state_code,
        document_type=doc_type,
        fields=extracted_fields,
        tables=ocr_result.tables,
        validation_status=final_val_status,
        validation_errors=validation_items,
        cross_record_validation=cross_record_result,
        gis_validation=gis_result,
        duplicate_analysis=dup_result,
        overall_confidence=overall_conf,
        requires_human_review=requires_review,
        review_reasons=review_reasons,
        pipeline_stages_completed=stages_completed,
        disagreements_resolved=disagreement_log,
        processing_time_total_ms=round(elapsed_ms, 2),
    )
