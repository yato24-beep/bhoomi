"""Semantic Pipeline Orchestrator for Land Record Documents.

Orchestrates the entire non-training semantic understanding flow:
1. Conservative Unicode & whitespace normalization
2. Reading order preservation and sequential diagnostics
3. Field label anchor detection (Kannada & English)
4. 2D spatial key-value association
5. Field-specific deterministic validation
6. Tabular grid reconstruction with cell-to-region linkage
7. Full provenance tracking back to immutable OCR evidence
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from schemas import BoundingBox
from src.semantic.field_detector import FieldDetector
from src.semantic.normalizer import SemanticNormalizer
from src.semantic.schema import (
    BoundaryRecord,
    CadastralRecord,
    LandRecordDocument,
    OwnerRecord,
    SemanticFieldItem,
    SemanticTable,
    ValidationStatus,
)
from src.semantic.table_reconstructor import TableReconstructor
from src.semantic.validator import FieldValidator

logger = logging.getLogger(__name__)

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from schemas import BoundingBox, DocumentType
from src.semantic.confidence import EvidenceConfidenceCalculator
from src.semantic.engine import (
    BaseSemanticEngine,
    GeminiSemanticEngine,
    RuleSemanticEngine,
    SemanticEngineResult,
    SemanticEvidence,
)
from src.semantic.field_detector import FieldDetector
from src.semantic.normalizer import SemanticNormalizer
from src.semantic.regional_aliases import resolve_field_alias
from src.semantic.schema import (
    BoundaryRecord,
    CadastralRecord,
    FieldProvenance,
    LandRecordDocument,
    OwnerRecord,
    SemanticFieldItem,
    SemanticTable,
    ValidationStatus,
)
from src.semantic.table_reconstructor import TableReconstructor
from src.semantic.validator import FieldValidator

logger = logging.getLogger(__name__)


class SemanticPipeline:
    """Production semantic pipeline coordinating normalization, LLM reasoning, and validation."""

    def __init__(
        self,
        semantic_engine: Optional[BaseSemanticEngine] = None,
        confidence_threshold: float = 0.60,
    ):
        self.normalizer = SemanticNormalizer(convert_digits=True)
        self.field_detector = FieldDetector()
        self.validator = FieldValidator()
        self.table_reconstructor = TableReconstructor()
        self.confidence_calculator = EvidenceConfidenceCalculator()
        self.confidence_threshold = confidence_threshold

        # Initialize semantic reasoning engine
        if semantic_engine:
            self.semantic_engine = semantic_engine
        else:
            engine_choice = os.getenv("SEMANTIC_ENGINE", "gemini").lower()
            if engine_choice == "gemini":
                self.semantic_engine = GeminiSemanticEngine()
            else:
                self.semantic_engine = RuleSemanticEngine()

    def process(
        self,
        regions: Sequence[Any],
        document_id: Optional[str] = None,
        page_number: int = 1,
        document_type: str = "Unknown / Not classified",
        table_bounding_boxes: Optional[List[BoundingBox]] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        ner_candidates: Optional[Dict[str, Any]] = None,
        is_cadastral: Optional[bool] = None,
        classification_result: Optional[Any] = None,
        **kwargs: Any,
    ) -> LandRecordDocument:
        """Processes OCR regions into an auditable structured land record semantic document.

        Args:
            regions: Ordered list of RecognizedRegionResult or OCRRegion objects.
            document_id: Document canonical identifier.
            page_number: 1-indexed document page number.
            document_type: Classified document type (default "Unknown / Not classified").
            table_bounding_boxes: Optional bounding boxes of detected table regions.
            ner_candidates: Optional dictionary of upstream NER candidate extractions.
            is_cadastral: Boolean flag indicating if document is a cadastral land record.
            classification_result: Upstream DocumentClassificationResult or classification dictionary.

        Returns:
            LandRecordDocument with structured fields, tables, validation reports, and provenance.
        """
        # Step 0: Pre-Semantic Cadastral Gate
        effective_is_cadastral = is_cadastral
        if effective_is_cadastral is None and classification_result is not None:
            if isinstance(classification_result, dict):
                effective_is_cadastral = classification_result.get(
                    "is_cadastral",
                    classification_result.get("is_land_record", True),
                )
                if classification_result.get("document_type") in (
                    DocumentType.NOT_LAND_RECORD,
                    "not_land_record",
                    "not_a_land_record",
                ):
                    effective_is_cadastral = False
            elif hasattr(classification_result, "document_type"):
                effective_is_cadastral = (
                    classification_result.document_type != DocumentType.NOT_LAND_RECORD
                )

        if effective_is_cadastral is None:
            doc_type_lower = str(document_type).lower().strip()
            if doc_type_lower in (
                "not_land_record",
                "not a land record",
                "non-cadastral",
                "non_cadastral",
                "non-cadastral narrative",
            ):
                effective_is_cadastral = False

        if effective_is_cadastral is False:
            logger.info(
                f"[{document_id}] Pre-semantic cadastral gate: document is non-cadastral (is_cadastral=False). Suppressing cadastral extraction."
            )
            return LandRecordDocument(
                document_id=document_id,
                document_type="Not a land record",
                fields={},
                extra_fields={},
                validation_summary={
                    "gate_status": "cadastral_extraction_suppressed",
                    "reason": "Pre-semantic cadastral gate suppressed extraction because document was classified as non-cadastral",
                    "is_cadastral": False,
                    "total_extracted_fields": 0,
                    "valid_fields_count": 0,
                    "warning_fields_count": 0,
                    "invalid_fields_count": 0,
                    "overall_semantic_valid": True,
                },
            )

        # Step 1: Normalize all region texts non-destructively
        for reg in regions:
            raw_t = getattr(reg, "raw_text", getattr(reg, "text", ""))
            norm_res = self.normalizer.normalize(raw_t)
            if not getattr(reg, "normalized_text", None):
                setattr(reg, "normalized_text", norm_res.normalized_ocr_text)

        # Step 2: Reading Order Verification & Diagnostics
        sorted_regions = list(regions)
        sorted_regions.sort(
            key=lambda r: (
                round(getattr(r.bbox, "y_min", 0.0) / 25.0) * 25.0 if getattr(r, "bbox", None) else 0.0,
                getattr(r.bbox, "x_min", 0.0) if getattr(r, "bbox", None) else 0.0,
            )
        )

        # Step 3: Table Reconstruction
        reconstructed_tables: List[SemanticTable] = []
        if table_bounding_boxes:
            for t_idx, t_box in enumerate(table_bounding_boxes):
                t_obj = self.table_reconstructor.reconstruct_table(
                    regions=sorted_regions,
                    table_bbox=t_box,
                    table_id=f"table_{t_idx}",
                    page_number=page_number,
                )
                reconstructed_tables.append(t_obj)

        # Region lookup by region_id
        region_by_id: Dict[str, Any] = {
            getattr(r, "region_id", ""): r for r in sorted_regions if getattr(r, "region_id", None)
        }
        all_ocr_texts = [getattr(r, "raw_text", getattr(r, "text", "")) for r in sorted_regions]

        # Step 4: Semantic AI Reasoning via SemanticEngine
        evidence = SemanticEvidence(
            document_id=document_id,
            page_number=page_number,
            document_type=document_type,
            raw_text=" ".join(all_ocr_texts),
            regions=sorted_regions,
            tables=reconstructed_tables,
            ner_candidates=ner_candidates or {},
        )

        engine_result: SemanticEngineResult = self.semantic_engine.extract(evidence)

        # Fallback to deterministic spatial association if AI engine reported error
        fallback_used = False
        if engine_result.status == "error":
            logger.info(f"AI SemanticEngine reported error: {engine_result.error}. Falling back to RuleSemanticEngine.")
            rule_engine = RuleSemanticEngine()
            engine_result = rule_engine.extract(evidence)
            fallback_used = True

        # Step 5: Field Packaging, Provenance Linking, Deterministic Validation & Calibration
        validated_fields: Dict[str, SemanticFieldItem] = {}

        for fname, ext_field in engine_result.fields.items():
            if not ext_field.raw_value or not ext_field.raw_value.strip():
                continue

            # Resolve regional aliases into Karnataka canonical schema
            canonical_fname, source_region = resolve_field_alias(fname)

            # Link provenance back to source OCR region
            src_reg_id = ext_field.source_region_id or ""
            src_reg = region_by_id.get(src_reg_id)
            if not src_reg and sorted_regions:
                # Find matching region containing raw_value if source_region_id was omitted or imprecise
                for r in sorted_regions:
                    r_text = getattr(r, "raw_text", getattr(r, "text", ""))
                    if ext_field.raw_value in r_text:
                        src_reg = r
                        src_reg_id = getattr(r, "region_id", "")
                        break

            src_bbox = getattr(src_reg, "bbox", None) if src_reg else None
            src_raw_text = getattr(src_reg, "raw_text", getattr(src_reg, "text", ext_field.raw_value)) if src_reg else ext_field.raw_value

            prov = FieldProvenance(
                document_id=document_id,
                page_number=page_number,
                region_id=src_reg_id or "unknown_region",
                bbox=src_bbox,
                raw_ocr_text=src_raw_text,
                normalized_value=ext_field.normalized_value or ext_field.raw_value,
                extraction_method="gemini_semantic" if not fallback_used else "rule_spatial",
            )

            item = SemanticFieldItem(
                field_name=canonical_fname,
                value=ext_field.normalized_value or ext_field.raw_value,
                raw_value=ext_field.raw_value,
                source_region_ids=[src_reg_id] if src_reg_id else [],
                provenance=prov,
                model_confidence=ext_field.model_confidence,
                conflicts=ext_field.conflict_candidates,
            )

            # Deterministic domain & grounding validation
            validated_item = self.validator.validate_field(item, ocr_texts=all_ocr_texts)

            # Evidence-based engineering confidence scoring
            has_conflicts = bool(validated_item.conflicts and len(validated_item.conflicts) > 0)
            conf_score = self.confidence_calculator.calculate_field_confidence(
                field_name=canonical_fname,
                raw_value=validated_item.raw_value,
                normalized_value=validated_item.value,
                provenance=validated_item.provenance,
                validation_status=validated_item.validation_status,
                has_conflicts=has_conflicts,
                model_reported_confidence=ext_field.model_confidence,
            )
            validated_item.confidence = conf_score

            # Trigger warning if confidence is below threshold
            if conf_score < self.confidence_threshold and validated_item.validation_status == ValidationStatus.VALID:
                validated_item.validation_status = ValidationStatus.WARNING
                validated_item.validation_reason = f"Confidence {conf_score:.2f} below threshold {self.confidence_threshold:.2f}"

            # Merge or resolve duplicate canonical fields
            if canonical_fname in validated_fields:
                existing_item = validated_fields[canonical_fname]
                if validated_item.value != existing_item.value:
                    if validated_item.value not in existing_item.conflicts:
                        existing_item.conflicts.append(validated_item.value)
                    existing_item.confidence = self.confidence_calculator.calculate_field_confidence(
                        field_name=canonical_fname,
                        raw_value=existing_item.raw_value,
                        normalized_value=existing_item.value,
                        provenance=existing_item.provenance,
                        validation_status=existing_item.validation_status,
                        has_conflicts=True,
                        model_reported_confidence=existing_item.model_confidence,
                    )
            else:
                validated_fields[canonical_fname] = validated_item

        # Step 6: Assemble repeated records and canonical schema
        owners: List[OwnerRecord] = []
        owner_item = validated_fields.get("owner_name")
        father_item = validated_fields.get("father_name")
        if owner_item:
            owners.append(OwnerRecord(owner_name=owner_item, father_name=father_item))

        cadastral_list: List[CadastralRecord] = []
        if "survey_number" in validated_fields:
            cad_rec = CadastralRecord(
                survey_number=validated_fields["survey_number"],
                hissa_number=validated_fields.get("hissa_number"),
                khata_number=validated_fields.get("khata_number"),
                extent=validated_fields.get("extent"),
                land_type=validated_fields.get("land_type"),
                assessment=validated_fields.get("assessment"),
            )
            cadastral_list.append(cad_rec)

        # Validation summary metrics
        total_fields = len(validated_fields)
        valid_count = sum(1 for f in validated_fields.values() if f.validation_status == ValidationStatus.VALID)
        warning_count = sum(1 for f in validated_fields.values() if f.validation_status == ValidationStatus.WARNING)
        invalid_count = sum(1 for f in validated_fields.values() if f.validation_status == ValidationStatus.INVALID)

        val_summary = {
            "total_extracted_fields": total_fields,
            "valid_fields_count": valid_count,
            "warning_fields_count": warning_count,
            "invalid_fields_count": invalid_count,
            "overall_semantic_valid": (invalid_count == 0 and total_fields > 0),
            "fallback_used": fallback_used,
            "engine_used": self.semantic_engine.__class__.__name__,
        }

        # Structured diagnostic logging gated by environment flag (no secrets printed)
        enable_diag = os.getenv("SEMANTIC_DEBUG", "true").lower() in ("true", "1", "yes") or os.getenv("ENABLE_SEMANTIC_DIAGNOSTICS", "true").lower() in ("true", "1", "yes")
        if enable_diag:
            engine_name = self.semantic_engine.__class__.__name__
            model_name = getattr(self.semantic_engine, "model_name", "none")
            was_gemini_called = (engine_name == "GeminiSemanticEngine" and not fallback_used and engine_result.status == "success")
            grounded_count = sum(1 for f in validated_fields.values() if f.provenance and f.provenance.region_id and f.provenance.region_id != "unknown_region")
            grounding_res = f"PASS ({grounded_count}/{total_fields} grounded)" if total_fields > 0 else "NO_FIELDS"
            val_res = "PASS" if invalid_count == 0 else f"FLAGGED ({invalid_count} invalid, {warning_count} warning)"
            fb_status = "FALLBACK_RULE_ENGINE" if fallback_used else "NONE"
            diag_str = (
                f"[SEMANTIC DIAGNOSTIC] engine={engine_name} model={model_name} "
                f"regions_sent={len(sorted_regions)} gemini_called={was_gemini_called} "
                f"fields_returned={total_fields} grounding={grounding_res} "
                f"validation={val_res} fallback={fb_status}"
            )
            logger.info(diag_str)
            print(diag_str)

        # Handle aliases (registration_number <-> document_number, record_date <-> registration_date)
        reg_num = validated_fields.get("registration_number") or validated_fields.get("document_number")
        rec_date = validated_fields.get("record_date") or validated_fields.get("registration_date")

        doc = LandRecordDocument(
            document_id=document_id,
            document_type=engine_result.document_type or document_type,
            district=validated_fields.get("district"),
            taluk=validated_fields.get("taluk"),
            hobli=validated_fields.get("hobli"),
            village=validated_fields.get("village"),
            survey_number=validated_fields.get("survey_number"),
            hissa_number=validated_fields.get("hissa_number"),
            khata_number=validated_fields.get("khata_number"),
            owner_name=owner_item,
            cultivator_name=validated_fields.get("cultivator_name"),
            extent=validated_fields.get("extent"),
            land_type=validated_fields.get("land_type"),
            assessment=validated_fields.get("assessment"),
            mutation_number=validated_fields.get("mutation_number"),
            registration_number=reg_num,
            document_number=reg_num,
            record_date=rec_date,
            registration_date=rec_date,
            remarks=validated_fields.get("remarks"),
            owners=owners,
            cadastral_records=cadastral_list,
            tables=reconstructed_tables,
            extra_fields=validated_fields,
            validation_summary=val_summary,
        )

        return doc

