"""Document Processing Orchestrator for Land Record Digitization.

Orchestrates preprocessing, layout region ingestion, script/handwriting routing,
reading-order text merging, confidence aggregation, non-destructive normalization,
and human-review flagging for downstream consumption by Person C.
"""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("document_pipeline")

import numpy as np
from PIL import Image

from schemas import BoundingBox, DocumentPage, OCRResult, ValidationStatus
from src.classification.document_gate import LandRecordGateClassifier, GateClassificationResult
from src.classification.style_classifier import ScriptStyleClassifier, get_style_classifier
from src.detection.text_detector import DocumentTextDetector, DetectedLineGroup, DetectedWordBox
from src.extraction.land_record_ner import LandRecordFieldExtractor
from src.extraction.tabular_ner import TabularLayoutExtractor
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput
from src.handwriting.router import LanguageScriptRouter
from src.handwriting.trocr_recognizer import get_iitb_kannada_recognizer
from src.integration.person_a_adapter import PersonAAdapter
from src.integration.schemas import (
    ConfidenceState,
    DocumentProcessingRequest,
    DocumentProcessingResponse,
    ProcessingStatus,
    RecognizedRegionResult,
    RegionRequest,
    RegionType,
)
from src.integration.review_state import (
    CRITICAL_LAND_RECORD_FIELDS,
    HumanReviewItem,
    ReviewDecision,
    ReviewStateMachine,
    ReviewStatus,
)
from src.integration.structured_schema import (
    OCRPage,
    OCRRegion,
    PageTimings,
    RegionType as SRegionType,
    ScriptType as SScriptType,
    StructuredDocumentOCR,
)
from src.confidence.calibration import MultiEngineCalibrator
from src.semantic.semantic_pipeline import SemanticPipeline
from src.postprocessing.kannada_normalizer import KannadaNormalizer
from src.preprocessing.format_adapter import DocumentFormatAdapter
from src.preprocessing.line_segmentation import DocumentLineSegmenter, LineCrop
from src.preprocessing.line_word_decomposer import (
    LineWordDecomposer,
    LineDecompositionResult,
    WordBoxCandidate,
)
from src.preprocessing.image_enhancement import (
    PreprocessingResult,
    load_image_as_pil,
    preprocess_document_image,
    detect_and_correct_coarse_orientation,
)
from src.translation.field_translator import FieldAlignedTranslator

# Type alias for caller convenience
RegionInput = Union[BoundingBox, RegionRequest, Dict[str, Any], Tuple[int, int, int, int]]


# Backwards-compatible alias / wrapper
DocumentProcessingResult = DocumentProcessingResponse


class DocumentProcessingPipeline:
    """Production orchestrator combining visual layout gating, text detection, routing, OCR, normalization, NER, and field-aligned translation."""

    def __init__(
        self,
        router: Optional[LanguageScriptRouter] = None,
        normalizer: Optional[KannadaNormalizer] = None,
        segmenter: Optional[DocumentLineSegmenter] = None,
        gate_classifier: Optional[LandRecordGateClassifier] = None,
        text_detector: Optional[DocumentTextDetector] = None,
        style_classifier: Optional[ScriptStyleClassifier] = None,
        tabular_extractor: Optional[TabularLayoutExtractor] = None,
        format_adapter: Optional[DocumentFormatAdapter] = None,
        field_extractor: Optional[LandRecordFieldExtractor] = None,
        field_translator: Optional[FieldAlignedTranslator] = None,
        line_decomposer: Optional[LineWordDecomposer] = None,
        enable_line_word_decomposition: Optional[bool] = None,
        semantic_pipeline: Optional[SemanticPipeline] = None,
        calibrator: Optional[MultiEngineCalibrator] = None,
        default_language: str = "kannada",
        confidence_threshold: float = 0.60,
        apply_preprocessing: bool = True,
        apply_normalization: bool = True,
        apply_segmentation: Optional[bool] = None,
        enable_document_gating: bool = True,
    ):
        """Initializes the document pipeline."""
        self.router = router if router is not None else LanguageScriptRouter(default_language=default_language, auto_register_defaults=True)
        self.normalizer = normalizer if normalizer is not None else KannadaNormalizer()
        self.segmenter = segmenter if segmenter is not None else DocumentLineSegmenter()
        self.gate_classifier = gate_classifier if gate_classifier is not None else LandRecordGateClassifier()
        self.text_detector = text_detector if text_detector is not None else DocumentTextDetector()
        self.style_classifier = style_classifier if style_classifier is not None else get_style_classifier()
        self.tabular_extractor = tabular_extractor if tabular_extractor is not None else TabularLayoutExtractor()
        self.format_adapter = format_adapter if format_adapter is not None else DocumentFormatAdapter()
        self.field_extractor = field_extractor if field_extractor is not None else LandRecordFieldExtractor(confidence_threshold=confidence_threshold)
        self.field_translator = field_translator if field_translator is not None else FieldAlignedTranslator()
        self.line_decomposer = line_decomposer if line_decomposer is not None else LineWordDecomposer()
        self.semantic_pipeline = semantic_pipeline if semantic_pipeline is not None else SemanticPipeline()
        self.review_state_machine = ReviewStateMachine()
        if calibrator is not None:
            self.calibrator = calibrator
        else:
            cal_file = Path("models/calibration/multi_engine_calibrator.json")
            self.calibrator = MultiEngineCalibrator.load(cal_file) if cal_file.is_file() else MultiEngineCalibrator()
        if enable_line_word_decomposition is not None:
            self.enable_line_word_decomposition = enable_line_word_decomposition
        else:
            self.enable_line_word_decomposition = (
                os.environ.get("ENABLE_LINE_WORD_DECOMPOSITION", "false").strip().lower() == "true"
            )
        self._iitb_recognizer = None
        self.default_language = default_language
        self.confidence_threshold = confidence_threshold
        self.apply_preprocessing = apply_preprocessing
        self.apply_normalization = apply_normalization
        self.apply_segmentation = apply_segmentation
        self.enable_document_gating = enable_document_gating

    @property
    def iitb_recognizer(self):
        """Lazy singleton loader for IITB Kannada TrOCR v0.0.2 recognizer."""
        if self._iitb_recognizer is None:
            self._iitb_recognizer = get_iitb_kannada_recognizer(auto_load=True)
        return self._iitb_recognizer

    @staticmethod
    def _crop_region_image(image: Image.Image, bbox: Optional[BoundingBox]) -> Image.Image:
        """Crops sub-region from PIL image with boundary safety clamping."""
        if bbox is None:
            return image.copy()
        w, h = image.size
        x1 = max(0, min(bbox.x_min, w - 1))
        y1 = max(0, min(bbox.y_min, h - 1))
        x2 = max(x1 + 1, min(bbox.x_max, w))
        y2 = max(y1 + 1, min(bbox.y_max, h))
        return image.crop((x1, y1, x2, y2))

    @staticmethod
    def _compute_document_confidence(results: Sequence[RecognizedRegionResult]) -> Optional[float]:
        """Derives genuine document-level confidence from underlying character-weighted evidence.

        Zero fabrication: returns None if no valid confidence scores exist.
        """
        valid_items: List[Tuple[float, int]] = []
        for r in results:
            if r.confidence is not None and isinstance(r.confidence, (int, float)):
                char_weight = max(1, len(r.normalized_text.strip()))
                valid_items.append((float(r.confidence), char_weight))

        if not valid_items:
            return None

        total_weight = sum(w for _, w in valid_items)
        weighted_sum = sum(c * w for c, w in valid_items)
        return round(float(weighted_sum / total_weight), 4)

    def process_document(
        self,
        image: Optional[ImageInput] = None,
        request: Optional[DocumentProcessingRequest] = None,
        regions: Optional[Sequence[Any]] = None,
        is_handwritten: Optional[bool] = None,
        language: Optional[str] = None,
        script: Optional[str] = None,
        page_number: int = 1,
        document_id: Optional[str] = None,
        image_path: Optional[str] = None,
        apply_preprocessing: Optional[bool] = None,
        apply_normalization: Optional[bool] = None,
        apply_segmentation: Optional[bool] = None,
        document_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DocumentProcessingResponse:
        """Processes a full land record document or specified sub-regions.

        Accepts either a strongly typed DocumentProcessingRequest or flexible individual parameters.

        Args:
            image: Document image input (file path, PIL Image, numpy array, bytes).
            request: Optional pre-constructed DocumentProcessingRequest.
            regions: Optional candidate regions/bounding boxes (from Person A layout or caller).
            is_handwritten: Default text type classification (True=Handwritten, False=Printed).
            language: Default language for the document (defaults to pipeline default).
            script: Default script for the document.
            page_number: 1-indexed document page number.
            document_id: Optional document identifier.
            image_path: Optional string representation of the source image file path.
            apply_preprocessing: Whether to apply enhancement filters (defaults to pipeline config).
            apply_normalization: Whether to apply conservative text normalization.
            apply_segmentation: Whether to segment full images into line crops (defaults to apply_preprocessing).
            document_metadata: Optional dictionary of document-level metadata.
            **kwargs: Extra arguments forwarded to recognition engines.

        Returns:
            DocumentProcessingResponse containing structured regions, merged text,
            document confidence, and human review decision for Person C.
        """
        start_time = time.perf_counter()

        # Step 1: Normalize input into a DocumentProcessingRequest
        if request is not None:
            req = request
        else:
            if image is None:
                raise ValueError("Must provide either 'image' or 'request' argument to process_document.")
            req = PersonAAdapter.create_document_request(
                image=image,
                regions=regions,
                document_id=document_id,
                page_number=page_number,
                language=language or self.default_language,
                script=script or ("Kannada" if (language or self.default_language) in ("kannada", "kn") else "Latin"),
                is_handwritten=is_handwritten,
                document_metadata=document_metadata or {},
                apply_preprocessing=self.apply_preprocessing if apply_preprocessing is None else apply_preprocessing,
                apply_normalization=self.apply_normalization if apply_normalization is None else apply_normalization,
            )

        # Determine image path string
        resolved_img_path = (
            str(req.image)
            if isinstance(req.image, (str, Path))
            else (image_path or f"page_{req.page_number}.png")
        )

        # Telemetry tracking for each processing stage
        t_decode_start = time.perf_counter()
        t_render_start = t_decode_start
        try:
            pages = self.format_adapter.load_pages(req.image)
            t_decode_end = time.perf_counter()
            page_idx = max(0, min(req.page_number - 1, len(pages) - 1))
            raw_pil = pages[page_idx]
            t_render_end = time.perf_counter()
        except Exception as ingest_err:
            logger.warning("Format adapter fallback: %s", ingest_err)
            raw_pil = load_image_as_pil(req.image)
            t_decode_end = time.perf_counter()
            t_render_end = t_decode_end

        file_decode_ms = round((t_decode_end - t_decode_start) * 1000.0, 2)
        page_rendering_ms = round((t_render_end - t_render_start) * 1000.0, 2)
        orig_size = raw_pil.size

        # Step 2a: Document Orientation Detection & Correction (0, 90, 180, 270)
        doc_orientation_deg = 0
        if req.apply_preprocessing:
            try:
                raw_pil, doc_orientation_deg = detect_and_correct_coarse_orientation(raw_pil)
            except Exception as ori_err:
                logger.warning("Orientation correction notice: %s", ori_err)

        # Step 2b: Visual Land Record Layout Gate (Reject non-land-record images before heavy OCR)
        gate_result = None
        gate_ms = 0.0
        if self.enable_document_gating and (req.is_handwritten is not True) and (req.regions is None or len(req.regions) == 0):
            t_gate_start = time.perf_counter()
            gate_result = self.gate_classifier.classify_image(raw_pil)
            gate_ms = round((time.perf_counter() - t_gate_start) * 1000.0, 2)
            if not gate_result.is_land_record:
                total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                rejected_page = DocumentPage(
                    page_number=req.page_number,
                    image_path=resolved_img_path,
                    ocr_results=[],
                    metadata={
                        "rejected": True,
                        "gate_classification": gate_result.to_dict(),
                        "confidence": 0.0,
                        "is_handwritten": is_handwritten,
                        "language": language or self.default_language,
                        "script": script or "Kannada",
                    },
                )
                return DocumentProcessingResponse(
                    document_id=req.document_id,
                    page_number=req.page_number,
                    image_path=resolved_img_path,
                    ordered_regions=[],
                    merged_text="",
                    original_ocr="",
                    clean_kannada_text="",
                    kannada_translation="",
                    translated_text="",
                    english_translation="",
                    original_kannada_text="",
                    document_confidence=0.0,
                    recognition_confidence=0.0,
                    detection_confidence=0.0,
                    routing_confidence=0.0,
                    field_confidence=0.0,
                    verification_status="needs_verification",
                    status="rejected_not_land_record",
                    requires_human_review=True,
                    warnings=[f"REJECTED: {gate_result.rejection_reason}"],
                    engine_breakdown={},
                    processing_time_ms=total_time_ms,
                    page=rejected_page,
                    is_land_record=False,
                    gate_classification=gate_result.to_dict(),
                    extracted_fields={},
                    bilingual_fields={},
                    stage_timings={
                        "file_decode_ms": file_decode_ms,
                        "page_rendering_ms": page_rendering_ms,
                        "gate_classification_ms": gate_ms,
                        "layout_detection_ms": 0.0,
                        "crop_generation_ms": 0.0,
                        "ocr_inference_ms": 0.0,
                        "reconstruction_ms": 0.0,
                        "translation_ms": 0.0,
                        "ner_table_ms": 0.0,
                        "semantic_ms": 0.0,
                        "validation_ms": 0.0,
                        "total_ms": total_time_ms,
                    },
                )

        # Step 3: Preprocess document image
        prep_metadata: Dict[str, Any] = {}
        if req.apply_preprocessing:
            prep_res: PreprocessingResult = preprocess_document_image(
                image_input=raw_pil,
                apply_thresholding=False,
            )
            processed_pil = prep_res.image
            prep_metadata = dict(prep_res.audit_metadata)
            prep_metadata["document_orientation_degrees"] = doc_orientation_deg
        else:
            processed_pil = raw_pil.copy()
            prep_metadata = {
                "pipeline_steps": ["raw_passthrough"],
                "original_size": {"width": orig_size[0], "height": orig_size[1]},
                "document_orientation_degrees": doc_orientation_deg,
            }

        cur_w, cur_h = processed_pil.size

        # Resolve line segmentation setting:
        should_segment = (
            apply_segmentation
            if apply_segmentation is not None
            else (self.apply_segmentation if self.apply_segmentation is not None else (req.apply_preprocessing and req.is_handwritten is not False))
        )

        # Step 4: Decompose into regions (word boxes + reading order)
        t_layout_start = time.perf_counter()
        detected_line_groups: List[DetectedLineGroup] = []
        if req.regions and len(req.regions) > 0:
            adapted_regions = PersonAAdapter.adapt_layout_regions(
                regions=req.regions,
                image_size=(cur_w, cur_h),
                default_language=req.language,
                default_script=req.script,
                default_is_handwritten=req.is_handwritten,
            )
        else:
            try:
                detected_line_groups = self.text_detector.detect_lines_and_words(processed_pil)
            except Exception as det_err:
                logger.warning("Text detector notice: %s", det_err)
                detected_line_groups = []

            if detected_line_groups:
                adapted_regions = [
                    RegionRequest(
                        region_id=f"line_{lg.line_index:03d}",
                        bbox=lg.line_bbox,
                        language=req.language,
                        script=req.script or "Kannada",
                        is_handwritten=req.is_handwritten,
                        region_type=RegionType.TEXT,
                        metadata={
                            "word_boxes": [wb.bbox.model_dump() for wb in lg.word_boxes],
                            "reading_order_index": lg.line_index - 1,
                            "detection_confidence": getattr(lg, "line_confidence", 0.95),
                            "region_type_detected": "line",
                        },
                    )
                    for lg in detected_line_groups
                ]
            elif should_segment:
                try:
                    line_crops = self.segmenter.segment_into_lines(processed_pil, page_number=req.page_number)
                except Exception as seg_err:
                    line_crops = []
                if line_crops:
                    adapted_regions = [
                        RegionRequest(
                            region_id=lc.line_id,
                            bbox=lc.bbox,
                            language=lc.language,
                            script=lc.script,
                            is_handwritten=lc.is_handwritten,
                            region_type=RegionType.TEXT,
                            metadata={
                                "reading_order_index": lc.reading_order_index,
                                "region_type_detected": "line",
                            },
                        )
                        for lc in line_crops
                    ]
                else:
                    adapted_regions = [
                        RegionRequest(
                            region_id="page_001",
                            bbox=BoundingBox(x_min=0, y_min=0, x_max=cur_w, y_max=cur_h),
                            language=req.language,
                            script=req.script,
                            is_handwritten=False,  # Full page strictly barred from TrOCR!
                            region_type=RegionType.TEXT,
                            metadata={
                                "is_full_page": True,
                                "region_type_detected": "page",
                                "routing_reason": "Full-page image routed to EasyOCR; TrOCR strictly barred from full-page inputs",
                            },
                        )
                    ]
            else:
                adapted_regions = [
                    RegionRequest(
                        region_id="page_001",
                        bbox=BoundingBox(x_min=0, y_min=0, x_max=cur_w, y_max=cur_h),
                        language=req.language,
                        script=req.script,
                        is_handwritten=False,  # Full page strictly barred from TrOCR!
                        region_type=RegionType.TEXT,
                        metadata={
                            "is_full_page": True,
                            "region_type_detected": "page",
                            "routing_reason": "Full-page image routed to EasyOCR; TrOCR strictly barred from full-page inputs",
                        },
                    )
                ]

        layout_detection_ms = round((time.perf_counter() - t_layout_start) * 1000.0, 2)

        # Step 5: Run per-region style classification, batching, & automatic routing
        t_crop_start = time.perf_counter()
        recognized_regions: List[RecognizedRegionResult] = []
        ocr_results_compat: List[OCRResult] = []
        engine_breakdown: Dict[str, int] = {}
        review_warnings: List[str] = []

        # Batching optimization: Collect all handwritten Kannada word crops across all regions for a single GPU pass
        batched_word_crops: List[Tuple[int, int, Image.Image, BoundingBox]] = []
        region_crop_cache: Dict[int, Image.Image] = {}
        region_routing_decisions: Dict[int, Dict[str, Any]] = {}

        for reg_idx, region_req in enumerate(adapted_regions):
            reg_crop = self._crop_region_image(processed_pil, region_req.bbox)
            region_crop_cache[reg_idx] = reg_crop
            target_lang = region_req.language or req.language

            # Per-region script/style visual classification
            style_decision = self.style_classifier.classify_crop(reg_crop)
            routing_conf = style_decision.routing_confidence
            inferred_is_hw = style_decision.is_handwritten

            # Full-page safeguard: NEVER route full page to TrOCR
            is_full_page = (
                (region_req.bbox.x_max - region_req.bbox.x_min >= cur_w * 0.95)
                and (region_req.bbox.y_max - region_req.bbox.y_min >= cur_h * 0.95)
            )

            # Routing override precedence: caller request > region request > visual classifier
            if is_full_page:
                reg_is_hw = False
                routing_reason = "Full-page image routed to EasyOCR; TrOCR strictly barred from full-page inputs"
            elif req.is_handwritten is not None:
                reg_is_hw = req.is_handwritten
                engine_dest = "IITB TrOCR" if reg_is_hw else "EasyOCR"
                routing_reason = f"Caller override: is_handwritten={req.is_handwritten} -> Routed to {engine_dest}"
            elif region_req.is_handwritten is not None:
                reg_is_hw = region_req.is_handwritten
                engine_dest = "IITB TrOCR" if reg_is_hw else "EasyOCR"
                routing_reason = f"Region metadata override: is_handwritten={region_req.is_handwritten} -> Routed to {engine_dest}"
            else:
                reg_is_hw = inferred_is_hw
                engine_dest = "IITB TrOCR" if reg_is_hw else "EasyOCR"
                routing_reason = f"Style classifier decision: is_handwritten={inferred_is_hw} (conf={routing_conf:.2f}) -> Routed to {engine_dest}"

            wb_dicts = region_req.metadata.get("word_boxes", []) if region_req.metadata else []
            decomp_res: Optional[LineDecompositionResult] = None

            # Automatic line decomposition for handwritten Kannada line regions without pre-computed word boxes (experimental, disabled by default)
            if self.enable_line_word_decomposition and reg_is_hw and target_lang in ("kannada", "kn") and not is_full_page and not wb_dicts:
                w_crop, h_crop = reg_crop.size
                is_line_candidate = (
                    (region_req.metadata and region_req.metadata.get("region_type_detected") == "line")
                    or (w_crop > 1.8 * max(h_crop, 1))
                )
                if is_line_candidate:
                    decomp_res = self.line_decomposer.decompose(reg_crop, line_id=region_req.region_id)
                    if decomp_res.is_valid and decomp_res.word_candidates:
                        wb_dicts = [b.model_dump() for b in decomp_res.word_boxes]
                        base_x = region_req.bbox.x_min if region_req.bbox else 0.0
                        base_y = region_req.bbox.y_min if region_req.bbox else 0.0
                        for wb_i, cand in enumerate(decomp_res.word_candidates):
                            wb_box = BoundingBox(
                                x_min=base_x + cand.bbox.x_min,
                                y_min=base_y + cand.bbox.y_min,
                                x_max=base_x + cand.bbox.x_max,
                                y_max=base_y + cand.bbox.y_max,
                            )
                            batched_word_crops.append((reg_idx, wb_i, cand.crop, wb_box))

            region_routing_decisions[reg_idx] = {
                "reg_is_hw": reg_is_hw,
                "routing_conf": routing_conf,
                "routing_reason": routing_reason,
                "target_lang": target_lang,
                "wb_dicts": wb_dicts,
                "is_full_page": is_full_page,
                "decomp_res": decomp_res,
            }

            # Caller-provided word boxes
            if reg_is_hw and target_lang in ("kannada", "kn") and wb_dicts and decomp_res is None:
                for wb_i, wbd in enumerate(wb_dicts):
                    wb_box = BoundingBox(**wbd)
                    w_crop = self._crop_region_image(processed_pil, wb_box)
                    batched_word_crops.append((reg_idx, wb_i, w_crop, wb_box))

        crop_generation_ms = round((time.perf_counter() - t_crop_start) * 1000.0, 2)

        # Execute batched inference on all collected handwritten word crops in one GPU pass
        t_ocr_start = time.perf_counter()
        batched_word_results_map: Dict[Tuple[int, int], Any] = {}
        if batched_word_crops:
            crops_only = [item[2] for item in batched_word_crops]
            boxes_only = [item[3] for item in batched_word_crops]
            all_word_results = self.iitb_recognizer.recognize_batch(crops_only, bboxes=boxes_only)
            for item, w_res in zip(batched_word_crops, all_word_results):
                reg_idx, wb_i = item[0], item[1]
                batched_word_results_map[(reg_idx, wb_i)] = w_res

        # Process each region with either batched word outputs or individual routing
        for reg_idx, region_req in enumerate(adapted_regions):
            reg_start = time.perf_counter()
            reg_crop = region_crop_cache[reg_idx]
            r_info = region_routing_decisions[reg_idx]
            reg_is_hw = r_info["reg_is_hw"]
            routing_conf = r_info["routing_conf"]
            routing_reason = r_info["routing_reason"]
            target_lang = r_info["target_lang"]
            wb_dicts = r_info["wb_dicts"]
            is_full_page = r_info["is_full_page"]
            decomp_res = r_info.get("decomp_res")

            crop_prep_info = dict(prep_metadata)
            if region_req.metadata:
                crop_prep_info["region_custom_metadata"] = region_req.metadata

            detected_reg_type = "word"
            if is_full_page:
                detected_reg_type = "page"
            elif wb_dicts and len(wb_dicts) > 1:
                detected_reg_type = "line"
            elif region_req.metadata and region_req.metadata.get("region_type_detected") == "line":
                detected_reg_type = "line"
            elif region_req.bbox:
                box_w = region_req.bbox.x_max - region_req.bbox.x_min
                box_h = region_req.bbox.y_max - region_req.bbox.y_min
                if box_w > 1.8 * max(box_h, 1):
                    detected_reg_type = "line"

            reg_needs_review = False
            review_reason = None
            calibrated_conf = None  # None indicates uncalibrated

            # Check if line segmentation failed explicitly
            if decomp_res is not None and not decomp_res.is_valid:
                reg_needs_review = True
                review_reason = f"Multi-word line crop routed to word-oriented TrOCR model; uncalibrated line recognition requires officer review (segmentation uncertain: {decomp_res.failure_reason})"
                review_warnings.append(f"Region '{region_req.region_id}': {review_reason}")

            if reg_is_hw and target_lang in ("kannada", "kn") and wb_dicts and len(wb_dicts) > 0:
                # Reconstruct line from batched word outputs in exact reading order
                word_results = [batched_word_results_map[(reg_idx, wb_i)] for wb_i in range(len(wb_dicts))]
                rec_words = [w.text.strip() for w in word_results if w.text.strip()]
                raw_text = " ".join(rec_words)
                confs = [w.confidence for w in word_results if w.confidence is not None]
                conf_val = round(float(np.mean(confs)), 4) if confs else None
                engine_name = "iitb_kannada_v002"
                engine_ver = "iitb_kannada_v002"
                routing_reason = "Handwritten Kannada line decomposed into word crops batched through IITB TrOCR v0.0.2"

                # Check fallback on empty or invalid reconstructed line
                if not raw_text:
                    reg_needs_review = True
                    review_reason = "Line word decomposition yielded empty OCR text"
                    review_warnings.append(f"Region '{region_req.region_id}': {review_reason}")

                line_decomp_payload = {
                    "original_line_crop": reg_crop,
                    "detected_word_boxes": [b.model_dump() for b in decomp_res.word_boxes] if decomp_res else wb_dicts,
                    "per_word_results": [
                        {"box_index": i, "text": w.text, "confidence": w.confidence, "bbox": w.bbox.model_dump() if w.bbox else None}
                        for i, w in enumerate(word_results)
                    ],
                    "reconstructed_line_text": raw_text,
                    "segmentation_diagnostics": decomp_res.diagnostics if decomp_res else {},
                    "review_status": reg_needs_review,
                    "failure_reason": decomp_res.failure_reason if decomp_res else None,
                }

                ocr_metadata = {
                    "recognizer": "iitb_kannada_v002",
                    "model_path": "models/trocr/experimental/iitb_kannada_v002",
                    "granularity": "word_batch_reconstruction",
                    "routing_reason": routing_reason,
                    "line_decomposition": line_decomp_payload,
                    "word_results": [
                        {"text": w.text, "confidence": w.confidence, "bbox": w.bbox.model_dump() if w.bbox else None}
                        for w in word_results
                    ],
                }
                ocr_res = OCRResult(
                    text=raw_text,
                    confidence=conf_val,
                    bbox=region_req.bbox,
                    is_handwritten=True,
                    page_number=req.page_number,
                    model_name=engine_name,
                    model_version=engine_ver,
                    metadata=ocr_metadata,
                )
            else:
                reg_kwargs = dict(kwargs)
                if "enable_multipass" not in reg_kwargs:
                    reg_kwargs["enable_multipass"] = False


                ocr_res = self.router.route_and_recognize(
                    image=reg_crop,
                    language=target_lang,
                    is_handwritten=reg_is_hw,
                    bbox=region_req.bbox,
                    page_number=req.page_number,
                    preprocessing_info=crop_prep_info,
                    **reg_kwargs,
                )
                raw_text = ocr_res.text or ""
                conf_val = ocr_res.confidence
                engine_name = str(ocr_res.model_name or "unknown_engine")
                engine_ver = str(ocr_res.model_version) if ocr_res.model_version is not None else None
                if isinstance(ocr_res.metadata, dict):
                    ocr_metadata = dict(ocr_res.metadata.get("metadata", ocr_res.metadata))
                else:
                    ocr_metadata = {}
                ocr_metadata["routing_reason"] = routing_reason

                if decomp_res is not None:
                    ocr_metadata["line_decomposition"] = {
                        "original_line_crop": reg_crop,
                        "detected_word_boxes": [],
                        "per_word_results": [],
                        "reconstructed_line_text": "",
                        "segmentation_diagnostics": decomp_res.diagnostics,
                        "review_status": True,
                        "failure_reason": decomp_res.failure_reason,
                    }

            # Step 6: Non-destructive normalization & lexicon suggestions
            candidate_suggestions: List[str] = []
            if req.apply_normalization and target_lang in ("kannada", "kn"):
                norm_res = self.normalizer.normalize(raw_text)
                normalized_text = norm_res.normalized_text
                candidate_suggestions = norm_res.candidate_suggestions
            else:
                normalized_text = raw_text.strip()

            # Determine region status and review triggers
            reg_status = ProcessingStatus.SUCCESS
            if ocr_metadata.get("requires_human_review", False):
                reg_needs_review = True

            if ocr_metadata.get("engine_status") == "unsupported_language_model":
                reg_status = ProcessingStatus.UNSUPPORTED_LANGUAGE
                reg_needs_review = True
                review_reason = review_reason or f"Unsupported handwriting model for '{target_lang}'"
                review_warnings.append(f"Region '{region_req.region_id}': {review_reason}")
            elif not normalized_text:
                reg_status = ProcessingStatus.EMPTY
                if reg_is_hw is not False:
                    reg_needs_review = True
                    review_reason = review_reason or "Empty recognition output"
                    review_warnings.append(f"Region '{region_req.region_id}': {review_reason}")
            elif conf_val is not None and conf_val < self.confidence_threshold:
                reg_status = ProcessingStatus.LOW_CONFIDENCE
                reg_needs_review = True
                review_reason = f"Low recognizer confidence ({conf_val:.2f} < {self.confidence_threshold:.2f})"
                review_warnings.append(f"Region '{region_req.region_id}': {review_reason}")
            elif reg_is_hw and target_lang in ("kannada", "kn") and detected_reg_type == "line" and ("trocr" in engine_name.lower() or "iitb" in engine_name.lower()):
                w_crop, h_crop = reg_crop.size
                if w_crop > 2.0 * max(h_crop, 1):
                    reg_needs_review = True
                    review_reason = review_reason or "Multi-word line crop routed to word-oriented TrOCR model; uncalibrated line recognition requires officer review"
                    review_warnings.append(f"Region '{region_req.region_id}': {review_reason}")

            reg_duration_ms = round((time.perf_counter() - reg_start) * 1000.0, 2)
            engine_breakdown[engine_name] = engine_breakdown.get(engine_name, 0) + 1

            det_conf = (
                region_req.metadata.get("detection_confidence", 0.95)
                if region_req.metadata
                else 0.95
            )

            calibrated_conf, _ = self.calibrator.calibrate(
                engine_name=engine_name,
                raw_score=conf_val,
            )
            reg_confidence_state = (
                ConfidenceState.REVIEW_REQUIRED if reg_needs_review
                else (ConfidenceState.CALIBRATED if calibrated_conf is not None else ConfidenceState.UNCALIBRATED)
            )

            reg_result = RecognizedRegionResult(
                region_id=region_req.region_id,
                raw_text=raw_text,
                normalized_text=normalized_text,
                bbox=region_req.bbox,
                page_number=req.page_number,
                language=target_lang,
                script=region_req.script or "Kannada",
                is_handwritten=ocr_res.is_handwritten,
                confidence=conf_val,
                recognizer_confidence_raw=conf_val,
                calibrated_confidence=calibrated_conf,
                confidence_state=reg_confidence_state,
                detection_confidence=det_conf,
                routing_confidence=routing_conf,
                region_type=detected_reg_type,
                routing_reason=routing_reason,
                review_reason=review_reason,
                model_name=engine_name,
                model_version=engine_ver,
                inference_time_ms=reg_duration_ms,
                preprocessing_metadata=prep_metadata,
                requires_human_review=reg_needs_review,
                status=reg_status,
                candidate_suggestions=candidate_suggestions,
                custom_metadata=ocr_metadata,
            )
            recognized_regions.append(reg_result)

            # Build backward-compatible OCRResult
            compat_meta = dict(ocr_res.metadata)
            compat_meta["metadata"] = {
                **ocr_metadata,
                "region_id": region_req.region_id,
                "raw_text": raw_text,
                "normalized_text": normalized_text,
                "candidate_suggestions": candidate_suggestions,
                "requires_human_review": reg_needs_review,
                "inference_time_ms": reg_duration_ms,
                "region_type": detected_reg_type,
                "review_reason": review_reason,
                "routing_reason": routing_reason,
            }
            ocr_res.metadata = compat_meta
            ocr_results_compat.append(ocr_res)

        ocr_inference_ms = round((time.perf_counter() - t_ocr_start) * 1000.0, 2)

        # Step 7: Reading order text merging & reconstruction
        t_recon_start = time.perf_counter()
        recognized_lines = [r.normalized_text for r in recognized_regions if r.normalized_text]
        merged_text = "\n".join(recognized_lines)

        original_kannada_lines = [
            r.normalized_text for r in recognized_regions
            if r.normalized_text and any('\u0c80' <= c <= '\u0cff' for c in r.normalized_text)
        ]
        original_kannada_text = "\n".join(original_kannada_lines) if original_kannada_lines else merged_text

        doc_confidence = self._compute_document_confidence(recognized_regions)
        is_low_conf = (doc_confidence is not None and doc_confidence < self.confidence_threshold)
        reconstruction_ms = round((time.perf_counter() - t_recon_start) * 1000.0, 2)

        # Step 7b: Safe Gated Translation Stage (Prevents bad OCR -> hallucinated English translation cascades)
        t_trans_start = time.perf_counter()
        TRANS_HIGH = 0.85
        TRANS_MEDIUM = 0.60

        has_uncalibrated_line = any(
            r.region_type == "line" and r.is_handwritten and "iitb" in (r.model_name or "")
            for r in recognized_regions
        )

        translation_status = "completed"
        if is_low_conf or not merged_text or has_uncalibrated_line or (doc_confidence is not None and doc_confidence < TRANS_MEDIUM):
            clean_kannada_text = original_kannada_text
            kannada_translation = original_kannada_text
            translated_text = "[Translation skipped: Low OCR confidence]"
            english_translation = "[Translation skipped: Low OCR confidence]"
            translation_status = "skipped_low_confidence"
            trans_res_dict = None
            if has_uncalibrated_line:
                msg = "Machine translation blocked: Multi-word handwritten line recognition is uncalibrated and requires human review"
                if msg not in review_warnings:
                    review_warnings.append(msg)
                logger.info("[TRANSLATION GATE] Blocked translation: uncalibrated multi-word TrOCR line detected")
            elif not merged_text:
                review_warnings.append("No text extracted from document")
            else:
                conf_display = f"{doc_confidence:.2f}" if doc_confidence is not None else "0.00"
                msg = f"Machine translation blocked: OCR recognition confidence ({conf_display}) is below {TRANS_MEDIUM:.2f} threshold"
                if msg not in review_warnings:
                    review_warnings.append(msg)
                logger.info("[TRANSLATION GATE] Blocked translation: confidence %s < threshold %.2f", conf_display, TRANS_MEDIUM)
        else:
            if doc_confidence is not None and doc_confidence < TRANS_HIGH:
                translation_status = "flagged_medium_confidence"
                msg = f"Machine translation generated on medium-confidence OCR ({doc_confidence:.2f}); officer review recommended"
                if msg not in review_warnings:
                    review_warnings.append(msg)
                logger.info("[TRANSLATION GATE] Generated translation with medium-confidence warning")
            else:
                translation_status = "completed"
                logger.info("[TRANSLATION GATE] Generated high-confidence translation")

            from src.translation.translator import translate_document_text
            trans_res = translate_document_text(
                text=merged_text,
                regions=recognized_regions,
            )
            clean_kannada_text = trans_res.kannada_translation or original_kannada_text
            kannada_translation = clean_kannada_text
            translated_text = trans_res.english_translation or merged_text
            english_translation = translated_text
            trans_res_dict = trans_res.model_dump()
            if trans_res.purity_report.requires_review:
                for w in trans_res.purity_report.warnings:
                    if w not in review_warnings:
                        review_warnings.append(w)

        translation_ms = round((time.perf_counter() - t_trans_start) * 1000.0, 2)

        # Step 8: Spatial Land-Record Field Extraction & NER Table Processing
        t_ner_table_start = time.perf_counter()
        # Match field anchors strictly to spatially adjacent value cells (below, right, inline)
        # Never populate a field merely because a matching word occurs somewhere else on the page
        spatial_fields = self.tabular_extractor.extract_from_regions(recognized_regions)
        extracted_fields_dict = {}
        for fname, fld in spatial_fields.items():
            d = fld.to_dict()
            if "raw_value" not in d or not d.get("raw_value"):
                d["raw_value"] = getattr(fld, "value_text", None) or d.get("value_text")
            if "normalized_value" not in d or not d.get("normalized_value"):
                d["normalized_value"] = d.get("raw_value")
            extracted_fields_dict[fname] = d

        # Fallback to regex NER only for fields not captured spatially
        ner_fields = self.field_extractor.extract_fields(
            lines=recognized_lines,
            line_confidences=[(r.confidence if r.confidence is not None else 0.0) for r in recognized_regions if r.normalized_text],
        )
        for fname, f_obj in ner_fields.items():
            if fname not in extracted_fields_dict:
                d = f_obj.to_dict()
                if "raw_value" not in d or not d.get("raw_value"):
                    d["raw_value"] = getattr(f_obj, "raw_value", None) or d.get("value")
                if "normalized_value" not in d or not d.get("normalized_value"):
                    d["normalized_value"] = getattr(f_obj, "normalized_value", None) or d.get("raw_value")
                extracted_fields_dict[fname] = d

        # Step 9: Field-Aligned Translation & Transliteration
        bilingual_dict = {}
        if translation_status != "skipped_low_confidence":
            bilingual_fields = self.field_translator.translate_structured_fields(ner_fields)
            bilingual_dict = {
                fname: {
                    "field_name": b.field_name,
                    "kannada": b.kannada_value,
                    "english": b.english_value,
                    "method": b.method,
                    "confidence": b.confidence,
                }
                for fname, b in bilingual_fields.items()
            }
        ner_table_ms = round((time.perf_counter() - t_ner_table_start) * 1000.0, 2)

        # Step 9b: Semantic V1 Pipeline Orchestration (AI-driven and Layout aware)
        t_sem_start = time.perf_counter()
        is_cad = gate_result.is_land_record if gate_result else True
        semantic_doc = self.semantic_pipeline.process(
            document_id=req.document_id or f"doc_{int(time.time())}",
            page_number=req.page_number,
            regions=recognized_regions,
            image_width=cur_w,
            image_height=cur_h,
            ner_candidates=extracted_fields_dict,
            is_cadastral=is_cad,
            classification_result=gate_result.to_dict() if gate_result else None,
        )
        semantic_ms = round((time.perf_counter() - t_sem_start) * 1000.0, 2)

        # Step 10: Validation, Confidence & Review Queue Resolution
        t_val_start = time.perf_counter()

        # Merge canonical semantic fields into extracted_fields_dict and translate
        for fname, fitem in semantic_doc.fields.items():
            prov_dict = fitem.provenance.model_dump() if fitem.provenance else None
            conflicts_list = [c.model_dump() for c in fitem.conflicts] if fitem.conflicts else []

            # Step 10a: Kannada -> English translation/transliteration for canonical fields
            trans_info = self.field_translator.translate_field(
                field_name=fname,
                value=fitem.value,
                raw_value=fitem.raw_value,
                provenance=prov_dict,
                confidence=float(fitem.confidence) if fitem.confidence is not None else None,
            )
            fitem.english_value = trans_info.get("english_value")
            fitem.translation_status = trans_info.get("translation_status")
            fitem.translation_engine = trans_info.get("translation_engine")

            extracted_fields_dict[fname] = {
                "field_name": fname,
                "raw_value": fitem.raw_value,
                "normalized_value": fitem.value,
                "english_value": trans_info.get("english_value"),
                "translation_status": trans_info.get("translation_status"),
                "translation_engine": trans_info.get("translation_engine"),
                "confidence": round(float(fitem.confidence), 4) if fitem.confidence is not None else 0.85,
                "validation_status": fitem.validation_status.value if hasattr(fitem.validation_status, "value") else str(fitem.validation_status),
                "validation_reason": fitem.validation_reason,
                "provenance": prov_dict,
                "source_region_id": fitem.provenance.region_id if fitem.provenance else None,
                "bbox": fitem.provenance.bbox.model_dump() if (fitem.provenance and fitem.provenance.bbox) else None,
                "conflicts": conflicts_list,
                "method": fitem.provenance.extraction_method if fitem.provenance else "semantic_layer",
            }

        # Step 10b: Update bilingual_dict with all canonical extracted fields
        bilingual_fields = self.field_translator.translate_structured_fields(extracted_fields_dict)
        bilingual_dict = {
            fname: {
                "field_name": b.field_name,
                "kannada": b.kannada_value,
                "english": b.english_value,
                "method": b.method,
                "confidence": b.confidence,
                "translation_status": b.translation_status,
                "provenance": b.provenance,
            }
            for fname, b in bilingual_fields.items()
        }

        # Document-level review decision
        doc_requires_review = len(review_warnings) > 0
        if is_low_conf:
            doc_requires_review = True
            warning_msg = f"Low confidence: Overall document confidence is low ({doc_confidence:.2f} < {self.confidence_threshold:.2f})"
            if warning_msg not in review_warnings and not any("confidence is low" in w for w in review_warnings):
                review_warnings.append(warning_msg)
        # Flag empty or near-empty OCR output as needing review
        if not merged_text or len(merged_text.strip()) < 3:
            doc_requires_review = True
            if "No text extracted from document" not in review_warnings:
                review_warnings.append("No text extracted from document")

        # Build HumanReviewItem queue for all regions/fields requiring human inspection
        review_items_list: List[HumanReviewItem] = []

        # Check semantic fields requiring human review and enqueue review items
        for fname, fitem in semantic_doc.fields.items():
            needs_field_review = False
            field_reasons = []
            val_status_str = getattr(fitem.validation_status, "value", str(fitem.validation_status)).upper()
            if val_status_str in ("INVALID", "WARNING"):
                needs_field_review = True
                field_reasons.append(fitem.validation_reason or f"Semantic validation flag: {val_status_str}")
            if fitem.conflicts and len(fitem.conflicts) > 0:
                needs_field_review = True
                field_reasons.append(f"Ambiguous candidates detected ({len(fitem.conflicts)} alternatives)")

            if needs_field_review:
                doc_requires_review = True
                prov = fitem.provenance
                r_id = prov.region_id if (prov and prov.region_id) else f"semantic_{fname}"
                reason_str = "; ".join(field_reasons)
                warning_entry = f"Semantic [{fname}]: {reason_str}"
                if warning_entry not in review_warnings:
                    review_warnings.append(warning_entry)

                rev_item = self.review_state_machine.create_review_item(
                    document_id=req.document_id,
                    page_number=req.page_number,
                    region_id=r_id,
                    bbox=prov.bbox if prov else None,
                    raw_ocr_text=fitem.raw_value,
                    recognizer="semantic_pipeline",
                    review_reason=f"Field [{fname}]: {reason_str}",
                    recognizer_confidence_raw=fitem.confidence,
                    calibrated_confidence=None,
                    is_critical_field=(fname in CRITICAL_LAND_RECORD_FIELDS),
                    metadata={
                        "field_name": fname,
                        "raw_value": fitem.raw_value,
                        "normalized_value": fitem.value,
                        "validation_status": fitem.validation_status.value if hasattr(fitem.validation_status, "value") else str(fitem.validation_status),
                        "validation_reason": fitem.validation_reason,
                        "conflicts": [c.model_dump() for c in fitem.conflicts] if fitem.conflicts else [],
                        "provenance": prov.model_dump() if prov else None,
                    },
                )
                review_items_list.append(rev_item)

        for r in recognized_regions:
            if r.requires_human_review:
                rev_item = self.review_state_machine.create_review_item(
                    document_id=req.document_id,
                    page_number=req.page_number,
                    region_id=r.region_id,
                    bbox=r.bbox,
                    raw_ocr_text=r.raw_text,
                    recognizer=r.model_name or "unknown",
                    review_reason=r.review_reason or "Review triggered by pipeline policy",
                    recognizer_confidence_raw=r.recognizer_confidence_raw,
                    calibrated_confidence=r.calibrated_confidence,
                    translated_text=None,
                    translation_status="blocked_by_policy" if r.is_handwritten else "not_translated",
                    metadata={
                        "region_type": r.region_type,
                        "script": r.script,
                        "is_handwritten": r.is_handwritten,
                    },
                )
                review_items_list.append(rev_item)

        pipeline_diagnostics = {}
        if "iitb_kannada_v002" in engine_breakdown:
            logger.info("[OCR DIAGNOSTIC] recognizer=iitb_kannada_v002 model_path=models/trocr/experimental/iitb_kannada_v002")
            print("[OCR DIAGNOSTIC] recognizer=iitb_kannada_v002 model_path=models/trocr/experimental/iitb_kannada_v002")
            pipeline_diagnostics["recognizer"] = "iitb_kannada_v002"
            pipeline_diagnostics["model_path"] = "models/trocr/experimental/iitb_kannada_v002"

        # Step 10: Calibrated 4-part confidence reporting
        recognition_confidence = doc_confidence
        det_confs = [r.detection_confidence for r in recognized_regions if r.detection_confidence is not None]
        detection_confidence = round(float(np.mean(det_confs)), 4) if det_confs else 1.0
        route_confs = [r.routing_confidence for r in recognized_regions if r.routing_confidence is not None]
        routing_confidence = round(float(np.mean(route_confs)), 4) if route_confs else 1.0
        field_confs = [f["confidence"] for f in extracted_fields_dict.values() if isinstance(f, dict) and "confidence" in f]
        field_confidence = round(float(np.mean(field_confs)), 4) if field_confs else (1.0 if not extracted_fields_dict else None)

        # Calibrated overall confidence calculation
        calibrated_confs = [r.calibrated_confidence for r in recognized_regions if r.calibrated_confidence is not None]
        doc_calibrated_conf = (
            round(float(np.mean(calibrated_confs)), 4)
            if (len(calibrated_confs) == len(recognized_regions) and len(recognized_regions) > 0)
            else None
        )

        # Confidence state determination
        doc_confidence_state = (
            ConfidenceState.REVIEW_REQUIRED if doc_requires_review
            else (ConfidenceState.CALIBRATED if doc_calibrated_conf is not None else ConfidenceState.UNCALIBRATED)
        )

        # Explicit verification status
        is_accepted = (
            not doc_requires_review
            and doc_confidence is not None
            and doc_confidence >= self.confidence_threshold
            and (gate_result is None or gate_result.is_land_record)
        )
        verification_status = "accepted" if is_accepted else "needs_verification"

        doc_status = "flagged_for_review" if doc_requires_review else "completed"
        validation_ms = round((time.perf_counter() - t_val_start) * 1000.0, 2)
        total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        stage_timings = {
            "file_decode_ms": file_decode_ms,
            "page_rendering_ms": page_rendering_ms,
            "gate_classification_ms": gate_ms,
            "layout_detection_ms": layout_detection_ms,
            "crop_generation_ms": crop_generation_ms,
            "ocr_inference_ms": ocr_inference_ms,
            "reconstruction_ms": reconstruction_ms,
            "translation_ms": translation_ms,
            "ner_table_ms": ner_table_ms,
            "semantic_ms": semantic_ms,
            "validation_ms": validation_ms,
            "total_ms": total_time_ms,
        }

        # Assemble unified Structured OCR Output
        structured_regions = [
            OCRRegion(
                region_id=r.region_id,
                bbox=[
                    r.bbox.x_min if r.bbox else 0,
                    r.bbox.y_min if r.bbox else 0,
                    r.bbox.x_max if r.bbox else cur_w,
                    r.bbox.y_max if r.bbox else cur_h,
                ],
                region_type=SRegionType(r.region_type) if r.region_type in ("word", "line", "field", "page") else SRegionType.WORD,
                script=SScriptType(r.script) if r.script in ("Kannada", "English", "Mixed", "Unknown") else SScriptType.KANNADA,
                handwriting=bool(r.is_handwritten),
                recognizer=r.model_name or "unknown",
                text=r.normalized_text,
                recognizer_confidence_raw=r.recognizer_confidence_raw,
                calibrated_confidence=None,  # Null: uncalibrated
                needs_review=r.requires_human_review,
                review_reason=r.review_reason,
                routing_reason=r.routing_reason,
            )
            for r in recognized_regions
        ]

        structured_page = OCRPage(
            page_number=req.page_number,
            width=cur_w,
            height=cur_h,
            regions=structured_regions,
            merged_text=merged_text,
            translated_text=translated_text if translation_status != "skipped_low_confidence" else None,
            translation_status=translation_status,
            timings=PageTimings(**stage_timings),
        )

        # Document Type Classifier Contract: Strict neutral behavior without fabricated categories
        doc_type_val = None
        doc_type_state = "UNKNOWN"
        classifier_source = None
        classifier_score = None
        classifier_evidence = []

        if gate_result and not gate_result.is_land_record:
            doc_type_val = "Not a land record"
            doc_type_state = "CONFIRMED"
            classifier_source = "land_record_gate_classifier"
            classifier_score = gate_result.confidence
            classifier_evidence = list(gate_result.reasons)

        structured_doc = StructuredDocumentOCR(
            document_id=req.document_id or f"doc_{int(time.time())}",
            document_type=doc_type_val or "Unknown / Not classified",
            is_land_record=gate_result.is_land_record if gate_result else None,
            pages=[structured_page],
            overall_confidence_calibrated=doc_calibrated_conf,
            status=doc_status,
            requires_human_review=doc_requires_review,
            total_timings=PageTimings(**stage_timings),
        )

        compat_page = DocumentPage(
            page_number=req.page_number,
            image_path=resolved_img_path,
            ocr_results=ocr_results_compat,
            metadata={
                "processing_time_sec": round(total_time_ms / 1000.0, 4),
                "original_dimensions": {"width": orig_size[0], "height": orig_size[1]},
                "processed_dimensions": {"width": cur_w, "height": cur_h},
                "preprocessing": prep_metadata,
                "requires_human_review": doc_requires_review,
                "confidence": doc_confidence if doc_confidence is not None else 0.0,
                "is_handwritten": is_handwritten,
                "language": language or self.default_language,
                "script": script or "Kannada",
                "diagnostics": pipeline_diagnostics,
                "stage_timings": stage_timings,
                "structured_ocr": structured_doc.model_dump(),
            },
        )

        return DocumentProcessingResponse(
            document_id=req.document_id,
            page_number=req.page_number,
            image_path=resolved_img_path,
            ordered_regions=recognized_regions,
            merged_text=merged_text,
            original_ocr=merged_text,
            clean_kannada_text=clean_kannada_text,
            kannada_translation=kannada_translation,
            translated_text=translated_text,
            english_translation=english_translation,
            original_kannada_text=original_kannada_text,
            translation_result=trans_res_dict,
            translation_status=translation_status,
            document_confidence=doc_confidence,
            recognition_confidence=recognition_confidence,
            detection_confidence=detection_confidence,
            routing_confidence=routing_confidence,
            field_confidence=field_confidence,
            calibrated_confidence=doc_calibrated_conf,
            confidence_state=doc_confidence_state,
            verification_status=verification_status,
            status=doc_status,
            requires_human_review=doc_requires_review,
            warnings=review_warnings,
            engine_breakdown=engine_breakdown,
            processing_time_ms=total_time_ms,
            page=compat_page,
            is_land_record=gate_result.is_land_record if gate_result else True,
            gate_classification=gate_result.to_dict() if gate_result else None,
            extracted_fields=extracted_fields_dict,
            bilingual_fields=bilingual_dict,
            semantic_data=semantic_doc.model_dump(),
            tables=[t.model_dump() for t in semantic_doc.tables],
            review_items=[ri.to_dict() for ri in review_items_list],
            document_type=doc_type_val,
            document_type_state=doc_type_state,
            classifier_source=classifier_source,
            classifier_score=classifier_score,
            classifier_evidence=classifier_evidence,
            stage_timings=stage_timings,
            structured_ocr=structured_doc.model_dump(),
            diagnostics=pipeline_diagnostics,
        )


# Module-level singleton
_DEFAULT_PIPELINE: Optional[DocumentProcessingPipeline] = None


def get_default_pipeline() -> DocumentProcessingPipeline:
    """Returns or creates the shared DocumentProcessingPipeline singleton."""
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = DocumentProcessingPipeline()
    return _DEFAULT_PIPELINE


def process_document(
    image: Optional[ImageInput] = None,
    request: Optional[DocumentProcessingRequest] = None,
    regions: Optional[Sequence[Any]] = None,
    is_handwritten: Optional[bool] = None,
    language: Optional[str] = None,
    script: Optional[str] = None,
    page_number: int = 1,
    document_id: Optional[str] = None,
    image_path: Optional[str] = None,
    pipeline: Optional[DocumentProcessingPipeline] = None,
    **kwargs: Any,
) -> DocumentProcessingResponse:
    """Public functional entry point for processing a document image.

    Args:
        image: Document image (file path, PIL Image, array, bytes).
        request: Optional DocumentProcessingRequest.
        regions: Optional list of bounding boxes or region definitions.
        is_handwritten: Optional boolean flag for printed vs. handwritten.
        language: Language identifier (default: 'kannada').
        script: Script identifier.
        page_number: 1-indexed page number.
        document_id: Optional document identifier string.
        image_path: Optional path for metadata.
        pipeline: Optional custom DocumentProcessingPipeline instance.
        **kwargs: Extra parameters passed to recognizers.

    Returns:
        DocumentProcessingResponse containing structured regions, merged text,
        confidence, and review decision.
    """
    pipe = pipeline or get_default_pipeline()
    return pipe.process_document(
        image=image,
        request=request,
        regions=regions,
        is_handwritten=is_handwritten,
        language=language,
        script=script,
        page_number=page_number,
        document_id=document_id,
        image_path=image_path,
        **kwargs,
    )
