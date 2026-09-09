"""Document Processing Orchestrator for Land Record Digitization.

Orchestrates preprocessing, layout region ingestion, script/handwriting routing,
reading-order text merging, confidence aggregation, non-destructive normalization,
and human-review flagging for downstream consumption by Person C.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from schemas import BoundingBox, DocumentPage, OCRResult
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput
from src.handwriting.router import LanguageScriptRouter
from src.integration.person_a_adapter import PersonAAdapter
from src.integration.schemas import (
    DocumentProcessingRequest,
    DocumentProcessingResponse,
    ProcessingStatus,
    RecognizedRegionResult,
    RegionRequest,
    RegionType,
)
from src.postprocessing.kannada_normalizer import KannadaNormalizer
from src.preprocessing.image_enhancement import (
    PreprocessingResult,
    load_image_as_pil,
    preprocess_document_image,
)

# Type alias for caller convenience
RegionInput = Union[BoundingBox, RegionRequest, Dict[str, Any], Tuple[int, int, int, int]]


# Backwards-compatible alias / wrapper
DocumentProcessingResult = DocumentProcessingResponse


class DocumentProcessingPipeline:
    """Production orchestrator combining layout adaptation, routing, OCR, normalization, and audit trails."""

    def __init__(
        self,
        router: Optional[LanguageScriptRouter] = None,
        normalizer: Optional[KannadaNormalizer] = None,
        default_language: str = "kannada",
        confidence_threshold: float = 0.60,
        apply_preprocessing: bool = True,
        apply_normalization: bool = True,
    ):
        """Initializes the document pipeline.

        Args:
            router: Pre-configured or mock LanguageScriptRouter. If None, instantiates default.
            normalizer: Optional KannadaNormalizer for conservative text sanitation and lexicon suggestions.
            default_language: Default document language script (e.g. 'kannada').
            confidence_threshold: Threshold below which pages/regions are flagged for review.
            apply_preprocessing: Whether to apply deskew/denoise/contrast enhancements by default.
            apply_normalization: Whether to apply conservative text normalization by default.
        """
        self.router = router if router is not None else LanguageScriptRouter(default_language=default_language)
        self.normalizer = normalizer if normalizer is not None else KannadaNormalizer()
        self.default_language = default_language
        self.confidence_threshold = confidence_threshold
        self.apply_preprocessing = apply_preprocessing
        self.apply_normalization = apply_normalization

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

        # Step 2: Load raw image
        raw_pil = load_image_as_pil(req.image)
        orig_size = raw_pil.size

        # Step 3: Preprocess document image
        prep_metadata: Dict[str, Any] = {}
        if req.apply_preprocessing:
            prep_res: PreprocessingResult = preprocess_document_image(
                image_input=raw_pil,
                apply_thresholding=False,
            )
            processed_pil = prep_res.image
            prep_metadata = prep_res.audit_metadata
        else:
            processed_pil = raw_pil.copy()
            prep_metadata = {
                "pipeline_steps": ["raw_passthrough"],
                "original_size": {"width": orig_size[0], "height": orig_size[1]},
            }

        # Step 4: Decompose into regions
        if req.regions and len(req.regions) > 0:
            adapted_regions = PersonAAdapter.adapt_layout_regions(
                regions=req.regions,
                image_size=orig_size,
                default_language=req.language,
                default_script=req.script,
                default_is_handwritten=req.is_handwritten,
            )
        else:
            # Full image mode: treat whole image as single region
            adapted_regions = [
                RegionRequest(
                    region_id="region_001",
                    bbox=BoundingBox(x_min=0, y_min=0, x_max=orig_size[0], y_max=orig_size[1]),
                    language=req.language,
                    script=req.script,
                    is_handwritten=req.is_handwritten,
                    region_type=RegionType.TEXT,
                )
            ]

        # Step 5: Run routing & recognition across regions
        recognized_regions: List[RecognizedRegionResult] = []
        ocr_results_compat: List[OCRResult] = []
        engine_breakdown: Dict[str, int] = {}
        review_warnings: List[str] = []

        for reg_idx, region_req in enumerate(adapted_regions):
            reg_start = time.perf_counter()
            reg_crop = self._crop_region_image(processed_pil, region_req.bbox)
            target_lang = region_req.language or req.language
            reg_is_hw = region_req.is_handwritten if region_req.is_handwritten is not None else req.is_handwritten

            # Combine region metadata and preprocessing info
            crop_prep_info = dict(prep_metadata)
            if region_req.metadata:
                crop_prep_info["region_custom_metadata"] = region_req.metadata

            # Route to appropriate recognizer
            ocr_res = self.router.route_and_recognize(
                image=reg_crop,
                language=target_lang,
                is_handwritten=reg_is_hw,
                bbox=region_req.bbox,
                page_number=req.page_number,
                preprocessing_info=crop_prep_info,
                **kwargs,
            )

            raw_text = ocr_res.text or ""
            reg_meta = ocr_res.metadata.get("metadata", {})
            engine_name = ocr_res.model_name or "unknown_engine"
            engine_ver = ocr_res.model_version
            conf_val = ocr_res.confidence

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
            reg_needs_review = reg_meta.get("requires_human_review", False)

            if reg_meta.get("engine_status") == "unsupported_language_model":
                reg_status = ProcessingStatus.UNSUPPORTED_LANGUAGE
                reg_needs_review = True
                review_warnings.append(
                    f"Region '{region_req.region_id}': Unsupported handwriting model for '{target_lang}'"
                )
            elif not normalized_text:
                reg_status = ProcessingStatus.EMPTY
                if reg_is_hw is not False:
                    reg_needs_review = True
                    review_warnings.append(f"Region '{region_req.region_id}': Empty recognition output")
            elif conf_val is not None and conf_val < self.confidence_threshold:
                reg_status = ProcessingStatus.LOW_CONFIDENCE
                reg_needs_review = True
                review_warnings.append(
                    f"Region '{region_req.region_id}': Low confidence ({conf_val:.2f} < {self.confidence_threshold:.2f})"
                )

            # Track engine breakdown
            engine_breakdown[engine_name] = engine_breakdown.get(engine_name, 0) + 1
            reg_duration_ms = round((time.perf_counter() - reg_start) * 1000.0, 2)

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
                model_name=engine_name,
                model_version=engine_ver,
                inference_time_ms=reg_duration_ms,
                preprocessing_metadata=prep_metadata,
                requires_human_review=reg_needs_review,
                status=reg_status,
                candidate_suggestions=candidate_suggestions,
                custom_metadata=ocr_res.metadata,
            )
            recognized_regions.append(reg_result)

            # Build backward-compatible OCRResult
            compat_meta = dict(ocr_res.metadata)
            compat_meta["metadata"] = {
                **reg_meta,
                "region_id": region_req.region_id,
                "raw_text": raw_text,
                "normalized_text": normalized_text,
                "candidate_suggestions": candidate_suggestions,
                "requires_human_review": reg_needs_review,
                "inference_time_ms": reg_duration_ms,
            }
            ocr_res.metadata = compat_meta
            ocr_results_compat.append(ocr_res)

        # Step 7: Assemble full document text & aggregate confidence
        recognized_lines = [r.normalized_text for r in recognized_regions if r.normalized_text]
        merged_text = "\n".join(recognized_lines)

        doc_confidence = self._compute_document_confidence(recognized_regions)

        # Document-level review decision
        doc_requires_review = len(review_warnings) > 0
        if doc_confidence is not None and doc_confidence < self.confidence_threshold:
            doc_requires_review = True
            warning_msg = f"Overall document confidence is low ({doc_confidence:.2f} < {self.confidence_threshold:.2f})"
            if warning_msg not in review_warnings:
                review_warnings.append(warning_msg)
        elif not merged_text:
            doc_requires_review = True
            review_warnings.append("No text extracted from document")

        doc_status = "flagged_for_review" if doc_requires_review else "completed"
        total_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Backward compatibility: Construct DocumentPage
        compat_page = DocumentPage(
            page_number=req.page_number,
            image_path=resolved_img_path,
            ocr_results=ocr_results_compat,
            metadata={
                "processing_time_sec": round(total_time_ms / 1000.0, 4),
                "original_dimensions": {"width": orig_size[0], "height": orig_size[1]},
                "preprocessing": prep_metadata,
                "document_language": req.language,
                "regions_processed": len(adapted_regions),
                "document_confidence": doc_confidence,
                "requires_human_review": doc_requires_review,
                "document_id": req.document_id,
            },
        )

        return DocumentProcessingResponse(
            document_id=req.document_id,
            page_number=req.page_number,
            image_path=resolved_img_path,
            ordered_regions=recognized_regions,
            merged_text=merged_text,
            document_confidence=doc_confidence,
            status=doc_status,
            requires_human_review=doc_requires_review,
            warnings=review_warnings,
            engine_breakdown=engine_breakdown,
            processing_time_ms=total_time_ms,
            page=compat_page,
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
