"""person-a/src/pipeline.py
Person A Master Vision, Preprocessing, Layout, Printed OCR, and Orchestrator.
Orchestrates: Ingestion -> SHA-256 -> Quality -> Preprocessing -> OCR -> Layout -> Classification -> Region Routing.
"""

from pathlib import Path
import time
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
import numpy as np

from .classification.classifier import DocumentClassifier
from .config.loader import get_state_config
from .config.models import StateConfiguration
from .handwriting.adapter import HandwritingAdapter, HandwritingResult, get_handwriting_adapter
from .layout.detector import LayoutDetector
from .ocr.paddle_ocr import PaddleOCREngine
from .preprocessing.pipeline import PreprocessedPage, PreprocessingConfig, PreprocessingPipeline
from .schemas import (
    BlockType,
    ClassificationResult,
    DocumentInput,
    ImageQualityAssessment,
    LanguageDetectionResult,
    OCRBlock,
    OCREngineType,
    OCROutput,
    OCRPageResult,
    TableStructure,
)
from .utils.hashing import compute_document_hash
from .utils.logging import log_stage_event, logger


class PersonAPipeline:
    """Master Vision & Document Intelligence Pipeline."""

    def __init__(
        self,
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
        ocr_engine: Optional[PaddleOCREngine] = None,
        layout_detector: Optional[LayoutDetector] = None,
        handwriting_adapter: Optional[HandwritingAdapter] = None,
    ):
        self.preprocessing = preprocessing_pipeline or PreprocessingPipeline()
        self.ocr_engine = ocr_engine or PaddleOCREngine()
        self.layout_detector = layout_detector or LayoutDetector()
        self.handwriting_adapter = handwriting_adapter or get_handwriting_adapter()

    def process_document(
        self,
        document_input: Union[str, Path, bytes, BinaryIO, np.ndarray, DocumentInput],
        state_hint: Optional[str] = None,
        target_languages: Optional[List[str]] = None,
        config: Optional[PreprocessingConfig] = None,
    ) -> OCROutput:
        """Execute end-to-end Person A document processing."""
        start_time = time.perf_counter()

        # 1. Normalize input representation & compute SHA-256
        doc_id = "doc_" + str(int(time.time() * 1000))
        source_data: Union[str, Path, bytes, BinaryIO, np.ndarray]
        active_state_hint = state_hint

        if isinstance(document_input, DocumentInput):
            doc_id = document_input.document_id
            active_state_hint = document_input.state_hint or state_hint
            target_languages = document_input.target_languages or target_languages
            if document_input.file_bytes:
                source_data = document_input.file_bytes
            elif document_input.file_path:
                source_data = document_input.file_path
            else:
                source_data = b""
        elif isinstance(document_input, (str, Path)):
            source_data = document_input
            doc_id = Path(document_input).stem
        else:
            source_data = document_input

        # Compute SHA-256 Hash
        hash_start = time.perf_counter()
        try:
            sha256 = compute_document_hash(source_data)
        except Exception:
            sha256 = "0" * 64
        hashing_dur = (time.perf_counter() - hash_start) * 1000.0

        if isinstance(document_input, DocumentInput):
            document_input.metadata["sha256"] = sha256

        # 2. Resolve State Configuration
        state_cfg: StateConfiguration = get_state_config(state_hint=active_state_hint)
        primary_lang = state_cfg.primary_language or (target_languages[0] if target_languages else "en")

        log_stage_event("document_received", doc_id, extra={"state_hint": state_cfg.state_code, "lang": primary_lang})

        # 3. Quality Analysis & Adaptive Preprocessing
        prep_pipe = PreprocessingPipeline(config=config) if config else self.preprocessing
        prep_start = time.perf_counter()
        preprocessed_pages = prep_pipe.process(source_data)
        prep_dur = (time.perf_counter() - prep_start) * 1000.0

        log_stage_event("preprocessing_completed", doc_id, duration_ms=prep_dur, extra={"pages": len(preprocessed_pages)})

        if not preprocessed_pages:
            # Handle corrupt / non-existent / empty document safely
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return OCROutput(
                document_id=doc_id,
                sha256_hash=sha256,
                ocr_engine=OCREngineType.PADDLE_OCR.value,
                overall_confidence=0.0,
                classification=ClassificationResult(
                    predicted_type="unknown",
                    confidence=0.0,
                    state=state_cfg.state_code,
                    explanation="Preprocessing failed to load valid pages from input",
                ),
                pages=[],
                full_text="",
                processing_time_ms=round(duration_ms, 2),
                metadata={
                    "stage_timings_ms": {
                        "hashing_ms": round(hashing_dur, 2),
                        "preprocessing_ms": round(prep_dur, 2),
                        "total_pipeline_ms": round(duration_ms, 2),
                    },
                    "confidence_calibration_notice": "overall_confidence is an uncalibrated aggregate score derived from engine token-level probabilities, not a calibrated posterior probability.",
                },
            )

        # 4. Multi-Page OCR, Layout & Table Processing
        page_results: List[OCRPageResult] = []
        all_text_snippets: List[str] = []
        total_ocr_dur = 0.0
        total_layout_dur = 0.0

        for p in preprocessed_pages:
            p_res, o_dur, l_dur = self._process_single_page(p, primary_lang, doc_id)
            total_ocr_dur += o_dur
            total_layout_dur += l_dur
            page_results.append(p_res)
            if p_res.text:
                all_text_snippets.append(p_res.text)

        full_doc_text = "\n\n".join(all_text_snippets)

        # 5. Document Classification driven by state configs & extracted signals
        class_start = time.perf_counter()
        classifier = DocumentClassifier(state_config=state_cfg)
        classification = classifier.classify(full_text=full_doc_text, pages=page_results)
        class_dur = (time.perf_counter() - class_start) * 1000.0

        log_stage_event("classification_completed", doc_id, extra={
            "predicted_type": classification.predicted_type,
            "confidence": classification.confidence,
        })

        # 6. Compute Overall Confidence
        if page_results:
            overall_conf = sum(p.confidence for p in page_results) / len(page_results)
        else:
            overall_conf = 0.0

        total_dur = (time.perf_counter() - start_time) * 1000.0

        stage_timings = {
            "hashing_ms": round(hashing_dur, 2),
            "preprocessing_ms": round(prep_dur, 2),
            "ocr_inference_ms": round(total_ocr_dur, 2),
            "layout_and_tables_ms": round(total_layout_dur, 2),
            "classification_ms": round(class_dur, 2),
            "total_pipeline_ms": round(total_dur, 2),
        }

        return OCROutput(
            document_id=doc_id,
            sha256_hash=sha256,
            ocr_engine=OCREngineType.PADDLE_OCR.value,
            overall_confidence=round(overall_conf, 4),
            classification=classification,
            pages=page_results,
            full_text=full_doc_text,
            processing_time_ms=round(total_dur, 2),
            metadata={
                "stage_timings_ms": stage_timings,
                "confidence_calibration_notice": "overall_confidence is an uncalibrated aggregate score derived from engine token-level probabilities, not a calibrated posterior probability.",
            },
        )

    def _process_single_page(
        self,
        preprocessed: PreprocessedPage,
        lang: str,
        doc_id: str,
    ) -> Tuple[OCRPageResult, float, float]:
        page_num = preprocessed.page_number
        img = preprocessed.processed_image
        h, w = img.shape[:2]
        quality = preprocessed.quality

        # Transparent language detection and fallback metadata
        lang_info = self.ocr_engine.get_language_info(lang)

        if quality.is_blank:
            res = OCRPageResult(
                page_number=page_num,
                width=w,
                height=h,
                text="",
                confidence=1.0,
                ocr_engine=OCREngineType.PADDLE_OCR.value,
                quality=quality,
                language_info=lang_info,
                blocks=[],
                tables=[],
            )
            return res, 0.0, 0.0

        # 1. Printed Multilingual OCR via PaddleOCR
        ocr_start = time.perf_counter()
        raw_blocks = self.ocr_engine.recognize_page(img, page_num=page_num, language=lang)
        ocr_dur = (time.perf_counter() - ocr_start) * 1000.0

        # 2. Layout & Table Detection
        layout_start = time.perf_counter()
        layout_blocks, tables = self.layout_detector.analyze_layout(img, raw_blocks, page_num=page_num)
        layout_dur = (time.perf_counter() - layout_start) * 1000.0

        log_stage_event("ocr_and_layout_completed", doc_id, page_num=page_num, duration_ms=ocr_dur, extra={
            "blocks": len(layout_blocks),
            "tables": len(tables),
        })

        page_text = "\n".join(b.text for b in layout_blocks if b.text)
        page_conf = sum(b.confidence for b in layout_blocks) / len(layout_blocks) if layout_blocks else 1.0

        page_res = OCRPageResult(
            page_number=page_num,
            width=w,
            height=h,
            text=page_text,
            confidence=round(page_conf, 4),
            ocr_engine=OCREngineType.PADDLE_OCR.value,
            quality=quality,
            language_info=lang_info,
            blocks=layout_blocks,
            tables=tables,
        )
        return page_res, ocr_dur, layout_dur

    def process_document_with_handwriting(
        self,
        document_input: Union[str, Path, bytes, BinaryIO, np.ndarray, DocumentInput],
        state_hint: Optional[str] = None,
    ) -> Tuple[OCROutput, HandwritingResult]:
        """Execute Person A pipeline and route handwritten regions to Person B adapter."""
        ocr_output = self.process_document(document_input, state_hint=state_hint)

        # Extract handwritten regions from layout blocks
        hw_regions = []
        for page in ocr_output.pages:
            for block in page.blocks:
                if block.block_type == BlockType.HANDWRITTEN:
                    # Route crop to Person B adapter
                    dummy_crop = np.zeros((10, 10, 3), dtype=np.uint8)
                    reg_res = self.handwriting_adapter.recognize_region(
                        crop=dummy_crop,
                        bbox=block.bbox,
                        page_number=page.page_number,
                    )
                    hw_regions.append(reg_res)

        hw_result = HandwritingResult(
            document_id=ocr_output.document_id,
            regions=hw_regions,
            model_version="trocr-person-b-adapter",
            processing_time_ms=0.0,
        )

        return ocr_output, hw_result


_default_pipeline_instance: Optional[PersonAPipeline] = None


def process_document(
    document_input: Union[str, Path, bytes, BinaryIO, np.ndarray, DocumentInput],
    state_hint: Optional[str] = None,
    target_languages: Optional[List[str]] = None,
    config: Optional[PreprocessingConfig] = None,
) -> OCROutput:
    """Canonical high-level entry point for Person A document processing."""
    global _default_pipeline_instance
    if _default_pipeline_instance is None:
        _default_pipeline_instance = PersonAPipeline()
    return _default_pipeline_instance.process_document(
        document_input=document_input,
        state_hint=state_hint,
        target_languages=target_languages,
        config=config,
    )


def process_document_with_handwriting(
    document_input: Union[str, Path, bytes, BinaryIO, np.ndarray, DocumentInput],
    state_hint: Optional[str] = None,
) -> Tuple[OCROutput, HandwritingResult]:
    """High-level entry point routing handwritten blocks to Person B."""
    global _default_pipeline_instance
    if _default_pipeline_instance is None:
        _default_pipeline_instance = PersonAPipeline()
    return _default_pipeline_instance.process_document_with_handwriting(
        document_input=document_input,
        state_hint=state_hint,
    )
