"""
src/integration/walking_skeleton.py
End-to-end Walking Skeleton connecting Person A, Person B, and Person C.
Executes the full pipeline:
RAW DOC -> HASH -> PREPROCESS -> PRINTED OCR -> LAYOUT -> CLASSIFY -> HANDWRITING -> EXTRACTION -> VALIDATION -> CONFIDENCE -> FINAL RESULT
"""

import time
from typing import Any, Dict, List, Optional
from loguru import logger

from schemas import (
    BoundingBox,
    DocumentClassificationResult,
    DocumentInput,
    DocumentOCRResult,
    DocumentType,
    FinalDocumentResult,
    HandwritingRegionResult,
    HandwritingResult,
    OCREngineType,
    OCRTextLine,
    PreprocessingMetadata,
    TableStructure,
)
from src.integration.person_c_service import extract_and_validate
from src.utils.hashing import compute_sha256_bytes, compute_sha256_text
from src.utils.logger import PipelineLogger


class PersonAStub:
    """
    Person A Walking-Skeleton interface.
    Generates DocumentOCRResult from input documents.
    """

    def process_document(self, doc_input: DocumentInput) -> DocumentOCRResult:
        doc_id = doc_input.document_id
        sha256 = doc_input.sha256_hash or (
            compute_sha256_bytes(doc_input.file_bytes) if doc_input.file_bytes
            else compute_sha256_text(doc_id)
        )

        # Preprocessing simulation
        preproc = PreprocessingMetadata(
            deskew_angle=0.4,
            super_resolution_applied=False,
            denoised=True,
            contrast_enhanced=True,
            original_resolution=(1800, 2400),
            processed_resolution=(1800, 2400),
            rotation_degrees=0,
            page_count=1,
        )

        # Sample standard text lines for UP Khatauni if no text provided
        raw_text = doc_input.metadata.get("sample_text", """
उत्तर प्रदेश शासन - राजस्व परिषद
खातौनी (अधिकार अभिलेख)
ग्राम का नाम: मऊ  परगना: मोहनलालगंज  तहसील: मोहनलालगंज  जनपद: लखनऊ
फसली वर्ष: 1428-1433
खाता संख्या: 00124
खातेदार का नाम: राम प्रसाद
पिता का नाम: श्याम लाल
गाटा संख्या: 142/1
क्षेत्रफल (हेक्टेयर): 0.4500
        """).strip()

        lines: List[OCRTextLine] = []
        for idx, line_str in enumerate(raw_text.split("\n")):
            line_str = line_str.strip()
            if not line_str:
                continue
            lines.append(
                OCRTextLine(
                    text=line_str,
                    confidence=0.96,
                    bbox=BoundingBox(x_min=50.0, y_min=100.0 + idx * 35.0, x_max=600.0, y_max=130.0 + idx * 35.0),
                    page_number=1,
                    language="hi",
                    engine=OCREngineType.PADDLE_OCR,
                    source_region="body",
                )
            )

        # Classification
        classification = DocumentClassificationResult(
            document_type=DocumentType.KHATAUNI,
            state=doc_input.selected_state or "UP",
            confidence=0.98,
            signals_matched=["खातौनी", "गाटा संख्या", "उत्तर प्रदेश"],
            language="hi",
        )

        return DocumentOCRResult(
            document_id=doc_id,
            sha256_hash=sha256,
            preprocessing=preproc,
            classification=classification,
            text_lines=lines,
            tables=[],
            raw_full_text=raw_text,
            pages_processed=1,
            processing_time_ms=120.0,
        )


class PersonBStub:
    """
    Person B Walking-Skeleton interface.
    Generates HandwritingResult from handwritten regions.
    """

    def recognize_handwriting(
        self, doc_id: str, ocr_result: DocumentOCRResult
    ) -> HandwritingResult:
        # Example handwritten annotation / filled area
        regions: List[HandwritingRegionResult] = []
        
        # If there's an annotated handwriting sample in metadata
        return HandwritingResult(
            document_id=doc_id,
            regions=regions,
            model_version="trocr-base-landrecords-v1",
            processing_time_ms=85.0,
        )


class EndToEndPipeline:
    """
    Orchestrates Person A, Person B, and Person C into the complete integrated flow.
    """

    def __init__(self):
        self.person_a = PersonAStub()
        self.person_b = PersonBStub()

    def process(self, doc_input: DocumentInput) -> FinalDocumentResult:
        logger_wrapper = PipelineLogger(doc_input.document_id)
        
        # 1. Hashing & Receipt
        sha256 = doc_input.sha256_hash or (
            compute_sha256_bytes(doc_input.file_bytes) if doc_input.file_bytes
            else compute_sha256_text(doc_input.document_id)
        )
        doc_input.sha256_hash = sha256
        logger_wrapper.doc_received(sha256, doc_input.selected_state)

        # 2. Person A: Preprocessing + OCR + Layout + Classification
        logger_wrapper.preprocessing_started()
        ocr_result = self.person_a.process_document(doc_input)
        logger_wrapper.preprocessing_completed(ocr_result.preprocessing.model_dump())
        logger_wrapper.ocr_completed(len(ocr_result.text_lines), len(ocr_result.tables))
        logger_wrapper.classification_completed(
            ocr_result.classification.document_type.value,
            ocr_result.classification.state,
            ocr_result.classification.confidence,
        )

        # 3. Person B: Handwriting Recognition (TrOCR)
        logger_wrapper.handwriting_started()
        hw_result = self.person_b.recognize_handwriting(doc_input.document_id, ocr_result)
        logger_wrapper.handwriting_completed(len(hw_result.regions))

        # 4. Person C: Extraction + Validation + Duplicates + GIS + Confidence
        final_result = extract_and_validate(
            ocr_result=ocr_result,
            handwriting_result=hw_result,
            selected_state=doc_input.selected_state,
        )

        return final_result
