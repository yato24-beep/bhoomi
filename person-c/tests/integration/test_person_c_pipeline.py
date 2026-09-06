"""
Integration tests for Person C extract_and_validate interface.
"""

import pytest
from schemas import (
    BoundingBox,
    DocumentClassificationResult,
    DocumentOCRResult,
    DocumentType,
    OCREngineType,
    OCRTextLine,
    ValidationStatus,
)
from src.integration.person_c_service import extract_and_validate


def test_person_c_full_up_khatauni_flow():
    # Construct realistic UP Khatauni OCR result
    lines = [
        OCRTextLine(
            text="उत्तर प्रदेश शासन राजस्व परिषद",
            confidence=0.99,
            bbox=BoundingBox(x_min=100, y_min=50, x_max=500, y_max=80),
            engine=OCREngineType.PADDLE_OCR,
        ),
        OCRTextLine(
            text="ग्राम: मऊ  तहसील: मोहनलालगंज  जनपद: लखनऊ",
            confidence=0.98,
            bbox=BoundingBox(x_min=50, y_min=90, x_max=600, y_max=120),
            engine=OCREngineType.PADDLE_OCR,
        ),
        OCRTextLine(
            text="खाता संख्या: 00124",
            confidence=0.96,
            bbox=BoundingBox(x_min=50, y_min=130, x_max=250, y_max=160),
            engine=OCREngineType.PADDLE_OCR,
        ),
        OCRTextLine(
            text="खातेदार का नाम: श्री राम प्रसाद  पिता: श्याम लाल",
            confidence=0.95,
            bbox=BoundingBox(x_min=50, y_min=170, x_max=550, y_max=200),
            engine=OCREngineType.PADDLE_OCR,
        ),
        OCRTextLine(
            text="गाटा संख्या: 142/1  क्षेत्रफल: 0.4500 हेक्टेयर",
            confidence=0.97,
            bbox=BoundingBox(x_min=50, y_min=210, x_max=550, y_max=240),
            engine=OCREngineType.PADDLE_OCR,
        ),
    ]

    ocr_input = DocumentOCRResult(
        document_id="DOC_INT_UP_001",
        sha256_hash="hash_up_khatauni_001",
        classification=DocumentClassificationResult(
            document_type=DocumentType.KHATAUNI,
            state="UP",
            confidence=0.99,
        ),
        text_lines=lines,
        raw_full_text="\n".join(l.text for l in lines),
    )

    final_result = extract_and_validate(ocr_input, selected_state="UP")

    assert final_result.document_id == "DOC_INT_UP_001"
    assert final_result.state == "UP"
    assert final_result.document_type == DocumentType.KHATAUNI
    assert final_result.validation_status == ValidationStatus.VALID

    # Check extracted fields
    assert "khasra_number" in final_result.fields
    assert final_result.fields["khasra_number"].raw_value == "142/1"
    assert final_result.fields["khasra_number"].normalized_value == "142/1"

    assert "owner_name" in final_result.fields
    assert final_result.fields["owner_name"].raw_value == "श्री राम प्रसाद"
    assert final_result.fields["owner_name"].normalized_value == "राम प्रसाद"

    assert "land_area" in final_result.fields
    assert final_result.fields["land_area"].normalized_value == 0.4500
    assert final_result.fields["land_area"].normalized_unit == "hectare"

    # Evidence traceability checks
    assert final_result.fields["khasra_number"].evidence.bbox is not None
    assert final_result.fields["khasra_number"].evidence.raw_ocr_text != ""

    # GIS verification check
    assert final_result.gis_validation.is_verified is True
    assert final_result.gis_validation.gis_recorded_area_hectares == 0.4500

    # Confidence check
    assert final_result.overall_confidence >= 0.85
    assert final_result.requires_human_review is False


def test_person_c_gis_mismatch_flags_review():
    lines = [
        OCRTextLine(
            text="ग्राम: मऊ  तहसील: मोहनलालगंज  जनपद: लखनऊ",
            confidence=0.98,
            bbox=BoundingBox(x_min=50, y_min=90, x_max=600, y_max=120),
        ),
        OCRTextLine(
            text="खातेदार का नाम: राम प्रसाद",
            confidence=0.95,
            bbox=BoundingBox(x_min=50, y_min=170, x_max=550, y_max=200),
        ),
        OCRTextLine(
            text="गाटा संख्या: 142/1  क्षेत्रफल: 5.5000 हेक्टेयर",  # Mismatched area (registered is 0.45 ha)
            confidence=0.97,
            bbox=BoundingBox(x_min=50, y_min=210, x_max=550, y_max=240),
        ),
    ]

    ocr_input = DocumentOCRResult(
        document_id="DOC_INT_MISMATCH",
        sha256_hash="hash_mismatch_002",
        classification=DocumentClassificationResult(
            document_type=DocumentType.KHATAUNI,
            state="UP",
        ),
        text_lines=lines,
        raw_full_text="\n".join(l.text for l in lines),
    )

    final_result = extract_and_validate(ocr_input, selected_state="UP")

    assert final_result.gis_validation.has_mismatch is True
    assert final_result.requires_human_review is True
    assert any("GIS Area Discrepancy" in r for r in final_result.review_reasons)
