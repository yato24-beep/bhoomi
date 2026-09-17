"""
tests/unit/test_semantic_understanding.py
Unit tests for the Semantic Understanding and Contextual Extraction Layer.
"""

import pytest
from schemas import (
    BoundingBox,
    DocumentClassificationResult,
    DocumentOCRResult,
    DocumentType,
    ExtractedField,
    ExtractionMethod,
    FieldEvidence,
    OCREngineType,
    OCRTextLine,
    ValidationStatus,
)
from src.extraction.semantic_understanding import (
    MultilingualSemanticEngine,
    SemanticExtractionResult,
    SemanticFieldExtraction,
    fuse_semantic_and_regex_fields,
    semantic_understand,
)
from src.integration.person_c_service import extract_and_validate


def test_noisy_kannada_paragraph_semantic_understanding():
    """Validates semantic extraction from severe OCR noise, corrupted tokens, merged words,
    and unstructured Kannada paragraph without explicit field labels.
    """
    noisy_kannada_text = (
        "ಜಯನಗರಿಉಪಏಭಾಗ ... ಜಯನಗರ ... ಸೈಬ್ವೇ-ನಂಬರ್-.2- ... ನಿವೇಶನ ... "
        "ಸಹಾಯಕಕಂದಾಯಿಅಧಕಾರ ... ಬೃಹತ್ಬೆಂಗಳೂದುಮಹಾನಗರಪಾ ... ಬೆಂಗಳೂದು"
    )

    regions = [
        OCRTextLine(
            text="ಜಯನಗರಿಉಪಏಭಾಗ ಜಯನಗರ",
            confidence=0.88,
            bbox=BoundingBox(x_min=10, y_min=20, x_max=300, y_max=50),
            page_number=1,
            language="kn",
        ),
        OCRTextLine(
            text="ಸೈಬ್ವೇ-ನಂಬರ್-.2- ನಿವೇಶನ",
            confidence=0.85,
            bbox=BoundingBox(x_min=10, y_min=60, x_max=250, y_max=90),
            page_number=1,
            language="kn",
        ),
        OCRTextLine(
            text="ಸಹಾಯಕಕಂದಾಯಿಅಧಕಾರ",
            confidence=0.82,
            bbox=BoundingBox(x_min=10, y_min=100, x_max=350, y_max=130),
            page_number=1,
            language="kn",
        ),
        OCRTextLine(
            text="ಬೃಹತ್ಬೆಂಗಳೂದುಮಹಾನಗರಪಾ ಬೆಂಗಳೂದು",
            confidence=0.90,
            bbox=BoundingBox(x_min=10, y_min=140, x_max=450, y_max=170),
            page_number=1,
            language="kn",
        ),
    ]

    result: SemanticExtractionResult = semantic_understand(
        ordered_ocr_text=noisy_kannada_text,
        regions=regions,
    )

    assert isinstance(result, SemanticExtractionResult)
    fields = result.fields

    # 1. Survey Number around '2' correctly inferred from 'ಸೈಬ್ವೇ-ನಂಬರ್-.2- ... ನಿವೇಶನ'
    assert "survey_number" in fields
    assert fields["survey_number"].extracted_value == "2"
    assert fields["survey_number"].confidence >= 0.80
    assert fields["survey_number"].bbox is not None
    assert fields["survey_number"].bbox.y_min == 60

    # 2. Locality identified as Jayanagar
    assert "locality" in fields
    assert fields["locality"].extracted_value == "Jayanagar"

    # 3. Taluk / Sub-division identified as Jayanagar Sub-division from 'ಜಯನಗರಿಉಪಏಭಾಗ'
    assert "taluk" in fields
    assert "Jayanagar" in fields["taluk"].extracted_value

    # 4. District / Corporation identified from 'ಬೃಹತ್ಬೆಂಗಳೂದುಮಹಾನಗರಪಾ' / 'ಬೆಂಗಳೂದು'
    assert "district" in fields
    assert any(k in fields["district"].extracted_value for k in ["Bengaluru", "Bruhat Bengaluru", "BBMP"])

    # 5. Issuing Authority identified from 'ಸಹಾಯಕಕಂದಾಯಿಅಧಕಾರ'
    assert "issuing_authority" in fields
    assert "ಸಹಾಯಕ ಕಂದಾಯ ಅಧಿಕಾರಿ" in fields["issuing_authority"].extracted_value or "Assistant Revenue Officer" in fields["issuing_authority"].extracted_value

    # 6. Absence / Non-hallucination verification
    # Absent fields must NOT be fabricated
    assert "date" not in fields or fields["date"].extracted_value is None
    assert "built_up_area" not in fields or fields["built_up_area"].extracted_value is None


def test_mixed_bilingual_document_preservation():
    """Validates that semantic layer handles clean/mixed Kannada-English land records
    and preserves structured fields like owner_name, property_number, site_area, ward, date.
    """
    mixed_doc_text = """
    ಬೃಹತ್ ಬೆಂಗಳೂರು ಮಹಾನಗರ ಪಾಲಿಕೆ
    ದೃಢೀಕರಣ ಪತ್ರ / Khata Certificate
    ದಿನಾಂಕ: 20-09-2024
    Mrs. Dorothy Charles ರವರ ಹೆಸರಿನಲ್ಲಿ ದಾಖಲಾಗಿರುತ್ತದೆ
    ಆಸ್ತಿ ಸಂಖ್ಯೆ: 68-76-470/a
    ವಾರ್ಡ್ ಸಂಖ್ಯೆ: 148-Ejipura
    ನಿವೇಶನದ ವಿಸ್ತೀರ್ಣ: 2200.00 Sq Ft
    ಕಟ್ಟಡದ ವಿಸ್ತೀರ್ಣ: 500.00 Sq Ft
    ಸ್ಥಳ: S.T. Bed, Koramangala, Bengaluru
    """

    result: SemanticExtractionResult = semantic_understand(
        ordered_ocr_text=mixed_doc_text,
    )

    fields = result.fields

    # Owner name
    assert "owner_name" in fields
    assert "Dorothy Charles" in fields["owner_name"].extracted_value

    # Khata / Property number
    assert "khata_number" in fields
    assert fields["khata_number"].extracted_value == "68-76-470/a"

    # Date
    assert "date" in fields
    assert fields["date"].extracted_value == "20-09-2024"

    # Ward
    assert "ward" in fields
    assert "148" in fields["ward"].extracted_value or "Ejipura" in fields["ward"].extracted_value

    # Site area & Built-up area
    assert "site_area" in fields
    assert "2200" in fields["site_area"].extracted_value

    assert "built_up_area" in fields
    assert "500" in fields["built_up_area"].extracted_value

    # Document type
    assert result.document_type == "Khata Certificate"


def test_zero_hallucination_and_noise_filtering():
    """Validates that greetings, decorative text, and unrelated noise are filtered out
    and no false land record fields are hallucinated.
    """
    noise_only_text = """
    ನಮಸ್ಕಾರ!
    ಧನ್ಯವಾದಗಳು
    *** ---------------- ***
    ಪುಟ ಸಂಖ್ಯೆ: 1
    ಅಧಿಕಾರಿಯ ಸಹಿ
    """

    result: SemanticExtractionResult = semantic_understand(
        ordered_ocr_text=noise_only_text,
    )

    # All legitimate fields should be empty / null
    assert len(result.fields) == 0
    # Noise indicators should be registered
    assert len(result.irrelevant_noise_detected) >= 2


def test_fusion_with_person_c_pipeline():
    """Validates full end-to-end Person C pipeline execution with the new semantic understanding layer."""
    noisy_kannada_lines = [
        OCRTextLine(
            text="ಜಯನಗರಿಉಪಏಭಾಗ ಜಯನಗರ",
            confidence=0.90,
            bbox=BoundingBox(x_min=10, y_min=20, x_max=300, y_max=50),
            page_number=1,
            language="kn",
        ),
        OCRTextLine(
            text="ಸೈಬ್ವೇ-ನಂಬರ್-.2- ನಿವೇಶನ",
            confidence=0.88,
            bbox=BoundingBox(x_min=10, y_min=60, x_max=250, y_max=90),
            page_number=1,
            language="kn",
        ),
        OCRTextLine(
            text="ಸಹಾಯಕಕಂದಾಯಿಅಧಕಾರ ಬೃಹತ್ಬೆಂಗಳೂದುಮಹಾನಗರಪಾ",
            confidence=0.91,
            bbox=BoundingBox(x_min=10, y_min=100, x_max=450, y_max=130),
            page_number=1,
            language="kn",
        ),
    ]

    ocr_res = DocumentOCRResult(
        document_id="DOC_SEMANTIC_TEST_001",
        sha256_hash="hash_semantic_001",
        text_lines=noisy_kannada_lines,
        raw_full_text="\n".join(l.text for l in noisy_kannada_lines),
    )

    final_doc = extract_and_validate(
        ocr_result=ocr_res,
        selected_state="KA",
    )

    # Verify pipeline stages completed
    assert "semantic_understanding" in final_doc.pipeline_stages_completed
    assert "extraction" in final_doc.pipeline_stages_completed
    assert "normalization" in final_doc.pipeline_stages_completed
    assert "confidence_scoring" in final_doc.pipeline_stages_completed

    # Verify extracted fields
    assert "khasra_number" in final_doc.fields or "survey_number" in final_doc.fields
    survey_field = final_doc.fields.get("khasra_number") or final_doc.fields.get("survey_number")
    assert survey_field.normalized_value == "2"
    assert survey_field.extraction_method in (ExtractionMethod.SEMANTIC_UNDERSTANDING, ExtractionMethod.REGEX)

    assert "district" in final_doc.fields
    assert "Bengaluru" in str(final_doc.fields["district"].normalized_value)

    assert "locality" in final_doc.fields
    assert "Jayanagar" in str(final_doc.fields["locality"].normalized_value)

    assert final_doc.overall_confidence >= 0.70
