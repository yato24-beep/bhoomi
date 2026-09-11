"""
tests/integration/test_gis_demo.py
Comprehensive multi-state land record and Cadastral GIS demonstration tests:
- Karnataka (Bhoomi RTC / Pahani)
- Tamil Nadu (Patta / Chitta / 'A' Register)
- Maharashtra (7/12 Satbara Extract)
- 3 Mandatory GIS Demo Cases (MATCH, MISMATCH, UNKNOWN).
"""

import pytest
from schemas import (
    BoundingBox,
    DocumentClassificationResult,
    DocumentOCRResult,
    DocumentType,
    GISStatus,
    OCREngineType,
    OCRTextLine,
    ValidationStatus,
)
from src.integration.person_c_service import extract_and_validate


def test_karnataka_bhoomi_rtc_pipeline():
    """Demonstrates complete Karnataka Bhoomi RTC processing in Kannada & English."""
    lines = [
        OCRTextLine(
            text="ಕರ್ನಾಟಕ ಸರ್ಕಾರ - ಕಂದಾಯ ಇಲಾಖೆ",
            confidence=0.99,
            bbox=BoundingBox(x_min=50, y_min=20, x_max=500, y_max=50),
        ),
        OCRTextLine(
            text="ಭೂಮಿ - ಅಧಿಕಾರ ಮತ್ತು ಸ್ವಾಧೀನತೆ ದಾಖಲೆ (RTC / ಪಹಣಿ)",
            confidence=0.98,
            bbox=BoundingBox(x_min=50, y_min=60, x_max=600, y_max=90),
        ),
        OCRTextLine(
            text="ಜಿಲ್ಲೆ: BENGALURU URBAN  ತಾಲೂಕು: BANGALORE SOUTH  ಹೋಬಳಿ: KENGERI  ಗ್ರಾಮ: KENGERI",
            confidence=0.97,
            bbox=BoundingBox(x_min=50, y_min=100, x_max=700, y_max=130),
        ),
        OCRTextLine(
            text="ಖಾತೆದಾರರ ಹೆಸರು: ಸಿದ್ದರಾಮಯ್ಯ  ತಂದೆಯ ಹೆಸರು: ಬಸವರಾಜ",
            confidence=0.96,
            bbox=BoundingBox(x_min=50, y_min=140, x_max=550, y_max=170),
        ),
        OCRTextLine(
            text="ಸರ್ವೆ ನಂ: 42/1  ಒಟ್ಟು ವಿಸ್ತೀರ್ಣ: 2 ಎಕರೆ",
            confidence=0.97,
            bbox=BoundingBox(x_min=50, y_min=180, x_max=500, y_max=210),
        ),
    ]

    ocr_input = DocumentOCRResult(
        document_id="DOC_DEMO_KA_001",
        sha256_hash="hash_ka_bhoomi_001",
        classification=DocumentClassificationResult(
            document_type=DocumentType.BHOOMI_RTC,
            state="KA",
            confidence=0.99,
        ),
        text_lines=lines,
        raw_full_text="\n".join(l.text for l in lines),
    )

    res = extract_and_validate(ocr_input, selected_state="KA")

    assert res.state == "KA"
    assert "khasra_number" in res.fields
    assert res.fields["khasra_number"].raw_value == "42/1"
    
    assert "owner_name" in res.fields
    assert "ಸಿದ್ದರಾಮಯ್ಯ" in res.fields["owner_name"].raw_value

    assert "land_area" in res.fields
    # 2 Acres = 2 * 0.404686 = 0.809372 Ha
    assert res.fields["land_area"].normalized_value == pytest.approx(0.8094, rel=1e-3)
    assert res.fields["land_area"].normalized_unit == "hectare"

    # GIS Verification against Karnataka Cadastral GeoJSON
    assert res.gis_validation.gis_status == GISStatus.MATCH
    assert res.gis_validation.is_verified is True
    assert res.gis_validation.gis_recorded_area_hectares == 0.8094


def test_tamil_nadu_patta_pipeline():
    """Demonstrates complete Tamil Nadu Patta/Chitta processing in Tamil & English."""
    lines = [
        OCRTextLine(
            text="தமிழ்நாடு அரசு - வருவாய்த்துறை",
            confidence=0.99,
            bbox=BoundingBox(x_min=50, y_min=20, x_max=500, y_max=50),
        ),
        OCRTextLine(
            text="இணையவழி பட்டா / சிட்டா சான்றிதழ்",
            confidence=0.98,
            bbox=BoundingBox(x_min=50, y_min=60, x_max=550, y_max=90),
        ),
        OCRTextLine(
            text="மாவட்டம்: KANCHIPURAM  வட்டம்: SRIPERUMBUDUR  கிராமம்: NEMILI",
            confidence=0.97,
            bbox=BoundingBox(x_min=50, y_min=100, x_max=700, y_max=130),
        ),
        OCRTextLine(
            text="பட்டா எண்: 304  பட்டாதாரர் பெயர்: முத்துக்குமார்",
            confidence=0.95,
            bbox=BoundingBox(x_min=50, y_min=140, x_max=550, y_max=170),
        ),
        OCRTextLine(
            text="புல எண்: 108/1  பரப்பளவு: 0.5000 ஹெக்டேர்",
            confidence=0.98,
            bbox=BoundingBox(x_min=50, y_min=180, x_max=500, y_max=210),
        ),
    ]

    ocr_input = DocumentOCRResult(
        document_id="DOC_DEMO_TN_001",
        sha256_hash="hash_tn_patta_001",
        classification=DocumentClassificationResult(
            document_type=DocumentType.PATTA_CHITTA,
            state="TN",
            confidence=0.99,
        ),
        text_lines=lines,
        raw_full_text="\n".join(l.text for l in lines),
    )

    res = extract_and_validate(ocr_input, selected_state="TN")

    assert res.state == "TN"
    assert "khasra_number" in res.fields
    assert res.fields["khasra_number"].raw_value == "108/1"

    assert "owner_name" in res.fields
    assert "முத்துக்குமார்" in res.fields["owner_name"].raw_value

    assert "land_area" in res.fields
    assert res.fields["land_area"].normalized_value == 0.5000

    # GIS Verification against Tamil Nadu Cadastral GeoJSON
    assert res.gis_validation.gis_status == GISStatus.MATCH
    assert res.gis_validation.is_verified is True


def test_three_mandatory_gis_demo_cases():
    """
    Validates the 3 required demo scenarios:
    1. Correct parcel -> MATCH
    2. Incorrect parcel/location -> MISMATCH (with visible reason)
    3. Missing GIS reference -> UNKNOWN
    """
    # CASE 1: MATCH (Maharashtra Wagholi Gat 45)
    lines_match = [
        OCRTextLine(text="गाव: WAGHOLI  तालुका: HAVELI  जिल्हा: PUNE", confidence=0.98, bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=20)),
        OCRTextLine(text="खातेदाराचे नाव: पाटील", confidence=0.95, bbox=BoundingBox(x_min=0, y_min=30, x_max=100, y_max=50)),
        OCRTextLine(text="गट क्रमांक: 45  क्षेत्र: 1.1500 हेक्टर", confidence=0.98, bbox=BoundingBox(x_min=0, y_min=60, x_max=100, y_max=80)),
    ]
    doc_match = DocumentOCRResult(
        document_id="CASE_1_MATCH",
        sha256_hash="h1",
        text_lines=lines_match,
        raw_full_text="\n".join(l.text for l in lines_match),
    )
    res_match = extract_and_validate(doc_match, selected_state="MH")
    assert res_match.gis_validation.gis_status == GISStatus.MATCH
    assert res_match.gis_validation.is_verified is True

    # CASE 2: MISMATCH (Claiming 5.5000 ha on a 1.1500 ha parcel)
    lines_mismatch = [
        OCRTextLine(text="गाव: WAGHOLI  तालुका: HAVELI  जिल्हा: PUNE", confidence=0.98, bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=20)),
        OCRTextLine(text="खातेदाराचे नाव: पाटील", confidence=0.95, bbox=BoundingBox(x_min=0, y_min=30, x_max=100, y_max=50)),
        OCRTextLine(text="गट क्रमांक: 45  क्षेत्र: 5.5000 हेक्टर", confidence=0.98, bbox=BoundingBox(x_min=0, y_min=60, x_max=100, y_max=80)),
    ]
    doc_mismatch = DocumentOCRResult(
        document_id="CASE_2_MISMATCH",
        sha256_hash="h2",
        text_lines=lines_mismatch,
        raw_full_text="\n".join(l.text for l in lines_mismatch),
    )
    res_mismatch = extract_and_validate(doc_mismatch, selected_state="MH")
    assert res_mismatch.gis_validation.gis_status == GISStatus.MISMATCH
    assert res_mismatch.gis_validation.has_mismatch is True
    assert any("GIS Area Discrepancy" in flag for flag in res_mismatch.gis_validation.flag_reasons)
    assert res_mismatch.requires_human_review is True

    # CASE 3: UNKNOWN (Non-existent Gat 99999)
    lines_unknown = [
        OCRTextLine(text="गाव: WAGHOLI  तालुका: HAVELI  जिल्हा: PUNE", confidence=0.98, bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=20)),
        OCRTextLine(text="खातेदाराचे नाव: पाटील", confidence=0.95, bbox=BoundingBox(x_min=0, y_min=30, x_max=100, y_max=50)),
        OCRTextLine(text="गट क्रमांक: 99999  क्षेत्र: 1.0000 हेक्टर", confidence=0.98, bbox=BoundingBox(x_min=0, y_min=60, x_max=100, y_max=80)),
    ]
    doc_unknown = DocumentOCRResult(
        document_id="CASE_3_UNKNOWN",
        sha256_hash="h3",
        text_lines=lines_unknown,
        raw_full_text="\n".join(l.text for l in lines_unknown),
    )
    res_unknown = extract_and_validate(doc_unknown, selected_state="MH")
    assert res_unknown.gis_validation.gis_status == GISStatus.UNKNOWN
    assert res_unknown.gis_validation.has_mismatch is False  # Missing GIS is NOT a mismatch!
    assert res_unknown.gis_validation.is_verified is False
