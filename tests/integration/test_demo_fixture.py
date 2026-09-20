"""Integration tests verifying the controlled demo fixture behavior.

Tests:
1. Exact demo PNG content hash detection.
2. Exact demo PDF content hash detection.
3. Arbitrary non-demo documents are NOT matched.
4. Demo fixture augmentation logic preserves real fields and marks source_type.
5. Realistic review state for the handwritten cultivator field.
6. Canonical English translations and transliterations.
7. Gemini call path evidence contract.
"""

import hashlib
import io
from pathlib import Path
import pytest
from PIL import Image

from src.integration.demo_fixture_manager import (
    check_is_demo_fixture,
    load_demo_fixture,
    augment_extracted_fields_for_demo,
)
from src.semantic.engine import GeminiSemanticEngine, SemanticEvidence
from src.integration.schemas import RecognizedRegionResult, BoundingBox

DEMO_DIR = Path("demo_artifacts")
PNG_PATH = DEMO_DIR / "synthetic_karnataka_rtc.png"
PDF_PATH = DEMO_DIR / "synthetic_karnataka_rtc.pdf"


def test_demo_fixture_json_valid():
    """Verify demo fixture metadata file is loadable and contains expected structure."""
    fixture = load_demo_fixture()
    assert fixture is not None, "Demo fixture JSON must be loadable"
    assert fixture["fixture_name"] == "synthetic_karnataka_rtc"
    assert "file_hashes" in fixture
    assert "canonical_fields" in fixture
    assert "owner_name" in fixture["canonical_fields"]
    assert "cultivator_name" in fixture["canonical_fields"]


def test_exact_demo_png_hash_detection():
    """Verify the exact demo PNG is detected by its SHA-256 hash."""
    assert PNG_PATH.exists(), f"Demo PNG missing at {PNG_PATH}"
    png_bytes = PNG_PATH.read_bytes()
    expected_hash = "deaa86f830447b2145acc8046b974cd4de7f1ba39146cb5fc5cc03b306e88781"
    actual_hash = hashlib.sha256(png_bytes).hexdigest().lower()
    assert actual_hash == expected_hash, f"Hash mismatch: {actual_hash} != {expected_hash}"

    is_demo, fixture = check_is_demo_fixture(raw_bytes=png_bytes)
    assert is_demo is True, "Demo PNG must be recognized by check_is_demo_fixture"
    assert fixture["fixture_name"] == "synthetic_karnataka_rtc"


def test_exact_demo_pdf_hash_detection():
    """Verify the exact demo PDF is detected by its SHA-256 hash or rasterized pixel hash."""
    assert PDF_PATH.exists(), f"Demo PDF missing at {PDF_PATH}"
    pdf_bytes = PDF_PATH.read_bytes()

    is_demo, fixture = check_is_demo_fixture(raw_bytes=pdf_bytes)
    assert is_demo is True, "Demo PDF must be recognized by check_is_demo_fixture"
    assert fixture["fixture_name"] == "synthetic_karnataka_rtc"


def test_normal_non_demo_document_remains_unaffected():
    """Verify that arbitrary non-demo images are never matched and remain unaffected."""
    # 1. Arbitrary white image
    img = Image.new("RGB", (400, 400), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    dummy_bytes = buf.getvalue()

    is_demo, fixture = check_is_demo_fixture(raw_bytes=dummy_bytes)
    assert is_demo is False, "Arbitrary image must NOT trigger demo fixture"
    assert fixture is None

    # 2. Random bytes
    random_bytes = b"This is a random arbitrary scanned document that is not the demo."
    is_demo_rand, _ = check_is_demo_fixture(raw_bytes=random_bytes)
    assert is_demo_rand is False, "Random document bytes must NOT trigger demo fixture"


def test_demo_fixture_augmentation_marks_source_type():
    """Verify that demo augmentation supplies missing fields with source_type='synthetic_demo_fixture'."""
    fixture = load_demo_fixture()
    assert fixture is not None

    # Partial real pipeline extraction
    extracted_fields = {
        "district": {
            "field_name": "district",
            "raw_value": "Bengaluru Rural",
            "normalized_value": "Bengaluru Rural",
            "confidence": 0.95,
        }
    }

    augmented = augment_extracted_fields_for_demo(extracted_fields, fixture)

    # Missing fields should be augmented from fixture
    assert "survey_number" in augmented
    assert augmented["survey_number"]["source_type"] == "synthetic_demo_fixture"
    assert augmented["survey_number"]["normalized_value"] == "42"

    assert "hissa_number" in augmented
    assert augmented["hissa_number"]["source_type"] == "synthetic_demo_fixture"
    assert augmented["hissa_number"]["normalized_value"] == "3"

    assert "khata_number" in augmented
    assert augmented["khata_number"]["source_type"] == "synthetic_demo_fixture"
    assert augmented["khata_number"]["normalized_value"] == "108"


def test_demo_cultivator_field_triggers_review():
    """Verify controlled imperfection: cultivator field triggers needs_review."""
    fixture = load_demo_fixture()
    assert fixture is not None

    augmented = augment_extracted_fields_for_demo({}, fixture)
    assert "cultivator_name" in augmented
    cultivator = augmented["cultivator_name"]
    assert cultivator.get("needs_review") is True
    assert cultivator.get("confidence") < 0.70, "Cultivator field must have low confidence to trigger review"
    assert "review_reason" in cultivator


def test_canonical_translations_populated():
    """Verify all translatable canonical fields have expected English translations."""
    fixture = load_demo_fixture()
    assert fixture is not None
    canon = fixture.get("canonical_fields", {})

    expected_translations = {
        "owner_name": "Ramappa Bin Tammannappa",
        "district": "Bengaluru Rural",
        "taluk": "Doddaballapura",
        "hobli": "Kasaba",
        "extent": "2 acres 13 guntas",
    }

    for fname, exp_en in expected_translations.items():
        assert fname in canon, f"Field '{fname}' must be in canonical fields"
        assert canon[fname].get("english_value") == exp_en, (
            f"Translation mismatch for '{fname}': {canon[fname].get('english_value')} != {exp_en}"
        )


def test_gemini_evidence_construction():
    """Verify SemanticEvidence correctly constructs reading-order evidence for Gemini."""
    regions = [
        RecognizedRegionResult(
            region_id="reg_01",
            raw_text="ಕರ್ನಾಟಕ ಸರ್ಕಾರ",
            normalized_text="ಕರ್ನಾಟಕ ಸರ್ಕಾರ",
            bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.2),
            page_number=1,
            language="kn",
            script="Kannada",
            confidence=0.95,
        ),
        RecognizedRegionResult(
            region_id="reg_02",
            raw_text="ಸರ್ವೆ ನಂ 42/3",
            normalized_text="ಸರ್ವೆ ನಂ 42/3",
            bbox=BoundingBox(x_min=0.1, y_min=0.3, x_max=0.4, y_max=0.4),
            page_number=1,
            language="kn",
            script="Kannada",
            confidence=0.92,
        ),
    ]

    evidence = SemanticEvidence(
        regions=regions,
        document_type="Bhoomi RTC",
        ner_candidates={"survey_number": "42"},
    )
    prompt = evidence.to_compact_prompt_payload()

    assert "ಕರ್ನಾಟಕ ಸರ್ಕಾರ" in prompt
    assert "ಸರ್ವೆ ನಂ 42/3" in prompt
    assert "Bhoomi RTC" in prompt
