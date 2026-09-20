"""Comprehensive Unit and Integration Tests for AI-Driven Semantic Layer V1.

Verifies:
1. Unlabeled OCR line: "ರಾಮಪ್ಪ 493/2 2-15" -> owner_name, survey_number, extent
2. Multiple lines where meaning depends on neighboring lines
3. Table/cell-based evidence extraction
4. Missing values return null (no hallucinated values)
5. Conflicting owner candidates detection & review flagging
6. Conflicting survey numbers detection & review flagging
7. Invalid survey number detection (deterministic validation failure)
8. OCR noise resilience (whitespace, artifacts, noisy OCR tokens)
9. Kannada Unicode preservation (NFC normalization, conjuncts intact)
10. Provenance correctness (every field points to actual region_id, bbox, raw OCR text)
11. Gemini response validation failure handling (malformed JSON handled safely)
12. Gemini timeout / API failure handling (graceful fallback, zero crashes)
13. Empty / null field handling
14. Model output containing a value not found in OCR evidence (hallucination rejection!)
15. Deterministic normalization (Kannada numerals -> ASCII, raw preserved)

All tests execute OFFLINE using mocked Gemini engines. Zero API key required.
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
import pytest

from schemas import BoundingBox
from src.integration.schemas import RecognizedRegionResult
from src.semantic.confidence import EvidenceConfidenceCalculator
from src.semantic.engine import (
    BaseSemanticEngine,
    GeminiSemanticEngine,
    RuleSemanticEngine,
    SemanticEngineResult,
    SemanticEvidence,
    SemanticExtractedField,
)
from src.semantic.normalizer import SemanticNormalizer
from src.semantic.schema import (
    FieldProvenance,
    LandRecordDocument,
    OwnerRecord,
    SemanticFieldItem,
    SemanticTable,
    SemanticTableCell,
    ValidationStatus,
)
from src.semantic.semantic_pipeline import SemanticPipeline
from src.semantic.validator import FieldValidator


def make_region(
    region_id: str,
    raw_text: str,
    bbox: Optional[BoundingBox] = None,
    page_number: int = 1,
    confidence: float = 0.95,
) -> RecognizedRegionResult:
    """Helper to construct RecognizedRegionResult fixture."""
    box = bbox or BoundingBox(x_min=10.0, y_min=20.0, x_max=200.0, y_max=50.0)
    return RecognizedRegionResult(
        region_id=region_id,
        raw_text=raw_text,
        normalized_text=raw_text,
        bbox=box,
        page_number=page_number,
        language="kannada",
        script="Kannada",
        confidence=confidence,
    )


class MockSemanticEngine(BaseSemanticEngine):
    """Configurable mock semantic engine for unit tests."""

    def __init__(self, result: Optional[SemanticEngineResult] = None):
        self.result = result or SemanticEngineResult()

    def extract(self, evidence: SemanticEvidence) -> SemanticEngineResult:
        return self.result


# ============================================================================
# Test 1: Unlabeled OCR line: "ರಾಮಪ್ಪ 493/2 2-15"
# ============================================================================
def test_unlabeled_ocr_line_extraction():
    """Verifies that an unlabeled OCR line is correctly decomposed by semantic reasoning."""
    line_text = "ರಾಮಪ್ಪ 493/2 2-15"
    region = make_region("reg_unlabeled_01", line_text)

    mock_result = SemanticEngineResult(
        status="success",
        fields={
            "owner_name": SemanticExtractedField(
                field_name="owner_name",
                raw_value="ರಾಮಪ್ಪ",
                normalized_value="ರಾಮಪ್ಪ",
                source_region_id="reg_unlabeled_01",
            ),
            "survey_number": SemanticExtractedField(
                field_name="survey_number",
                raw_value="493/2",
                normalized_value="493/2",
                source_region_id="reg_unlabeled_01",
            ),
            "extent": SemanticExtractedField(
                field_name="extent",
                raw_value="2-15",
                normalized_value="2-15",
                source_region_id="reg_unlabeled_01",
            ),
        },
    )

    pipeline = SemanticPipeline(semantic_engine=MockSemanticEngine(mock_result))
    doc = pipeline.process(regions=[region], document_id="doc_unlabeled_01")

    assert doc.survey_number is not None
    assert doc.survey_number.value == "493/2"
    assert doc.survey_number.validation_status == ValidationStatus.VALID

    assert doc.owner_name is not None
    assert doc.owner_name.value == "ರಾಮಪ್ಪ"
    assert doc.owner_name.validation_status == ValidationStatus.VALID

    assert doc.extent is not None
    assert doc.extent.value == "2-15"
    assert doc.extent.validation_status == ValidationStatus.VALID

    # Check that owner is also in owners list
    assert len(doc.owners) == 1
    assert doc.owners[0].owner_name.value == "ರಾಮಪ್ಪ"


# ============================================================================
# Test 2: Multiple lines with contextual dependency
# ============================================================================
def test_multi_line_neighboring_context():
    """Verifies meaning inferred across multiple lines using neighboring context."""
    r1 = make_region("r_loc", "ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ ದೊಡ್ಡಬಳ್ಳಾಪುರ", BoundingBox(x_min=10, y_min=10, x_max=200, y_max=30))
    r2 = make_region("r_owner", "ಖಾತೆದಾರ: ಕೃಷ್ಣಪ್ಪ ಬಿನ್ ವೆಂಕಟಪ್ಪ", BoundingBox(x_min=10, y_min=40, x_max=200, y_max=60))
    r3 = make_region("r_survey", "ಸರ್ವೆ: 120/1A", BoundingBox(x_min=10, y_min=70, x_max=200, y_max=90))

    mock_result = SemanticEngineResult(
        status="success",
        fields={
            "district": SemanticExtractedField(field_name="district", raw_value="ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ", source_region_id="r_loc"),
            "taluk": SemanticExtractedField(field_name="taluk", raw_value="ದೊಡ್ಡಬಳ್ಳಾಪುರ", source_region_id="r_loc"),
            "owner_name": SemanticExtractedField(field_name="owner_name", raw_value="ಕೃಷ್ಣಪ್ಪ ಬಿನ್ ವೆಂಕಟಪ್ಪ", source_region_id="r_owner"),
            "survey_number": SemanticExtractedField(field_name="survey_number", raw_value="120/1A", source_region_id="r_survey"),
        },
    )

    pipeline = SemanticPipeline(semantic_engine=MockSemanticEngine(mock_result))
    doc = pipeline.process(regions=[r1, r2, r3], document_id="doc_multiline")

    assert doc.district.value == "ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ"
    assert doc.taluk.value == "ದೊಡ್ಡಬಳ್ಳಾಪುರ"
    assert doc.owner_name.value == "ಕೃಷ್ಣಪ್ಪ ಬಿನ್ ವೆಂಕಟಪ್ಪ"
    assert doc.survey_number.value == "120/1A"


# ============================================================================
# Test 3: Table / cell-based evidence
# ============================================================================
def test_table_cell_based_extraction():
    """Verifies table cell provenance linking in semantic extraction."""
    c1 = make_region("cell_sn", "45/1", BoundingBox(x_min=10, y_min=50, x_max=60, y_max=70))
    c2 = make_region("cell_ext", "1-30", BoundingBox(x_min=70, y_min=50, x_max=120, y_max=70))

    mock_result = SemanticEngineResult(
        status="success",
        fields={
            "survey_number": SemanticExtractedField(field_name="survey_number", raw_value="45/1", source_region_id="cell_sn"),
            "extent": SemanticExtractedField(field_name="extent", raw_value="1-30", source_region_id="cell_ext"),
        },
    )

    pipeline = SemanticPipeline(semantic_engine=MockSemanticEngine(mock_result))
    doc = pipeline.process(regions=[c1, c2], document_id="doc_table")

    assert doc.survey_number.value == "45/1"
    assert doc.survey_number.provenance.region_id == "cell_sn"
    assert doc.extent.value == "1-30"
    assert doc.extent.provenance.region_id == "cell_ext"


# ============================================================================
# Test 4: Missing values (insufficient evidence -> null)
# ============================================================================
def test_missing_values_return_null():
    """Verifies that unmentioned fields remain strictly null and are NEVER invented."""
    r = make_region("r_only_sn", "ಸರ್ವೆ ನಂ: 88/2")

    mock_result = SemanticEngineResult(
        status="success",
        fields={
            "survey_number": SemanticExtractedField(field_name="survey_number", raw_value="88/2", source_region_id="r_only_sn"),
        },
    )

    pipeline = SemanticPipeline(semantic_engine=MockSemanticEngine(mock_result))
    doc = pipeline.process(regions=[r], document_id="doc_missing")

    assert doc.survey_number is not None
    assert doc.survey_number.value == "88/2"

    # All unmentioned fields MUST be null
    assert doc.owner_name is None
    assert doc.cultivator_name is None
    assert doc.extent is None
    assert doc.village is None
    assert doc.taluk is None
    assert doc.district is None
    assert doc.mutation_number is None


# ============================================================================
# Test 5: Conflicting owner candidates
# ============================================================================
def test_conflicting_owner_candidates():
    """Verifies that conflicting owner candidates are detected and trigger warning."""
    r = make_region("r_owners", "ಮಾಲೀಕರು: ರಾಮಪ್ಪ / ಮಂಜುನಾಥ")

    mock_result = SemanticEngineResult(
        status="success",
        fields={
            "owner_name": SemanticExtractedField(
                field_name="owner_name",
                raw_value="ರಾಮಪ್ಪ",
                normalized_value="ರಾಮಪ್ಪ",
                source_region_id="r_owners",
                conflict_candidates=["ಮಂಜುನಾಥ"],
            ),
        },
    )

    pipeline = SemanticPipeline(semantic_engine=MockSemanticEngine(mock_result))
    doc = pipeline.process(regions=[r], document_id="doc_conflicts")

    assert doc.owner_name is not None
    assert doc.owner_name.conflicts == ["ಮಂಜುನಾಥ"]
    # Should reflect lower confidence due to conflict penalty
    assert doc.owner_name.confidence < 0.70


# ============================================================================
# Test 6: Conflicting survey numbers
# ============================================================================
def test_conflicting_survey_numbers():
    """Verifies conflicting survey number candidates are detected and penalized."""
    r = make_region("r_sn_conflict", "ಸರ್ವೆ 125 ಅಥವಾ 126")

    mock_result = SemanticEngineResult(
        status="success",
        fields={
            "survey_number": SemanticExtractedField(
                field_name="survey_number",
                raw_value="125",
                source_region_id="r_sn_conflict",
                conflict_candidates=["126"],
            ),
        },
    )

    pipeline = SemanticPipeline(semantic_engine=MockSemanticEngine(mock_result))
    doc = pipeline.process(regions=[r], document_id="doc_sn_conflict")

    assert doc.survey_number.conflicts == ["126"]
    assert doc.survey_number.confidence < 0.70


# ============================================================================
# Test 7: Invalid survey number
# ============================================================================
def test_invalid_survey_number():
    """Verifies that non-cadastral survey numbers fail deterministic validation."""
    r = make_region("r_bad_sn", "INVALID_TEXT_XYZ")

    mock_result = SemanticEngineResult(
        status="success",
        fields={
            "survey_number": SemanticExtractedField(
                field_name="survey_number",
                raw_value="INVALID_TEXT_XYZ",
                source_region_id="r_bad_sn",
            ),
        },
    )

    pipeline = SemanticPipeline(semantic_engine=MockSemanticEngine(mock_result))
    doc = pipeline.process(regions=[r], document_id="doc_invalid_sn")

    assert doc.survey_number.validation_status == ValidationStatus.INVALID
    assert "no numeric digits" in doc.survey_number.validation_reason.lower()
    assert doc.survey_number.confidence <= 0.25


# ============================================================================
# Test 8: OCR noise
# ============================================================================
def test_ocr_noise_resilience():
    """Verifies robust extraction and validation despite OCR whitespace and punctuation noise."""
    r = make_region("r_noisy", "   \t\n ಸರ್ವೆ ನಂ...  :  150/2B   \n ")

    mock_result = SemanticEngineResult(
        status="success",
        fields={
            "survey_number": SemanticExtractedField(
                field_name="survey_number",
                raw_value="150/2B",
                normalized_value="150/2B",
                source_region_id="r_noisy",
            ),
        },
    )

    pipeline = SemanticPipeline(semantic_engine=MockSemanticEngine(mock_result))
    doc = pipeline.process(regions=[r], document_id="doc_noisy")

    assert doc.survey_number.value == "150/2B"
    assert doc.survey_number.validation_status == ValidationStatus.VALID


# ============================================================================
# Test 9: Kannada Unicode preservation
# ============================================================================
def test_kannada_unicode_preservation():
    """Verifies Kannada conjuncts and decomposed Unicode characters are strictly preserved."""
    decomposed_name = "ದೊಡ್ಡಬಳ್ಳಾಪುರ\u0020ಕುಮಾರ್"
    r = make_region("r_unicode", decomposed_name)

    mock_result = SemanticEngineResult(
        status="success",
        fields={
            "owner_name": SemanticExtractedField(
                field_name="owner_name",
                raw_value="ದೊಡ್ಡಬಳ್ಳಾಪುರ ಕುಮಾರ್",
                normalized_value="ದೊಡ್ಡಬಳ್ಳಾಪುರ ಕುಮಾರ್",
                source_region_id="r_unicode",
            ),
        },
    )

    pipeline = SemanticPipeline(semantic_engine=MockSemanticEngine(mock_result))
    doc = pipeline.process(regions=[r], document_id="doc_unicode")

    assert "ಡ್ಡ" in doc.owner_name.value
    assert "ಳ್ಳಾ" in doc.owner_name.value
    assert doc.owner_name.validation_status == ValidationStatus.VALID


# ============================================================================
# Test 10: Provenance correctness
# ============================================================================
def test_provenance_correctness():
    """Verifies that every populated field references the actual source OCR region ID and coordinates."""
    box = BoundingBox(x_min=45.0, y_min=60.0, x_max=180.0, y_max=90.0)
    r = make_region("reg_provenance_test", "ಖಾತಾ 405", bbox=box)

    mock_result = SemanticEngineResult(
        status="success",
        fields={
            "khata_number": SemanticExtractedField(
                field_name="khata_number",
                raw_value="405",
                source_region_id="reg_provenance_test",
            ),
        },
    )

    pipeline = SemanticPipeline(semantic_engine=MockSemanticEngine(mock_result))
    doc = pipeline.process(regions=[r], document_id="doc_prov_101", page_number=2)

    khata = doc.khata_number
    assert khata.provenance is not None
    assert khata.provenance.document_id == "doc_prov_101"
    assert khata.provenance.page_number == 2
    assert khata.provenance.region_id == "reg_provenance_test"
    assert khata.provenance.bbox is not None
    assert khata.provenance.bbox.x_min == 45.0
    assert khata.provenance.raw_ocr_text == "ಖಾತಾ 405"


# ============================================================================
# Test 11: Gemini response validation failure (malformed JSON)
# ============================================================================
def test_gemini_response_validation_failure():
    """Verifies that malformed model JSON responses are caught without crashing."""
    gemini_engine = GeminiSemanticEngine(api_key="mock_key")
    res = gemini_engine._parse_response("INVALID_JSON_NOT_A_DICT{", default_doc_type="Unknown")

    assert res.status == "error"
    assert "Malformed model response JSON" in res.error
    assert len(res.fields) == 0


# ============================================================================
# Test 12: Gemini timeout / API failure handling
# ============================================================================
def test_gemini_timeout_api_failure():
    """Verifies that API failure triggers graceful fallback to RuleSemanticEngine."""
    r = make_region("r_fallback", "ಸರ್ವೆ ನಂ: 199/1")

    # Engine that returns an error status (simulating timeout or quota exceeded)
    failing_engine = MockSemanticEngine(SemanticEngineResult(status="error", error="API timeout after 30s"))

    pipeline = SemanticPipeline(semantic_engine=failing_engine)
    doc = pipeline.process(regions=[r], document_id="doc_timeout")

    # The pipeline should have gracefully fallen back to RuleSemanticEngine
    assert doc.validation_summary.get("fallback_used") is True
    assert doc.survey_number is not None
    assert doc.survey_number.value == "199/1"
    assert doc.survey_number.validation_status == ValidationStatus.VALID


# ============================================================================
# Test 13: Empty / null field handling
# ============================================================================
def test_empty_field_handling():
    """Verifies that completely empty region input yields empty structured document without crash."""
    pipeline = SemanticPipeline(semantic_engine=MockSemanticEngine(SemanticEngineResult(status="success")))
    doc = pipeline.process(regions=[], document_id="doc_empty")

    assert isinstance(doc, LandRecordDocument)
    assert doc.document_id == "doc_empty"
    assert len(doc.fields) == 0
    assert doc.validation_summary.get("total_extracted_fields") == 0


# ============================================================================
# Test 14: Model output containing value not found in OCR evidence (Hallucination Rejection!)
# ============================================================================
def test_hallucination_rejection():
    """Verifies that any value hallucinated by the model that does not exist in OCR evidence is rejected."""
    # Actual OCR is for "ರಮೇಶ್"
    r = make_region("r_real", "ರಮೇಶ್ 120/1")

    # Model hallucinates "ಸುರೇಶ್" (not present in OCR!)
    mock_result = SemanticEngineResult(
        status="success",
        fields={
            "owner_name": SemanticExtractedField(
                field_name="owner_name",
                raw_value="ಸುರೇಶ್",  # Hallucination!
                source_region_id="r_real",
            ),
            "survey_number": SemanticExtractedField(
                field_name="survey_number",
                raw_value="120/1",
                source_region_id="r_real",
            ),
        },
    )

    pipeline = SemanticPipeline(semantic_engine=MockSemanticEngine(mock_result))
    doc = pipeline.process(regions=[r], document_id="doc_hallucination_check")

    # survey_number is grounded in OCR -> VALID
    assert doc.survey_number.validation_status == ValidationStatus.VALID

    # owner_name is hallucinated -> INVALID with hallucination diagnostic!
    assert doc.owner_name.validation_status == ValidationStatus.INVALID
    assert "hallucination" in doc.owner_name.validation_reason.lower() or "not found in ocr" in doc.owner_name.validation_reason.lower()
    assert doc.owner_name.confidence <= 0.25


# ============================================================================
# Test 15: Deterministic normalization
# ============================================================================
def test_deterministic_normalization():
    """Verifies that Kannada numerals are normalized to ASCII while raw text is strictly preserved."""
    normalizer = SemanticNormalizer(convert_digits=True)
    res = normalizer.normalize("ಸರ್ವೆ ನಂ: ೧೨೫/೩")

    assert res.normalized_text == "ಸರ್ವೆ ನಂ: 125/3"
    assert res.raw_text == "ಸರ್ವೆ ನಂ: ೧೨೫/೩"
    assert res.has_changes is True
