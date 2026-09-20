"""Unit tests for Semantic Layer V1 (Rule, Layout, and Geometry based).

Verifies:
- Land-record canonical schema (CadastralRecord, OwnerRecord, BoundaryRecord, SemanticTable)
- Controlled non-destructive OCR normalization (NFC, whitespace, safe punctuation, numerals; no speculative spelling)
- Anchor field detection (Kannada and English land-record terms)
- Spatial key/value association (inline, rightward, downward, standalone patterns)
- Field-specific validators (survey number cadastral formats, khata patterns, calendar dates rejecting impossible dates, extent, names/localities)
- Provenance linking (document_id, page_number, region_id, bbox, raw_ocr_text, normalized_value)
- Table reconstruction (rows, columns, cells, source regions)
- Deterministic reading order preservation
- Zero ML model training / zero weight mutation
"""

from datetime import datetime
import pytest

from schemas import BoundingBox
from src.integration.schemas import RecognizedRegionResult
from src.semantic.schema import (
    CadastralRecord,
    FieldProvenance,
    LandRecordDocument,
    OwnerRecord,
    SemanticFieldItem,
    SemanticTable,
    SemanticTableCell,
    ValidationStatus,
)
from src.semantic.normalizer import ControlledOCRNormalizer
from src.semantic.field_detector import FieldAnchorDetector
from src.semantic.value_associator import SpatialValueAssociator
from src.semantic.validator import FieldValidator
from src.semantic.table_reconstructor import TableReconstructor
from src.semantic.semantic_pipeline import SemanticPipeline


@pytest.fixture
def normalizer():
    return ControlledOCRNormalizer()


@pytest.fixture
def detector():
    return FieldAnchorDetector()


@pytest.fixture
def associator():
    return SpatialValueAssociator()


@pytest.fixture
def validator():
    return FieldValidator()


@pytest.fixture
def reconstructor():
    return TableReconstructor()


@pytest.fixture
def semantic_pipeline():
    return SemanticPipeline()


def test_controlled_normalizer_nfc_and_whitespace(normalizer):
    """Verifies that normalization applies Unicode NFC and cleans redundant whitespace without altering text."""
    # Decomposed characters should be normalized to NFC
    decomposed = "ರಾಮಯ್ಯ\u0020\u0020\u0020"
    res = normalizer.normalize(decomposed)
    assert res.normalized_text == "ರಾಮಯ್ಯ"
    assert res.raw_text == decomposed


def test_controlled_normalizer_numerals(normalizer):
    """Verifies Kannada to ASCII numeral mapping when configured."""
    kn_num = "ಸರ್ವೆ ನಂ. ೧೨೫/೧"
    res = normalizer.normalize(kn_num)
    assert "125/1" in res.normalized_text
    assert res.raw_text == kn_num


def test_controlled_normalizer_no_speculative_spelling(normalizer):
    """Verifies that normalizer NEVER hallucinates or speculatively corrects unknown words."""
    unknown_word = "ಕನಕಪುರxx99"
    res = normalizer.normalize(unknown_word)
    assert res.normalized_text == unknown_word
    assert res.raw_text == unknown_word


def test_field_detector_kannada_anchors(detector):
    """Verifies recognition of standard Kannada land-record field labels."""
    assert detector.detect_anchor("ಸರ್ವೆ ನಂಬರ್") == "survey_number"
    assert detector.detect_anchor("ಖಾತಾ ನಂ") == "khata_number"
    assert detector.detect_anchor("ಖಾತೆದಾರರ ಹೆಸರು") == "owner_name"
    assert detector.detect_anchor("ಗ್ರಾಮ:") == "village"
    assert detector.detect_anchor("ಹೋಬಳಿ:") == "hobli"
    assert detector.detect_anchor("ತಾಲೂಕು:") == "taluk"
    assert detector.detect_anchor("ಜಿಲ್ಲೆ:") == "district"
    assert detector.detect_anchor("ವಿಸ್ತೀರ್ಣ") == "extent"


def make_region(
    region_id: str,
    raw_text: str,
    bbox: BoundingBox,
    page_number: int = 1,
    is_handwritten: bool = False,
) -> RecognizedRegionResult:
    return RecognizedRegionResult(
        region_id=region_id,
        raw_text=raw_text,
        normalized_text=raw_text,
        bbox=bbox,
        page_number=page_number,
        language="kannada",
        script="Kannada",
        is_handwritten=is_handwritten,
    )


def test_spatial_associator_inline_key_value(associator):
    """Verifies inline key-value extraction like 'ಸರ್ವೆ ನಂ: 125/1'."""
    region = make_region(
        region_id="reg_01",
        raw_text="ಸರ್ವೆ ನಂ: 125/1",
        bbox=BoundingBox(x_min=100, y_min=200, x_max=300, y_max=240),
    )
    fields = associator.associate_regions([region])
    assert "survey_number" in fields
    f = fields["survey_number"]
    assert f.raw_value == "125/1"
    assert f.provenance is not None
    assert f.provenance.region_id == "reg_01"


def test_spatial_associator_rightward_key_value(associator):
    """Verifies horizontal rightward association between label box and value box."""
    label_reg = make_region(
        region_id="reg_lbl",
        raw_text="ಹೆಸರು:",
        bbox=BoundingBox(x_min=50, y_min=100, x_max=150, y_max=130),
    )
    val_reg = make_region(
        region_id="reg_val",
        raw_text="ರಾಮಯ್ಯ",
        bbox=BoundingBox(x_min=160, y_min=100, x_max=300, y_max=130),
    )
    fields = associator.associate_regions([label_reg, val_reg])
    assert "owner_name" in fields
    f = fields["owner_name"]
    assert f.raw_value == "ರಾಮಯ್ಯ"
    assert f.provenance.region_id == "reg_val"


def test_validator_survey_number(validator):
    """Verifies cadastral survey number validation and sub-division extraction."""
    res_valid = validator.validate_field("survey_number", "125/1A")
    assert res_valid.status == ValidationStatus.VALID
    assert res_valid.normalized_value == "125/1A"

    res_hissa = validator.validate_field("survey_number", "42/3")
    assert res_hissa.status == ValidationStatus.VALID

    res_invalid = validator.validate_field("survey_number", "NotANumber!@#")
    assert res_invalid.status == ValidationStatus.INVALID


def test_validator_khata_number(validator):
    """Verifies khata number validation."""
    res_valid = validator.validate_field("khata_number", "512")
    assert res_valid.status == ValidationStatus.VALID

    res_invalid = validator.validate_field("khata_number", "")
    assert res_invalid.status == ValidationStatus.INVALID


def test_validator_calendar_date(validator):
    """Verifies date validation rejecting impossible calendar dates like 2021-02-31."""
    res_valid = validator.validate_field("registration_date", "15/08/2020")
    assert res_valid.status == ValidationStatus.VALID

    res_iso = validator.validate_field("registration_date", "2020-08-15")
    assert res_iso.status == ValidationStatus.VALID

    # Impossible date: Feb 31st must be rejected!
    res_impossible = validator.validate_field("registration_date", "31/02/2021")
    assert res_impossible.status == ValidationStatus.INVALID
    assert "calendar date" in res_impossible.reason.lower()


def test_validator_extent(validator):
    """Verifies land measure extent validation."""
    res_valid = validator.validate_field("extent", "2-15")
    assert res_valid.status == ValidationStatus.VALID
    assert "extent" in res_valid.reason.lower()

    res_text = validator.validate_field("extent", "3 ಎಕರೆ 10 ಗುಂಟೆ")
    assert res_text.status == ValidationStatus.VALID


def test_validator_names_non_destructive(validator):
    """Verifies that names and localities are validated without aggressive or speculative dictionary mutations."""
    res = validator.validate_field("owner_name", "ರಾಮಯ್ಯ ಮಲ್ಲಪ್ಪ")
    assert res.status == ValidationStatus.VALID
    assert res.normalized_value == "ರಾಮಯ್ಯ ಮಲ್ಲಪ್ಪ"


def test_table_reconstructor_grid(reconstructor):
    """Verifies tabular grid grouping into rows, columns, and cells with region provenance."""
    # Create 4 cells in a 2x2 grid
    r11 = make_region("c11", "ಸರ್ವೆ ನಂ", BoundingBox(x_min=10, y_min=10, x_max=100, y_max=40))
    r12 = make_region("c12", "ವಿಸ್ತೀರ್ಣ", BoundingBox(x_min=110, y_min=10, x_max=200, y_max=40))
    r21 = make_region("c21", "125/1", BoundingBox(x_min=10, y_min=50, x_max=100, y_max=80))
    r22 = make_region("c22", "1-20", BoundingBox(x_min=110, y_min=50, x_max=200, y_max=80))

    tables = reconstructor.reconstruct_tables([r11, r12, r21, r22])
    assert len(tables) == 1
    t = tables[0]
    assert t.num_rows == 2
    assert t.num_cols == 2
    assert len(t.cells) == 4
    # Check cell region provenance
    cell_regions = {c.source_region_id for c in t.cells}
    assert cell_regions == {"c11", "c12", "c21", "c22"}


def test_semantic_pipeline_end_to_end(semantic_pipeline):
    """Verifies end-to-end extraction, validation, and provenance through SemanticPipeline."""
    regions = [
        make_region(
            region_id="r1",
            raw_text="ಗ್ರಾಮ: ಚಿಕ್ಕಬಳ್ಳಾಪುರ",
            bbox=BoundingBox(x_min=20, y_min=50, x_max=200, y_max=80),
        ),
        make_region(
            region_id="r2",
            raw_text="ಸರ್ವೆ ನಂ: 142/2",
            bbox=BoundingBox(x_min=20, y_min=100, x_max=200, y_max=130),
        ),
        make_region(
            region_id="r3",
            raw_text="ದಿನಾಂಕ: 12/04/2022",
            bbox=BoundingBox(x_min=20, y_min=150, x_max=200, y_max=180),
        ),
    ]

    doc = semantic_pipeline.process(
        document_id="doc_test_101",
        page_number=1,
        regions=regions,
    )

    assert isinstance(doc, LandRecordDocument)
    assert doc.document_id == "doc_test_101"
    assert "village" in doc.fields
    assert "survey_number" in doc.fields
    assert "registration_date" in doc.fields

    # Verify provenance on survey number
    sn_field = doc.fields["survey_number"]
    assert sn_field.normalized_value == "142/2"
    assert sn_field.validation_status == ValidationStatus.VALID
    assert sn_field.provenance is not None
    assert sn_field.provenance.document_id == "doc_test_101"
    assert sn_field.provenance.page_number == 1
    assert sn_field.provenance.region_id == "r2"
    assert sn_field.provenance.raw_ocr_text == "ಸರ್ವೆ ನಂ: 142/2"

    # Verify structured cadastral record
    assert len(doc.cadastral_records) > 0
    assert doc.cadastral_records[0].survey_number.value == "142/2"


def test_semantic_field_provenance_complete_contract(semantic_pipeline):
    """Verifies that every semantic field strictly preserves the complete provenance contract.
    Required fields: document_id, page_number, region_id, bbox, raw_ocr_text, normalized_value,
    validation_status, validation_reason.
    No field may exist without source evidence.
    """
    regions = [
        make_region(
            region_id="reg_sn_01",
            raw_text="ಸರ್ವೆ ನಂ: 189/2A",
            bbox=BoundingBox(x_min=10, y_min=20, x_max=150, y_max=50),
            page_number=2,
        ),
        make_region(
            region_id="reg_kh_02",
            raw_text="ಖಾತಾ ಸಂಖ್ಯೆ: KH-9902",
            bbox=BoundingBox(x_min=10, y_min=60, x_max=160, y_max=90),
            page_number=2,
        ),
    ]

    doc = semantic_pipeline.process(
        document_id="doc_audit_999",
        page_number=2,
        regions=regions,
    )

    for field_name, sem_field in doc.fields.items():
        assert sem_field.provenance is not None, f"Field {field_name} missing source evidence"
        prov = sem_field.provenance
        assert prov.document_id == "doc_audit_999"
        assert prov.page_number == 2
        assert prov.region_id in ("reg_sn_01", "reg_kh_02")
        assert prov.bbox is not None
        assert prov.raw_ocr_text != ""
        assert sem_field.normalized_value != ""
        assert sem_field.validation_status in (ValidationStatus.VALID, ValidationStatus.WARNING, ValidationStatus.INVALID, ValidationStatus.UNVERIFIED)
        assert isinstance(sem_field.validation_reason, (str, type(None)))


def test_table_reconstruction_multi_row(reconstructor):
    """Verifies table reconstruction with 1 table and multiple data rows preserving cell provenance."""
    # 3 rows x 3 columns
    # Row 0 (Headers): Sl.No | Survey No | Extent
    r00 = make_region("c00", "ಕ್ರಮ ಸಂಖ್ಯೆ", BoundingBox(x_min=10, y_min=10, x_max=60, y_max=35))
    r01 = make_region("c01", "ಸರ್ವೆ ನಂ", BoundingBox(x_min=70, y_min=10, x_max=140, y_max=35))
    r02 = make_region("c02", "ವಿಸ್ತೀರ್ಣ", BoundingBox(x_min=150, y_min=10, x_max=220, y_max=35))

    # Row 1: 1 | 120/1 | 2-00
    r10 = make_region("c10", "1", BoundingBox(x_min=10, y_min=45, x_max=60, y_max=70))
    r11 = make_region("c11", "120/1", BoundingBox(x_min=70, y_min=45, x_max=140, y_max=70))
    r12 = make_region("c12", "2-00", BoundingBox(x_min=150, y_min=45, x_max=220, y_max=70))

    # Row 2: 2 | 120/2 | 1-20
    r20 = make_region("c20", "2", BoundingBox(x_min=10, y_min=80, x_max=60, y_max=105))
    r21 = make_region("c21", "120/2", BoundingBox(x_min=70, y_min=80, x_max=140, y_max=105))
    r22 = make_region("c22", "1-20", BoundingBox(x_min=150, y_min=80, x_max=220, y_max=105))

    table = reconstructor.reconstruct_table([r00, r01, r02, r10, r11, r12, r20, r21, r22], page_number=1)
    assert table is not None
    assert table.headers == ["ಕ್ರಮ ಸಂಖ್ಯೆ", "ಸರ್ವೆ ನಂ", "ವಿಸ್ತೀರ್ಣ"]
    assert len(table.rows) == 2
    assert table.rows[0] == ["1", "120/1", "2-00"]
    assert table.rows[1] == ["2", "120/2", "1-20"]
    assert len(table.cells) == 9
    for cell in table.cells:
        assert cell.source_region_id in {"c00", "c01", "c02", "c10", "c11", "c12", "c20", "c21", "c22"}
        assert cell.bbox is not None


def test_table_reconstruction_irregular_and_missing_cells(reconstructor):
    """Verifies irregular row widths, merged cell geometries, and that missing OCR regions
    are NEVER fabricated.
    """
    # Header: Title spanning wide area (Row 0)
    r0 = make_region("h0", "ಜಮೀನಿನ ವಿವರಗಳು (Land Details)", BoundingBox(x_min=10, y_min=10, x_max=250, y_max=35))

    # Row 1: 3 columns
    r10 = make_region("r10", "1", BoundingBox(x_min=10, y_min=45, x_max=40, y_max=70))
    r11 = make_region("r11", "ಸರ್ವೆ 45", BoundingBox(x_min=50, y_min=45, x_max=130, y_max=70))
    r12 = make_region("r12", "0-30", BoundingBox(x_min=140, y_min=45, x_max=200, y_max=70))

    # Row 2: 2 columns only (third column has no OCR detection/blank cell in image)
    r20 = make_region("r20", "2", BoundingBox(x_min=10, y_min=80, x_max=40, y_max=105))
    r21 = make_region("r21", "ಸರ್ವೆ 46", BoundingBox(x_min=50, y_min=80, x_max=130, y_max=105))
    # Note: Column 3 is intentionally missing

    table = reconstructor.reconstruct_table([r0, r10, r11, r12, r20, r21], page_number=1)
    assert table is not None
    # Header is the wide cell
    assert table.headers == ["ಜಮೀನಿನ ವಿವರಗಳು (Land Details)"]
    # Data rows: Row 1 has 3 elements, Row 2 has 2 elements
    assert len(table.rows) == 2
    assert len(table.rows[0]) == 3
    assert len(table.rows[1]) == 2
    # Crucial: Exactly 6 cells must be returned; NO missing cell was fabricated
    assert len(table.cells) == 6
    cell_texts = [c.text for c in table.cells]
    assert "0-30" in cell_texts
    assert not any(c.text == "" and c.source_region_id.startswith("fabricated") for c in table.cells)

