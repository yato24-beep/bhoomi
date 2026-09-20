"""Unit Tests for Semantic Improvements (Gating, Aliasing, Numerals, Kannada Preservations).

Tests explicitly:
1. Historical narrative classified as non-cadastral (pre-semantic gate suppression)
2. BBMP municipal certificate (unlabelled narrative extraction)
3. Karnataka RTC (canonical Bhoomi schema)
4. Unlabelled survey/owner/extent line (contextual disambiguation)
5. Conflicting candidates (multi-owner/survey conflict tracking)
6. Kannada numerals (verbatim preservation in raw, normalized in value)
7. Cross-state alias handling (Maharashtra/Tamil Nadu terms mapped to Karnataka schema)
"""

import unittest
from typing import List

from schemas import BoundingBox, DocumentType
from src.integration.schemas import RecognizedRegionResult
from src.semantic.engine import (
    BaseSemanticEngine,
    RuleSemanticEngine,
    SemanticEngineResult,
    SemanticEvidence,
    SemanticExtractedField,
)
from src.semantic.normalizer import SemanticNormalizer, normalize_kannada_numerals
from src.semantic.regional_aliases import REGIONAL_ALIASES, resolve_field_alias
from src.semantic.schema import ValidationStatus
from src.semantic.semantic_pipeline import SemanticPipeline


def _make_region(
    region_id: str,
    text: str,
    y_min: float = 10.0,
    x_min: float = 10.0,
    confidence: float = 0.95,
    language: str = "kn",
) -> RecognizedRegionResult:
    return RecognizedRegionResult(
        region_id=region_id,
        raw_text=text,
        normalized_text=text,
        bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_min + 200.0, y_max=y_min + 25.0),
        language=language,
        script="Kannada" if language == "kn" else "Latin",
        confidence=confidence,
    )


class MockNonCadastralEngine(BaseSemanticEngine):
    """Mock engine simulating attempt to extract from text."""
    def extract(self, evidence: SemanticEvidence) -> SemanticEngineResult:
        # If gate failed and engine was invoked, it would return erroneous fields
        return SemanticEngineResult(
            fields={
                "taluk": SemanticExtractedField(field_name="taluk", raw_value="ದೊಡ್ಡಬಳ್ಳಾಪುರ", source_region_id="reg_1"),
                "record_date": SemanticExtractedField(field_name="record_date", raw_value="1598", source_region_id="reg_2"),
            },
            document_type="Historical Story",
        )


class MockCadastralMockEngine(BaseSemanticEngine):
    """Mock engine returning specific semantic fields for deterministic test assertions."""
    def __init__(self, fields: dict, doc_type: str = "Bhoomi RTC"):
        self._fields = fields
        self._doc_type = doc_type

    def extract(self, evidence: SemanticEvidence) -> SemanticEngineResult:
        extracted = {}
        for fname, val in self._fields.items():
            if isinstance(val, tuple):
                raw, norm, reg_id, conflicts = val
            else:
                raw, norm, reg_id, conflicts = val, val, "reg_1", []
            extracted[fname] = SemanticExtractedField(
                field_name=fname,
                raw_value=raw,
                normalized_value=norm,
                source_region_id=reg_id,
                model_confidence=0.95,
                conflict_candidates=conflicts,
            )
        return SemanticEngineResult(fields=extracted, document_type=self._doc_type)


class TestSemanticImprovements(unittest.TestCase):

    def test_historical_narrative_classified_as_non_cadastral(self):
        """Task 6.1: Historical narrative classified as non-cadastral suppresses cadastral fields."""
        regions = [
            _make_region("reg_1", "ದೊಡ್ಡಬಳ್ಳಾಪುರ ಸ್ಥಳೀಯ ಆದಿನಾರಾಯಣ ದೇವಸ್ಥಾನದಿಂದ"),
            _make_region("reg_2", "ಕ್ರಿ.ಶ 1598 ರ ದಾಖಲೆಯಲ್ಲಿ ಈ ಸ್ಥಳವನ್ನು ಬಲ್ಲಲಾಪುರ ಎ೦ದು ಉಲ್ಲೇಖಿಸಲಾಗಿದೆ."),
        ]
        pipeline = SemanticPipeline(semantic_engine=MockNonCadastralEngine())

        # Test with explicit is_cadastral=False
        doc = pipeline.process(
            regions=regions,
            document_id="doddaballapura_narrative",
            is_cadastral=False,
            document_type="Historical Narrative",
        )

        self.assertEqual(doc.document_type, "Not a land record")
        self.assertEqual(len(doc.fields), 0)
        self.assertFalse(doc.requires_human_review)
        self.assertEqual(doc.validation_summary.get("gate_status"), "cadastral_extraction_suppressed")
        self.assertFalse(doc.validation_summary.get("is_cadastral"))

        # Test with classification_result dictionary
        doc2 = pipeline.process(
            regions=regions,
            document_id="doddaballapura_narrative_dict",
            classification_result={"is_land_record": False, "document_type": DocumentType.NOT_LAND_RECORD},
        )
        self.assertEqual(doc2.document_type, "Not a land record")
        self.assertEqual(len(doc2.fields), 0)
        self.assertEqual(doc2.validation_summary.get("gate_status"), "cadastral_extraction_suppressed")

    def test_bbmp_municipal_certificate(self):
        """Task 6.2: BBMP municipal certificate with unlabelled narrative extraction."""
        regions = [
            _make_region("line_1", "ಬೃಹತ್ ಬೆಂಗಳೂರು ಮಹಾನಗರ ಪಾಲಿಕೆ"),
            _make_region("line_2", "ಕೋರಮಂಗಲ ಉಪ ವಿಭಾಗ"),
            _make_region("line_3", "Mrs. Dorothy Charles ರವರ ಹೆಸರಿನಲ್ಲಿ ದಾಖಲಾಗಿರುತ್ತದೆ."),
            _make_region("line_4", "ನಿವೇಶನದ ವಿಸ್ತೀರ್ಣ: 2.20 ಎಕರೆ"),
        ]

        mock_fields = {
            "owner_name": ("Mrs. Dorothy Charles", "Mrs. Dorothy Charles", "line_3", []),
            "taluk": ("ಕೋರಮಂಗಲ", "Koramangala", "line_2", []),
            "district": ("ಬೆಂಗಳೂರು", "Bengaluru", "line_1", []),
            "extent": ("2.20 ಎಕರೆ", "2.20 acres", "line_4", []),
        }
        pipeline = SemanticPipeline(semantic_engine=MockCadastralMockEngine(mock_fields, doc_type="BBMP Certificate"))

        doc = pipeline.process(
            regions=regions,
            document_id="doc_bbmp",
            is_cadastral=True,
            document_type="BBMP Certificate",
        )

        self.assertIn("owner_name", doc.fields)
        self.assertEqual(doc.fields["owner_name"].value, "Mrs. Dorothy Charles")
        self.assertEqual(doc.fields["owner_name"].provenance.region_id, "line_3")
        self.assertEqual(doc.fields["taluk"].value, "Koramangala")
        self.assertEqual(doc.fields["district"].value, "Bengaluru")
        self.assertEqual(doc.fields["extent"].value, "2.20 acres")

    def test_karnataka_rtc(self):
        """Task 6.3: Canonical Karnataka Bhoomi RTC extraction."""
        regions = [
            _make_region("line_1", "ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಕಂದಾಯ ಇಲಾಖೆ"),
            _make_region("line_2", "ಭೂಮಿ - ಅಧಿಕಾರ ಮತ್ತು ಸ್ವಾಧೀನತೆ ದಾಖಲೆ RTC"),
            _make_region("line_3", "ಜಿಲ್ಲೆ: BENGALURU URBAN ತಾಲೂಕು: BANGALORE SOUTH ಗ್ರಾಮ: KENGERI"),
            _make_region("line_4", "ಖಾತೆದಾರರ ಹೆಸರು: ಸಿದ್ದರಾಮಯ್ಯ ತಂದೆ: ಬಸವರಾಜ"),
            _make_region("line_5", "ಸರ್ವೆ ನಂ: 42/1 ಹಿಸ್ಸಾ: 1 ವಿಸ್ತೀರ್ಣ: 2-00 ಎಕರೆ"),
        ]

        mock_fields = {
            "district": ("BENGALURU URBAN", "BENGALURU URBAN", "line_3", []),
            "taluk": ("BANGALORE SOUTH", "BANGALORE SOUTH", "line_3", []),
            "village": ("KENGERI", "KENGERI", "line_3", []),
            "survey_number": ("42/1", "42/1", "line_5", []),
            "hissa_number": ("1", "1", "line_5", []),
            "owner_name": ("ಸಿದ್ದರಾಮಯ್ಯ", "ಸಿದ್ದರಾಮಯ್ಯ", "line_4", []),
            "father_name": ("ಬಸವರಾಜ", "ಬಸವರಾಜ", "line_4", []),
            "extent": ("2-00 ಎಕರೆ", "2.00 acres", "line_5", []),
        }
        pipeline = SemanticPipeline(semantic_engine=MockCadastralMockEngine(mock_fields, doc_type="Bhoomi RTC"))

        doc = pipeline.process(
            regions=regions,
            document_id="bhoomi_kengeri",
            is_cadastral=True,
            document_type="Bhoomi RTC",
        )

        self.assertEqual(doc.fields["survey_number"].value, "42/1")
        self.assertEqual(doc.fields["owner_name"].value, "ಸಿದ್ದರಾಮಯ್ಯ")
        self.assertEqual(doc.fields["owner_name"].raw_value, "ಸಿದ್ದರಾಮಯ್ಯ")
        self.assertEqual(doc.fields["village"].value, "KENGERI")
        self.assertEqual(doc.fields["survey_number"].validation_status, ValidationStatus.VALID)

    def test_unlabelled_survey_owner_extent_line(self):
        """Task 6.4: Unlabelled survey, owner, extent line contextual inference."""
        # Realistic raw line: "ರಾಮಪ್ಪ 493/2 2-15 ದೊಡ್ಡಬಳ್ಳಾಪುರ" (no labels)
        raw_text = "ರಾಮಪ್ಪ 493/2 2-15 ದೊಡ್ಡಬಳ್ಳಾಪುರ"
        regions = [_make_region("line_1", raw_text)]

        mock_fields = {
            "owner_name": ("ರಾಮಪ್ಪ", "ರಾಮಪ್ಪ", "line_1", []),
            "survey_number": ("493/2", "493/2", "line_1", []),
            "extent": ("2-15", "2-15", "line_1", []),
            "taluk": ("ದೊಡ್ಡಬಳ್ಳಾಪುರ", "Doddaballapura", "line_1", []),
        }
        pipeline = SemanticPipeline(semantic_engine=MockCadastralMockEngine(mock_fields))

        doc = pipeline.process(
            regions=regions,
            document_id="unlabelled_line_doc",
            is_cadastral=True,
        )

        self.assertEqual(doc.fields["owner_name"].value, "ರಾಮಪ್ಪ")
        self.assertEqual(doc.fields["survey_number"].value, "493/2")
        self.assertEqual(doc.fields["extent"].value, "2-15")
        self.assertEqual(doc.fields["taluk"].value, "Doddaballapura")
        for f in doc.fields.values():
            self.assertEqual(f.provenance.region_id, "line_1")
            self.assertEqual(f.validation_status, ValidationStatus.VALID)

    def test_conflicting_candidates(self):
        """Task 6.5: Disambiguating and tracking conflicting candidate values."""
        regions = [
            _make_region("line_1", "Survey No. 124/2A and Survey No. 124/25"),
            _make_region("line_2", "Owners: Ramesh Kisan Patil and Suresh Kisan Patil"),
        ]

        mock_fields = {
            "survey_number": ("124/2A", "124/2A", "line_1", ["124/25"]),
            "owner_name": ("Ramesh Kisan Patil", "Ramesh Kisan Patil", "line_2", ["Suresh Kisan Patil"]),
        }
        pipeline = SemanticPipeline(semantic_engine=MockCadastralMockEngine(mock_fields))

        doc = pipeline.process(
            regions=regions,
            document_id="conflicts_doc",
            is_cadastral=True,
        )

        self.assertEqual(doc.fields["survey_number"].value, "124/2A")
        self.assertIn("124/25", doc.fields["survey_number"].conflicts)
        self.assertIn("Suresh Kisan Patil", doc.fields["owner_name"].conflicts)

    def test_kannada_numerals(self):
        """Task 6.6: Kannada numeral normalization and verbatim preservation."""
        normalizer = SemanticNormalizer(convert_digits=True)

        # 1. Numeral conversion helper test
        kannada_num_str = "ಸರ್ವೆ ೪೯೩/೨ ವಿಸ್ತೀರ್ಣ ೨-೧೫ ಖಾತೆ ೧೪೯"
        normalized = normalize_kannada_numerals(kannada_num_str)
        self.assertEqual(normalized, "ಸರ್ವೆ 493/2 ವಿಸ್ತೀರ್ಣ 2-15 ಖಾತೆ 149")

        # 2. Pipeline non-destructive handling
        regions = [_make_region("line_1", "ಸರ್ವೆ ನಂ: ೪೯೩/೨ ವಿಸ್ತೀರ್ಣ: ೨ ಎಕರೆ")]
        mock_fields = {
            "survey_number": ("೪೯೩/೨", "493/2", "line_1", []),
            "extent": ("೨ ಎಕರೆ", "2 acres", "line_1", []),
        }
        pipeline = SemanticPipeline(semantic_engine=MockCadastralMockEngine(mock_fields))

        doc = pipeline.process(
            regions=regions,
            document_id="kannada_num_doc",
            is_cadastral=True,
        )

        # Raw value MUST preserve original Kannada digits verbatim
        self.assertEqual(doc.fields["survey_number"].raw_value, "೪೯೩/೨")
        # Standardized value should have ASCII numerals
        self.assertEqual(doc.fields["survey_number"].value, "493/2")
        # Validation and grounding must pass without mismatch
        self.assertEqual(doc.fields["survey_number"].validation_status, ValidationStatus.VALID)

    def test_cross_state_alias_handling(self):
        """Task 6.7: Mapping Maharashtra and Tamil Nadu terms to Karnataka primary schema."""
        # 1. Test alias resolver
        self.assertEqual(resolve_field_alias("gat_number"), ("survey_number", "Maharashtra"))
        self.assertEqual(resolve_field_alias("khatedar"), ("owner_name", "Maharashtra"))
        self.assertEqual(resolve_field_alias("patta_number"), ("khata_number", "Tamil Nadu"))
        self.assertEqual(resolve_field_alias("pula_en"), ("survey_number", "Tamil Nadu"))
        self.assertEqual(resolve_field_alias("pattadharar"), ("owner_name", "Tamil Nadu"))
        self.assertEqual(resolve_field_alias("khasra_number"), ("survey_number", "North India"))
        self.assertEqual(resolve_field_alias("survey_number"), ("survey_number", None))

        # 2. Test pipeline integration with Maharashtra 7/12 aliases
        regions = [
            _make_region("line_1", "Gat Number: 45"),
            _make_region("line_2", "Khatedar: Suresh Patil"),
            _make_region("line_3", "Kshetra: 1.15 Hec"),
        ]

        mock_fields = {
            "gat_number": ("45", "45", "line_1", []),
            "khatedar": ("Suresh Patil", "Suresh Patil", "line_2", []),
            "kshetra": ("1.15 Hec", "1.15 Hec", "line_3", []),
        }
        pipeline = SemanticPipeline(semantic_engine=MockCadastralMockEngine(mock_fields))

        doc = pipeline.process(
            regions=regions,
            document_id="alias_doc",
            is_cadastral=True,
        )

        # Must map to Karnataka canonical schema
        self.assertIn("survey_number", doc.fields)
        self.assertIn("owner_name", doc.fields)
        self.assertIn("extent", doc.fields)
        self.assertEqual(doc.fields["survey_number"].value, "45")
        self.assertEqual(doc.fields["owner_name"].value, "Suresh Patil")
        self.assertEqual(doc.fields["extent"].value, "1.15 Hec")


if __name__ == "__main__":
    unittest.main()
