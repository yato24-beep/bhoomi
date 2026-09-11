"""
Unit tests for single-field rule validation.
"""

import pytest
from schemas import BoundingBox, ExtractedField, ExtractionMethod, ValidationStatus
from src.utils.config_loader import ConfigLoader
from src.validation.rules import RuleValidator


@pytest.fixture
def validator():
    loader = ConfigLoader()
    up_cfg = loader.get_state_config("UP")
    return RuleValidator(up_cfg)


def test_valid_fields_pass(validator):
    fields = {
        "khasra_number": ExtractedField(
            field_name="khasra_number",
            raw_value="142/1",
            normalized_value="142/1",
            confidence=0.95,
        ),
        "owner_name": ExtractedField(
            field_name="owner_name",
            raw_value="राम प्रसाद",
            normalized_value="राम प्रसाद",
            confidence=0.95,
        ),
        "land_area": ExtractedField(
            field_name="land_area",
            raw_value="0.4500",
            normalized_value=0.4500,
            confidence=0.95,
        ),
        "village": ExtractedField(
            field_name="village",
            raw_value="मऊ",
            normalized_value="मऊ",
            confidence=0.95,
        ),
        "tehsil": ExtractedField(
            field_name="tehsil",
            raw_value="मोहनलालगंज",
            normalized_value="मोहनलालगंज",
            confidence=0.95,
        ),
        "district": ExtractedField(
            field_name="district",
            raw_value="लखनऊ",
            normalized_value="लखनऊ",
            confidence=0.95,
        ),
        "khatauni_number": ExtractedField(
            field_name="khatauni_number",
            raw_value="124",
            normalized_value="124",
            confidence=0.95,
        ),
    }

    items, status = validator.validate_fields(fields)
    assert status == ValidationStatus.VALID
    assert len([i for i in items if not i.passed]) == 0


def test_invalid_area_range_fails(validator):
    fields = {
        "khasra_number": ExtractedField(field_name="khasra_number", raw_value="142", normalized_value="142", confidence=0.9),
        "owner_name": ExtractedField(field_name="owner_name", raw_value="राम", normalized_value="राम", confidence=0.9),
        "land_area": ExtractedField(field_name="land_area", raw_value="-5.0", normalized_value=-5.0, confidence=0.9),
    }

    items, status = validator.validate_fields(fields)
    assert status == ValidationStatus.INVALID
    errors = [i for i in items if not i.passed]
    assert any("below minimum" in e.message for e in errors)


def test_missing_mandatory_field_fails(validator):
    # Missing owner_name and village
    fields = {
        "khasra_number": ExtractedField(field_name="khasra_number", raw_value="142", normalized_value="142", confidence=0.9),
        "land_area": ExtractedField(field_name="land_area", raw_value="0.5", normalized_value=0.5, confidence=0.9),
    }

    items, status = validator.validate_fields(fields)
    assert status == ValidationStatus.INVALID
    missing_items = [i for i in items if "is missing" in i.message]
    assert len(missing_items) > 0
