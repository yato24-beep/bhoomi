"""
Unit tests for cross-record validation (parcel area sum vs header total, share summation).
"""

import pytest
from schemas import (
    BoundingBox,
    ExtractedField,
    TableStructure,
)
from src.validation.cross_record import CrossRecordValidator


def test_cross_record_area_sum_matches():
    validator = CrossRecordValidator(area_sum_tolerance_percent=5.0)

    fields = {
        "land_area": ExtractedField(
            field_name="land_area",
            raw_value="2.0000",
            normalized_value=2.0000,
            confidence=0.95,
        )
    }

    tables = [
        TableStructure(
            table_id="tbl_parcels",
            page_number=1,
            bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100),
            headers=["Khasra", "Area (Ha)"],
            rows=[
                ["101", "1.2500"],
                ["102", "0.7500"],
            ],
        )
    ]

    res = validator.validate_cross_record(fields, tables, {})
    assert res.passed is True
    assert res.area_sum_matches is True
    assert len(res.inconsistencies) == 0


def test_cross_record_area_sum_mismatch():
    validator = CrossRecordValidator(area_sum_tolerance_percent=5.0)

    fields = {
        "land_area": ExtractedField(
            field_name="land_area",
            raw_value="5.0000",  # Header says 5.0000 ha
            normalized_value=5.0000,
            confidence=0.95,
        )
    }

    tables = [
        TableStructure(
            table_id="tbl_parcels",
            page_number=1,
            bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100),
            headers=["Khasra", "Area (Ha)"],
            rows=[
                ["101", "1.0000"],
                ["102", "1.0000"],  # Sum is 2.0000 ha -> mismatch!
            ],
        )
    ]

    res = validator.validate_cross_record(fields, tables, {})
    assert res.passed is False
    assert res.area_sum_matches is False
    assert any("Area mismatch" in inc for inc in res.inconsistencies)


def test_cross_record_co_owner_shares():
    validator = CrossRecordValidator()

    fields = {}
    tables = [
        TableStructure(
            table_id="tbl_shares",
            page_number=1,
            bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100),
            headers=["Owner", "Share"],
            rows=[
                ["Ram Prasad", "1/2"],
                ["Shyam Lal", "1/2"],
            ],
        )
    ]

    res = validator.validate_cross_record(fields, tables, {})
    assert res.passed is True
    assert res.share_sum_matches is True
