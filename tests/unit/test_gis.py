"""
Unit tests for Cadastral GIS spatial validation and 4-state status classification:
1. MATCH
2. MISMATCH
3. UNKNOWN
4. INSUFFICIENT_DATA
Also tests Point-in-Polygon coordinate validation.
"""

import pytest
from schemas import ExtractedField, GISStatus
from src.database.gis import GISValidator


@pytest.fixture
def gis_engine():
    return GISValidator(area_mismatch_threshold_percent=10.0)


def test_gis_karnataka_match(gis_engine):
    # Karnataka Bhoomi RTC Survey 42/1 in Kengeri
    fields = {
        "district": ExtractedField(field_name="district", raw_value="BENGALURU URBAN", normalized_value="BENGALURU URBAN", confidence=0.98),
        "tehsil": ExtractedField(field_name="tehsil", raw_value="BANGALORE SOUTH", normalized_value="BANGALORE SOUTH", confidence=0.98),
        "village": ExtractedField(field_name="village", raw_value="KENGERI", normalized_value="KENGERI", confidence=0.98),
        "khasra_number": ExtractedField(field_name="khasra_number", raw_value="42/1", normalized_value="42/1", confidence=0.98),
        "land_area": ExtractedField(field_name="land_area", raw_value="0.8094", normalized_value=0.8094, confidence=0.98),
    }

    res = gis_engine.validate_gis(fields, state_code="KA", coordinates=(12.9115, 77.4865))
    assert res.gis_status == GISStatus.MATCH
    assert res.is_verified is True
    assert res.has_mismatch is False
    assert res.parcel_id_found is True
    assert res.point_in_polygon_passed is True
    assert res.gis_recorded_area_hectares == 0.8094


def test_gis_tamilnadu_match(gis_engine):
    # Tamil Nadu Patta Survey 108/1 in Nemili
    fields = {
        "district": ExtractedField(field_name="district", raw_value="KANCHIPURAM", normalized_value="KANCHIPURAM", confidence=0.98),
        "tehsil": ExtractedField(field_name="tehsil", raw_value="SRIPERUMBUDUR", normalized_value="SRIPERUMBUDUR", confidence=0.98),
        "village": ExtractedField(field_name="village", raw_value="NEMILI", normalized_value="NEMILI", confidence=0.98),
        "khasra_number": ExtractedField(field_name="khasra_number", raw_value="108/1", normalized_value="108/1", confidence=0.98),
        "land_area": ExtractedField(field_name="land_area", raw_value="0.5000", normalized_value=0.5000, confidence=0.98),
    }

    res = gis_engine.validate_gis(fields, state_code="TN", coordinates=(12.9815, 79.9415))
    assert res.gis_status == GISStatus.MATCH
    assert res.is_verified is True
    assert res.has_mismatch is False
    assert res.point_in_polygon_passed is True


def test_gis_area_mismatch_with_visible_reason(gis_engine):
    # Registered area for 142/1 is 0.45 ha. Document claims 2.85 ha (> 10% tolerance)
    fields = {
        "district": ExtractedField(field_name="district", raw_value="LUCKNOW", normalized_value="LUCKNOW", confidence=0.95),
        "tehsil": ExtractedField(field_name="tehsil", raw_value="MOHANLALGANJ", normalized_value="MOHANLALGANJ", confidence=0.95),
        "village": ExtractedField(field_name="village", raw_value="MAU", normalized_value="MAU", confidence=0.95),
        "khasra_number": ExtractedField(field_name="khasra_number", raw_value="142/1", normalized_value="142/1", confidence=0.95),
        "land_area": ExtractedField(field_name="land_area", raw_value="2.8500", normalized_value=2.8500, confidence=0.95),
    }

    res = gis_engine.validate_gis(fields, state_code="UP")
    assert res.gis_status == GISStatus.MISMATCH
    assert res.has_mismatch is True
    assert res.is_verified is False
    assert res.parcel_id_found is True
    assert any("GIS Area Discrepancy" in flag for flag in res.flag_reasons)


def test_gis_point_in_polygon_mismatch(gis_engine):
    # Valid UP parcel 142/1, but GPS coordinates placed far outside Lucknow (e.g. New Delhi 28.6139, 77.2090)
    fields = {
        "district": ExtractedField(field_name="district", raw_value="LUCKNOW", normalized_value="LUCKNOW", confidence=0.95),
        "tehsil": ExtractedField(field_name="tehsil", raw_value="MOHANLALGANJ", normalized_value="MOHANLALGANJ", confidence=0.95),
        "village": ExtractedField(field_name="village", raw_value="MAU", normalized_value="MAU", confidence=0.95),
        "khasra_number": ExtractedField(field_name="khasra_number", raw_value="142/1", normalized_value="142/1", confidence=0.95),
        "land_area": ExtractedField(field_name="land_area", raw_value="0.4500", normalized_value=0.4500, confidence=0.95),
    }

    res = gis_engine.validate_gis(fields, state_code="UP", coordinates=(28.6139, 77.2090))
    assert res.gis_status == GISStatus.MISMATCH
    assert res.has_mismatch is True
    assert res.point_in_polygon_passed is False
    assert any("Spatial Mismatch: Coordinates" in flag for flag in res.flag_reasons)


def test_gis_missing_reference_is_unknown(gis_engine):
    # Non-existent parcel 999999 -> Must be classified as UNKNOWN (NOT mismatch!)
    fields = {
        "district": ExtractedField(field_name="district", raw_value="LUCKNOW", normalized_value="LUCKNOW", confidence=0.95),
        "tehsil": ExtractedField(field_name="tehsil", raw_value="MOHANLALGANJ", normalized_value="MOHANLALGANJ", confidence=0.95),
        "village": ExtractedField(field_name="village", raw_value="MAU", normalized_value="MAU", confidence=0.95),
        "khasra_number": ExtractedField(field_name="khasra_number", raw_value="999999", normalized_value="999999", confidence=0.95),
        "land_area": ExtractedField(field_name="land_area", raw_value="0.5000", normalized_value=0.5000, confidence=0.95),
    }

    res = gis_engine.validate_gis(fields, state_code="UP")
    assert res.gis_status == GISStatus.UNKNOWN
    assert res.has_mismatch is False  # Explicit requirement: do NOT classify missing reference as mismatch
    assert res.is_verified is False
    assert res.parcel_id_found is False


def test_gis_insufficient_data(gis_engine):
    # Missing Khasra number
    fields = {
        "district": ExtractedField(field_name="district", raw_value="LUCKNOW", normalized_value="LUCKNOW", confidence=0.95),
        "village": ExtractedField(field_name="village", raw_value="MAU", normalized_value="MAU", confidence=0.95),
    }

    res = gis_engine.validate_gis(fields, state_code="UP")
    assert res.gis_status == GISStatus.INSUFFICIENT_DATA
    assert res.is_verified is False
    assert res.has_mismatch is False
