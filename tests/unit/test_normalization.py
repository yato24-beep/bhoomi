"""
Unit tests for field normalization (names, dates, land units, Fasli year).
"""

import pytest
from src.utils.config_loader import ConfigLoader
from src.validation.normalization import FieldNormalizer


@pytest.fixture
def normalizer():
    loader = ConfigLoader()
    up_cfg = loader.get_state_config("UP")
    return FieldNormalizer(up_cfg)


def test_name_normalization(normalizer):
    assert normalizer.normalize_name("श्री राम प्रसाद") == "राम प्रसाद"
    assert normalizer.normalize_name("श्रीमती कौशल्या देवी") == "कौशल्या देवी"
    assert normalizer.normalize_name("Late Rajesh Kumar /") == "Rajesh Kumar"
    assert normalizer.normalize_name("  Shri   Anil   Sharma  ") == "Anil Sharma"


def test_khasra_normalization(normalizer):
    assert normalizer.normalize_khasra("१४२ / १") == "142/1"
    assert normalizer.normalize_khasra("  142-2  ") == "142/2"
    assert normalizer.normalize_khasra("No. 89/1/A") == "89/1"


def test_land_area_unit_conversions(normalizer):
    # Standard Hectare
    val_ha, unit, norm_unit = normalizer.normalize_land_area("0.4500", "हेक्टेयर")
    assert val_ha == 0.4500
    assert norm_unit == "hectare"

    # Bigha to Hectare conversion (1 Bigha = 0.2529 Ha)
    val_bigha, _, _ = normalizer.normalize_land_area("2", "बीघा")
    assert val_bigha == pytest.approx(2 * 0.2529, rel=1e-3)

    # Acre to Hectare conversion (1 Acre = 0.404686 Ha)
    val_acre, _, _ = normalizer.normalize_land_area("1.5", "acre")
    assert val_acre == pytest.approx(1.5 * 0.404686, rel=1e-3)

    # Devanagari digits in area
    val_dev, _, _ = normalizer.normalize_land_area("०.८५००")
    assert val_dev == 0.8500


def test_date_normalization(normalizer):
    assert normalizer.normalize_date("15/08/2023") == "2023-08-15"
    assert normalizer.normalize_date("26-01-2024") == "2024-01-26"


def test_fasli_year_normalization(normalizer):
    str_fasli, start_greg = normalizer.normalize_fasli_year("1428-1433")
    assert str_fasli == "1428-1433"
    assert start_greg == 2020  # 1428 + 592 = 2020
