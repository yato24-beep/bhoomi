"""
Unit tests for structured field extraction across multiple state formats and Devanagari numerals.
"""

import pytest
from schemas import (
    BoundingBox,
    DocumentClassificationResult,
    DocumentOCRResult,
    DocumentType,
    OCREngineType,
    OCRTextLine,
    TableStructure,
)
from src.extraction.extractor import (
    FieldExtractor,
    convert_devanagari_numerals,
)
from src.utils.config_loader import ConfigLoader


def test_devanagari_numeral_conversion():
    assert convert_devanagari_numerals("१४२/१") == "142/1"
    assert convert_devanagari_numerals("०.४५००") == "0.4500"
    assert convert_devanagari_numerals("वर्ष १४२८-१४३३") == "वर्ष 1428-1433"


def test_up_khatauni_extraction():
    loader = ConfigLoader()
    state_cfg = loader.get_state_config("UP")
    extractor = FieldExtractor(state_cfg)

    lines = [
        OCRTextLine(
            text="ग्राम का नाम: मऊ  तहसील: मोहनलालगंज  जनपद: लखनऊ",
            confidence=0.98,
            bbox=BoundingBox(x_min=10, y_min=10, x_max=500, y_max=30),
            page_number=1,
            engine=OCREngineType.PADDLE_OCR,
        ),
        OCRTextLine(
            text="खाता संख्या: 00124",
            confidence=0.95,
            bbox=BoundingBox(x_min=10, y_min=40, x_max=200, y_max=60),
            page_number=1,
            engine=OCREngineType.PADDLE_OCR,
        ),
        OCRTextLine(
            text="खातेदार का नाम: राम प्रसाद  पिता: श्याम लाल",
            confidence=0.94,
            bbox=BoundingBox(x_min=10, y_min=70, x_max=400, y_max=90),
            page_number=1,
            engine=OCREngineType.PADDLE_OCR,
        ),
        OCRTextLine(
            text="गाटा संख्या: 142/1  क्षेत्रफल (हेक्टेयर): 0.4500",
            confidence=0.97,
            bbox=BoundingBox(x_min=10, y_min=100, x_max=450, y_max=120),
            page_number=1,
            engine=OCREngineType.PADDLE_OCR,
        ),
    ]

    ocr_res = DocumentOCRResult(
        document_id="DOC_TEST_UP",
        sha256_hash="abc123hash",
        text_lines=lines,
        raw_full_text="\n".join(l.text for l in lines),
    )

    fields = extractor.extract_all(ocr_res)

    assert "khasra_number" in fields
    assert fields["khasra_number"].raw_value == "142/1"
    assert fields["khasra_number"].bbox.y_min == 100

    assert "owner_name" in fields
    assert "राम प्रसाद" in fields["owner_name"].raw_value

    assert "village" in fields
    assert "मऊ" in fields["village"].raw_value

    assert "land_area" in fields
    assert "0.4500" in fields["land_area"].raw_value


def test_table_extraction():
    loader = ConfigLoader()
    state_cfg = loader.get_state_config("UP")
    extractor = FieldExtractor(state_cfg)

    table = TableStructure(
        table_id="tbl_1",
        page_number=1,
        bbox=BoundingBox(x_min=50, y_min=200, x_max=500, y_max=400),
        headers=["क्र०", "खसरा संख्या", "क्षेत्रफल (हे०)", "खातेदार"],
        rows=[
            ["1", "205/3", "1.2500", "सुरेश कुमार"],
            ["2", "205/4", "0.7500", "रमेश कुमार"],
        ],
    )

    ocr_res = DocumentOCRResult(
        document_id="DOC_TBL_TEST",
        sha256_hash="tbl123",
        text_lines=[],
        tables=[table],
        raw_full_text="",
    )

    fields = extractor.extract_all(ocr_res)
    assert "khasra_number" in fields
    assert fields["khasra_number"].raw_value == "205/3"
    assert fields["khasra_number"].extraction_method.value == "table_lookup"
