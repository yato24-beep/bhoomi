"""person-a/tests/test_person_a_pipeline.py
End-to-end integration tests for Person A pipeline and downstream adapters.
"""

from pathlib import Path
import numpy as np
import pytest

from src.pipeline import process_document, process_document_with_handwriting
from src.integration.person_c_adapter import convert_person_a_to_document_ocr_result
from src.integration.backend_adapter import process_backend_stream_to_result
from src.schemas import DocumentInput, OCROutput


def test_process_document_synthetic_712():
    sample = Path("data/samples/sample_land_record_7_12.png")
    if not sample.exists():
        pytest.skip(f"Fixture {sample} not found")

    output = process_document(str(sample), state_hint="MH")
    assert isinstance(output, OCROutput)
    assert output.document_id == "sample_land_record_7_12"
    assert len(output.pages) == 1
    assert output.pages[0].quality is not None
    assert output.classification.predicted_type in ("land_record_7_12", "land_record")

    # Verify downstream Person C contract conversion
    c_payload = convert_person_a_to_document_ocr_result(output)
    assert c_payload["document_id"] == output.document_id
    assert "text_lines" in c_payload
    assert "tables" in c_payload
    assert "classification" in c_payload


def test_process_document_blank_image():
    blank = np.full((300, 300, 3), 255, dtype=np.uint8)
    output = process_document(blank)
    assert len(output.pages) == 1
    assert output.pages[0].quality.is_blank is True
    assert output.pages[0].text == ""


def test_process_document_corrupt_bytes():
    corrupt = b"\x00\x01\x02NOT_AN_IMAGE"
    output = process_document(corrupt)
    assert len(output.pages) == 0
    assert output.classification.predicted_type == "unknown"


def test_process_with_handwriting_adapter():
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    ocr_out, hw_out = process_document_with_handwriting(dummy)
    assert isinstance(ocr_out, OCROutput)
    assert hw_out.document_id == ocr_out.document_id
    assert hw_out.model_version is not None


def test_table_multi_column_extraction():
    sample = Path("data/samples/sample_land_record_7_12.png")
    if not sample.exists():
        pytest.skip(f"Fixture {sample} not found")

    output = process_document(str(sample), state_hint="MH")
    assert len(output.pages) == 1
    tables = output.pages[0].tables
    assert len(tables) >= 1
    table = tables[0]
    # Verify multi-column, multi-row grid structure
    assert table.rows_count >= 2
    assert table.cols_count >= 2
    assert len(table.cells) >= 4
    # Verify cell row and column assignment
    for cell in table.cells:
        assert cell.row_index >= 0
        assert cell.col_index >= 0
    # Verify stage timings metadata
    assert "stage_timings_ms" in output.metadata
    assert "total_pipeline_ms" in output.metadata["stage_timings_ms"]
    # Verify transparent language fallback metadata
    assert output.pages[0].language_info is not None
    assert output.pages[0].language_info.requested_language == "mr"
    assert output.pages[0].language_info.actual_language in ("en", "mr")

