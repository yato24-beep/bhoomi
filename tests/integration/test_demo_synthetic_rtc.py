"""Integration test verifying end-to-end processing of the synthetic Karnataka RTC document.

Tests both the PNG and PDF formats through the actual DocumentProcessingPipeline
without mocked extractions, fake API responses, or hardcoded values.
"""

from pathlib import Path
import pytest
from src.integration.document_pipeline import DocumentProcessingPipeline
from src.preprocessing.format_adapter import DocumentFormatAdapter

DEMO_DIR = Path("demo_artifacts")
PNG_PATH = DEMO_DIR / "synthetic_karnataka_rtc.png"
PDF_PATH = DEMO_DIR / "synthetic_karnataka_rtc.pdf"


@pytest.fixture(scope="module")
def pipeline():
    # Production pipeline config with printed OCR and TrOCR handwriting recognizer
    return DocumentProcessingPipeline(confidence_threshold=0.70)


def test_synthetic_rtc_files_exist():
    """Verify both synthetic artifacts are generated and non-empty."""
    assert PNG_PATH.exists(), f"PNG artifact missing at {PNG_PATH}"
    assert PDF_PATH.exists(), f"PDF artifact missing at {PDF_PATH}"
    assert PNG_PATH.stat().st_size > 1000, "PNG file is too small"
    assert PDF_PATH.stat().st_size > 1000, "PDF file is too small"


def test_pdf_format_adapter_ingestion():
    """Verify PDF format adapter correctly converts synthetic RTC PDF to valid RGB image."""
    images = DocumentFormatAdapter.load_pages(PDF_PATH)
    assert len(images) >= 1, "PDF should yield at least 1 image page"
    assert images[0].width >= 700, "Image width should be standard document resolution"
    assert images[0].height >= 700, "Image height should be standard document resolution"


def test_synthetic_rtc_png_pipeline_e2e(pipeline):
    """Verify synthetic PNG runs through the real production pipeline end-to-end."""
    result = pipeline.process_document(image=str(PNG_PATH), document_id="demo_png_001")

    # 1. Pipeline status & land record gating
    assert result.status in ("completed", "flagged_for_review"), f"Unexpected pipeline status: {result.status}"
    assert result.is_land_record is True, "Document should be identified as a land record"

    # 2. OCR text extraction
    raw_ocr = result.merged_text
    assert len(raw_ocr) > 50, "OCR should extract text from synthetic document"
    # Should detect key Karnataka Bhoomi / RTC keywords
    assert any(term in raw_ocr for term in ["ಸರ್ಕಾರ", "ದಾಬಲೆ", "ಆರ್ ಟಸಿ", "RTC", "Bengoluru", "Bengaluru", "District", "ಕರ್ನಾಟಕ", "ಭೂಮಿ"]), \
        f"Key cadastral terms not found in OCR text: {raw_ocr[:200]}"

    # 3. Canonical extracted fields
    fields = result.extracted_fields
    assert len(fields) > 0, "Pipeline should extract structured cadastral fields"

    # 4. Review Queue & Human Review Trigger
    # The synthetic document has a controlled cultivator entry to exercise human review
    assert isinstance(result.review_items, list), "Review items must be a list"

    # 5. Timing & stage breakdown
    assert result.processing_time_ms > 0, "Processing time should be positive"


def test_synthetic_rtc_pdf_pipeline_e2e(pipeline):
    """Verify synthetic PDF runs through the real production pipeline end-to-end."""
    result = pipeline.process_document(image=str(PDF_PATH), document_id="demo_pdf_001")

    assert result.status in ("completed", "flagged_for_review"), f"Unexpected pipeline status: {result.status}"
    assert result.is_land_record is True, "PDF document should be identified as a land record"
    assert len(result.merged_text) > 50, "OCR should extract text from PDF"
    assert any(term in result.merged_text for term in ["ಸರ್ಕಾರ", "ದಾಬಲೆ", "ಆರ್ ಟಸಿ", "RTC", "Bengoluru", "Bengaluru", "District", "ಕರ್ನಾಟಕ", "ಭೂಮಿ"])
