import io
from app.pipeline.processor import MockDocumentProcessor, ProcessingResult


def test_mock_document_processor_invoice():
    """Test MockDocumentProcessor extraction logic on invoice files."""
    processor = MockDocumentProcessor()
    sample_stream = io.BytesIO(b"%PDF-1.4 sample commercial invoice content")

    result = processor.process(
        file_stream=sample_stream,
        filename="invoice_2026_01.pdf",
        content_type="application/pdf",
    )

    assert isinstance(result, ProcessingResult)
    assert result.confidence_score >= 0.90
    assert result.is_valid is True
    assert result.processing_time_ms > 0
    assert result.extracted_data["document_type"] == "Commercial Invoice"
    assert result.extracted_data["financials"]["total_amount"] == 1350.00
    assert len(result.extracted_data["line_items"]) == 2
    assert "financial_sum_validation" in str(result.validation_info["checks_passed"])


def test_mock_document_processor_receipt():
    """Test MockDocumentProcessor extraction logic on receipt files."""
    processor = MockDocumentProcessor()
    sample_stream = io.BytesIO(b"\x89PNG sample receipt bytes")

    result = processor.process(
        file_stream=sample_stream,
        filename="coffee_receipt.png",
        content_type="image/png",
    )

    assert isinstance(result, ProcessingResult)
    assert result.extracted_data["document_type"] == "Receipt"
    assert result.extracted_data["financials"]["total_amount"] == 46.11
    assert result.confidence_score >= 0.90
