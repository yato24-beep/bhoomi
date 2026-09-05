import abc
import time
from typing import BinaryIO, Dict, Any, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized spatial coordinates representing field location on document page."""
    x_min: float = Field(..., ge=0.0, le=1.0, description="Left coordinate (0.0 to 1.0)")
    y_min: float = Field(..., ge=0.0, le=1.0, description="Top coordinate (0.0 to 1.0)")
    x_max: float = Field(..., ge=0.0, le=1.0, description="Right coordinate (0.0 to 1.0)")
    y_max: float = Field(..., ge=0.0, le=1.0, description="Bottom coordinate (0.0 to 1.0)")
    unit: str = Field(default="normalized", description="Coordinate unit system")


class ExtractedFieldItem(BaseModel):
    """Schema for an individual extracted field with provenance and coordinates."""
    field_name: str = Field(..., description="Canonical identifier for the field")
    original_value: Optional[str] = Field(default=None, description="Raw text as recognized on document")
    normalized_value: Optional[str] = Field(default=None, description="Standardized, cleaned representation")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    source_page: int = Field(default=1, ge=1, description="Page number where field was located")
    bounding_box: Optional[BoundingBox] = Field(default=None, description="Spatial coordinates on the page")


class ProcessingResult(BaseModel):
    """Standardized output schema produced by any document processing pipeline."""
    
    extracted_data: Dict[str, Any] = Field(
        ...,
        description="Structured key-value pairs, line items, and parsed text extracted from document",
    )
    fields: List[ExtractedFieldItem] = Field(
        default_factory=list,
        description="Granular list of extracted fields with bounding boxes and normalization",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate extraction accuracy confidence score (0.0 to 1.0)",
    )
    is_valid: bool = Field(
        ...,
        description="True if business validation rules passed",
    )
    validation_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed report of validation rules evaluated, warnings, and errors",
    )
    processing_time_ms: int = Field(
        ...,
        description="Total pipeline execution duration in milliseconds",
    )


class BaseDocumentProcessor(abc.ABC):
    """Abstract interface defining the contract for document extraction engines."""

    @abc.abstractmethod
    def process(
        self,
        file_stream: BinaryIO,
        filename: str,
        content_type: Optional[str] = None,
    ) -> ProcessingResult:
        """Execute document parsing and feature extraction."""
        pass


class MockDocumentProcessor(BaseDocumentProcessor):
    """Mock processor simulating OCR, entity extraction, spatial bounding boxes, and validation."""

    def process(
        self,
        file_stream: BinaryIO,
        filename: str,
        content_type: Optional[str] = None,
    ) -> ProcessingResult:
        start_time = time.perf_counter()

        # Read sample stream
        file_stream.seek(0)
        byte_sample = file_stream.read(1024)
        file_size = len(byte_sample)

        lower_name = filename.lower()
        if "receipt" in lower_name:
            doc_type = "Receipt"
            vendor = "Metro Cafe & Market"
            invoice_num = "REC-88021"
            subtotal = 42.50
            tax = 3.61
            total = 46.11
            line_items = [
                {"description": "Organic Espresso", "quantity": 2, "unit_price": 4.50, "total": 9.00},
                {"description": "Artisan Panini", "quantity": 2, "unit_price": 14.00, "total": 28.00},
                {"description": "Mineral Water", "quantity": 1, "unit_price": 5.50, "total": 5.50},
            ]
            fields = [
                ExtractedFieldItem(
                    field_name="document_type",
                    original_value="RECEIPT",
                    normalized_value="Receipt",
                    confidence_score=0.99,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.35, y_min=0.04, x_max=0.65, y_max=0.08),
                ),
                ExtractedFieldItem(
                    field_name="receipt_number",
                    original_value="Order # REC-88021",
                    normalized_value="REC-88021",
                    confidence_score=0.97,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.30, y_min=0.10, x_max=0.70, y_max=0.13),
                ),
                ExtractedFieldItem(
                    field_name="vendor_name",
                    original_value="Metro Cafe & Market",
                    normalized_value="Metro Cafe & Market",
                    confidence_score=0.98,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.25, y_min=0.14, x_max=0.75, y_max=0.18),
                ),
                ExtractedFieldItem(
                    field_name="transaction_date",
                    original_value="08/28/2026 09:14 AM",
                    normalized_value="2026-08-28T09:14:00",
                    confidence_score=0.96,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.28, y_min=0.19, x_max=0.72, y_max=0.22),
                ),
                ExtractedFieldItem(
                    field_name="subtotal",
                    original_value="$42.50",
                    normalized_value="42.50",
                    confidence_score=0.99,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.60, y_min=0.65, x_max=0.88, y_max=0.68),
                ),
                ExtractedFieldItem(
                    field_name="tax_amount",
                    original_value="$3.61",
                    normalized_value="3.61",
                    confidence_score=0.98,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.60, y_min=0.69, x_max=0.88, y_max=0.72),
                ),
                ExtractedFieldItem(
                    field_name="total_amount",
                    original_value="$46.11",
                    normalized_value="46.11",
                    confidence_score=0.99,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.60, y_min=0.73, x_max=0.88, y_max=0.77),
                ),
            ]
        else:
            doc_type = "Commercial Invoice"
            vendor = "Acme Global Solutions Inc."
            invoice_num = "INV-2026-0892"
            subtotal = 1250.00
            tax = 100.00
            total = 1350.00
            line_items = [
                {"description": "Cloud Architecture Consulting (Hours)", "quantity": 10, "unit_price": 100.00, "total": 1000.00},
                {"description": "Platform Maintenance Subscription", "quantity": 1, "unit_price": 250.00, "total": 250.00},
            ]
            fields = [
                ExtractedFieldItem(
                    field_name="document_type",
                    original_value="COMMERCIAL INVOICE",
                    normalized_value="Commercial Invoice",
                    confidence_score=0.99,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.65, y_min=0.05, x_max=0.92, y_max=0.09),
                ),
                ExtractedFieldItem(
                    field_name="invoice_number",
                    original_value="Invoice No: INV-2026-0892",
                    normalized_value="INV-2026-0892",
                    confidence_score=0.98,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.65, y_min=0.10, x_max=0.92, y_max=0.14),
                ),
                ExtractedFieldItem(
                    field_name="vendor_name",
                    original_value="Acme Global Solutions Inc.",
                    normalized_value="Acme Global Solutions Inc.",
                    confidence_score=0.98,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.08, y_min=0.06, x_max=0.45, y_max=0.10),
                ),
                ExtractedFieldItem(
                    field_name="vendor_tax_id",
                    original_value="EIN: US-987654321",
                    normalized_value="US-987654321",
                    confidence_score=0.95,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.08, y_min=0.11, x_max=0.35, y_max=0.14),
                ),
                ExtractedFieldItem(
                    field_name="vendor_address",
                    original_value="100 Tech Enterprise Blvd, Suite 400, San Francisco, CA",
                    normalized_value="100 Tech Enterprise Blvd, Suite 400, San Francisco, CA",
                    confidence_score=0.94,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.08, y_min=0.15, x_max=0.55, y_max=0.20),
                ),
                ExtractedFieldItem(
                    field_name="customer_name",
                    original_value="Bill To: Vertex Dynamics Ltd.",
                    normalized_value="Vertex Dynamics Ltd.",
                    confidence_score=0.97,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.08, y_min=0.24, x_max=0.42, y_max=0.28),
                ),
                ExtractedFieldItem(
                    field_name="issue_date",
                    original_value="August 20, 2026",
                    normalized_value="2026-08-20",
                    confidence_score=0.96,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.65, y_min=0.15, x_max=0.90, y_max=0.18),
                ),
                ExtractedFieldItem(
                    field_name="due_date",
                    original_value="September 20, 2026",
                    normalized_value="2026-09-20",
                    confidence_score=0.95,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.65, y_min=0.19, x_max=0.90, y_max=0.22),
                ),
                ExtractedFieldItem(
                    field_name="subtotal",
                    original_value="$1,250.00",
                    normalized_value="1250.00",
                    confidence_score=0.99,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.72, y_min=0.75, x_max=0.92, y_max=0.78),
                ),
                ExtractedFieldItem(
                    field_name="tax_amount",
                    original_value="$100.00",
                    normalized_value="100.00",
                    confidence_score=0.98,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.72, y_min=0.79, x_max=0.92, y_max=0.82),
                ),
                ExtractedFieldItem(
                    field_name="total_amount",
                    original_value="$1,350.00",
                    normalized_value="1350.00",
                    confidence_score=0.99,
                    source_page=1,
                    bounding_box=BoundingBox(x_min=0.72, y_min=0.83, x_max=0.92, y_max=0.87),
                ),
            ]

        # Structured dictionary representation
        structured_data = {
            "document_type": doc_type,
            "invoice_number": invoice_num,
            "vendor": {
                "name": vendor,
                "address": "100 Tech Enterprise Blvd, Suite 400, San Francisco, CA",
            },
            "financials": {
                "currency": "USD",
                "subtotal": subtotal,
                "tax_amount": tax,
                "total_amount": total,
            },
            "line_items": line_items,
            "fields_count": len(fields),
        }

        # Validation Checks
        checks_passed = []
        validation_errors = []

        computed_total = round(subtotal + tax, 2)
        if computed_total == round(total, 2):
            checks_passed.append("financial_sum_validation: Subtotal + Tax == Total Amount")
        else:
            validation_errors.append(f"Math mismatch: {subtotal} + {tax} != {total}")

        is_valid = len(validation_errors) == 0
        validation_info = {
            "checks_passed": checks_passed,
            "errors": validation_errors,
            "rules_evaluated_count": len(checks_passed) + len(validation_errors),
        }

        avg_confidence = round(sum(f.confidence_score for f in fields) / len(fields), 2)
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        return ProcessingResult(
            extracted_data=structured_data,
            fields=fields,
            confidence_score=avg_confidence,
            is_valid=is_valid,
            validation_info=validation_info,
            processing_time_ms=max(duration_ms, 15),
        )


def get_document_processor() -> BaseDocumentProcessor:
    """Factory function returning the active pipeline processor."""
    return MockDocumentProcessor()
