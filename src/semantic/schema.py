"""Land Record Semantic Schema and Structured Representation Contracts.

Defines canonical schema for Karnataka land records (RTC, Pahani, Mutation Register,
Sale Deed) supporting repeated records (joint owners, multiple parcels) and strict
provenance links back to original OCR bounding boxes and raw transcriptions.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from schemas import BoundingBox


class ValidationStatus(str, Enum):
    """Field-level verification status."""
    VALID = "VALID"
    INVALID = "INVALID"
    WARNING = "WARNING"
    UNVERIFIED = "UNVERIFIED"


class FieldProvenance(BaseModel):
    """Immutable audit trail linking extracted semantic values back to raw OCR evidence."""
    document_id: Optional[str] = Field(None, description="Source document identifier")
    page_number: int = Field(1, ge=1, description="1-indexed document page number")
    region_id: str = Field(..., description="Unique source OCR region identifier")
    bbox: Optional[BoundingBox] = Field(None, description="Spatial coordinates of source text")
    raw_ocr_text: str = Field(..., description="Unaltered raw recognized OCR string")
    normalized_value: str = Field(..., description="Standardized or cleaned value")
    extraction_method: str = Field("rule_spatial", description="Method used: 'rule_spatial', 'regex', 'table_cell'")


class ValidationResult(BaseModel):
    """Validation report for an individual field."""
    status: ValidationStatus = Field(ValidationStatus.UNVERIFIED, description="Validation status")
    reason: Optional[str] = Field(None, description="Actionable explanation or error diagnostic")
    rule_name: str = Field("default_check", description="Name of validation rule executed")


class SemanticFieldItem(BaseModel):
    """Standardized representation of a single semantic field."""
    field_name: str = Field(..., description="Canonical field identifier")
    value: str = Field(..., description="Standardized value")
    raw_value: str = Field(..., description="Original raw extracted text from OCR")
    validation_status: ValidationStatus = Field(ValidationStatus.UNVERIFIED)
    validation_reason: Optional[str] = Field(None)
    source_region_ids: List[str] = Field(default_factory=list, description="List of source region IDs")
    provenance: Optional[FieldProvenance] = Field(None, description="Detailed evidence provenance")
    confidence: Optional[float] = Field(None, description="Calibrated / engineering confidence score")
    model_confidence: Optional[float] = Field(None, description="Uncalibrated confidence reported directly by AI engine")
    conflicts: List[str] = Field(default_factory=list, description="Conflicting candidate values detected during extraction")
    english_value: Optional[str] = Field(None, description="English translation or transliteration")
    translation_status: Optional[str] = Field(None, description="Translation state: TRANSLATED, ALREADY_ENGLISH, NOT_APPLICABLE, FAILED, UNAVAILABLE")
    translation_engine: Optional[str] = Field(None, description="Translation engine identifier")

    @property
    def status(self) -> ValidationStatus:
        return self.validation_status

    @property
    def reason(self) -> Optional[str]:
        return self.validation_reason

    @property
    def normalized_value(self) -> str:
        return self.value


class OwnerRecord(BaseModel):
    """Represents an individual land owner or co-sharer."""
    owner_name: SemanticFieldItem
    father_name: Optional[SemanticFieldItem] = None
    relationship: Optional[str] = Field(None, description="e.g. 'son of', 'wife of', 'ತಂದೆ', 'ಗಂಡ'")
    share_fraction: Optional[SemanticFieldItem] = None
    remarks: Optional[str] = None


class BoundaryRecord(BaseModel):
    """Four-sided property boundaries (ಚಕ್ಕುಬಂದಿ / Chakkubandi)."""
    east: Optional[SemanticFieldItem] = None
    west: Optional[SemanticFieldItem] = None
    north: Optional[SemanticFieldItem] = None
    south: Optional[SemanticFieldItem] = None


class CadastralRecord(BaseModel):
    """Survey and parcel identifier details."""
    survey_number: SemanticFieldItem
    hissa_number: Optional[SemanticFieldItem] = None
    khata_number: Optional[SemanticFieldItem] = None
    extent: Optional[SemanticFieldItem] = None
    land_type: Optional[SemanticFieldItem] = None  # Dry, Wet, Garden, Kharab
    assessment: Optional[SemanticFieldItem] = None  # Revenue tax amount


class SemanticTableCell(BaseModel):
    """Single cell within a reconstructed semantic table."""
    row_index: int
    col_index: int
    text: str
    bbox: Optional[BoundingBox] = None
    source_region_id: Optional[str] = None
    confidence: Optional[float] = None


class SemanticTable(BaseModel):
    """Reconstructed tabular section maintaining geometry and cell linkage."""
    table_id: str
    page_number: int = 1
    bbox: Optional[BoundingBox] = None
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    cells: List[SemanticTableCell] = Field(default_factory=list)

    @property
    def num_rows(self) -> int:
        if self.cells:
            return max(c.row_index for c in self.cells) + 1
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        if self.cells:
            return max(c.col_index for c in self.cells) + 1
        return len(self.headers)


class LandRecordDocument(BaseModel):
    """Canonical land record document semantic structure (RTC / Pahani / Mutation Register)."""
    model_config = ConfigDict(protected_namespaces=())

    document_id: Optional[str] = None
    document_type: str = Field("Unknown / Not classified", description="Classified document type or 'Unknown / Not classified'")
    
    # Jurisdiction / Location
    district: Optional[SemanticFieldItem] = None
    taluk: Optional[SemanticFieldItem] = None
    hobli: Optional[SemanticFieldItem] = None
    village: Optional[SemanticFieldItem] = None

    # Cadastral Identifiers
    survey_number: Optional[SemanticFieldItem] = None
    hissa_number: Optional[SemanticFieldItem] = None
    khata_number: Optional[SemanticFieldItem] = None
    
    # Parties / Ownership & Cultivation
    owner_name: Optional[SemanticFieldItem] = None
    cultivator_name: Optional[SemanticFieldItem] = None
    
    # Land Attributes
    extent: Optional[SemanticFieldItem] = None
    land_type: Optional[SemanticFieldItem] = None
    assessment: Optional[SemanticFieldItem] = None
    
    # Mutation & Transaction Records
    mutation_number: Optional[SemanticFieldItem] = None
    registration_number: Optional[SemanticFieldItem] = None
    document_number: Optional[SemanticFieldItem] = None
    record_date: Optional[SemanticFieldItem] = None
    registration_date: Optional[SemanticFieldItem] = None
    remarks: Optional[SemanticFieldItem] = None

    # Repeated Records
    owners: List[OwnerRecord] = Field(default_factory=list, description="Repeated joint owners or parties")
    cadastral_records: List[CadastralRecord] = Field(default_factory=list, description="Repeated survey parcel records")
    boundaries: Optional[BoundaryRecord] = None
    tables: List[SemanticTable] = Field(default_factory=list, description="Extracted table structures with cell geometry")
    
    # Arbitrary additional extracted fields
    extra_fields: Dict[str, SemanticFieldItem] = Field(default_factory=dict)
    
    # Summary of validation
    validation_summary: Dict[str, Any] = Field(default_factory=dict)

    @property
    def fields(self) -> Dict[str, SemanticFieldItem]:
        """Provides backward-compatible dictionary mapping of all populated fields."""
        f_dict: Dict[str, SemanticFieldItem] = {}
        for k in (
            "document_type", "district", "taluk", "hobli", "village", "survey_number",
            "hissa_number", "khata_number", "owner_name", "cultivator_name", "extent",
            "land_type", "assessment", "mutation_number", "registration_number",
            "document_number", "record_date", "registration_date", "remarks",
        ):
            val = getattr(self, k, None)
            if val is not None and isinstance(val, SemanticFieldItem):
                f_dict[k] = val
        f_dict.update(self.extra_fields)
        return f_dict

    @property
    def requires_human_review(self) -> bool:
        """Determines if the document requires human review based on validation summary."""
        if self.validation_summary.get("gate_status") == "cadastral_extraction_suppressed":
            return False
        invalid = self.validation_summary.get("invalid_fields_count", 0)
        warning = self.validation_summary.get("warning_fields_count", 0)
        return (invalid > 0 or warning > 0 or self.confidence_score < 0.70)

    @property
    def confidence_score(self) -> float:
        """Calculates document-level confidence score from field confidences."""
        confs = [f.confidence for f in self.fields.values() if f.confidence is not None]
        if confs:
            return round(sum(confs) / len(confs), 2)
        if self.validation_summary.get("gate_status") == "cadastral_extraction_suppressed":
            return 1.0
        return 0.0

