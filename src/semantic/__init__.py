"""Semantic Understanding and Extraction Layer V1 for Land Records.

Deterministic, rule- and geometry-based semantic extraction without black-box training.
Provides structured schema mapping, field detection, 2D spatial key-value association,
conservative normalization, field-level validation, table reconstruction, and full provenance tracking.
"""

from src.semantic.schema import (
    BoundaryRecord,
    CadastralRecord,
    LandRecordDocument,
    OwnerRecord,
    SemanticFieldItem,
    SemanticTable,
    SemanticTableCell,
    ValidationResult,
    ValidationStatus,
)
from src.semantic.confidence import EvidenceConfidenceCalculator
from src.semantic.engine import (
    BaseSemanticEngine,
    GeminiSemanticEngine,
    LocalSemanticEngine,
    RuleSemanticEngine,
    SemanticEngineResult,
    SemanticEvidence,
    SemanticExtractedField,
)
from src.semantic.field_detector import FieldDetector
from src.semantic.normalizer import SemanticNormalizer
from src.semantic.schema import (
    BoundaryRecord,
    CadastralRecord,
    FieldProvenance,
    LandRecordDocument,
    OwnerRecord,
    SemanticFieldItem,
    SemanticTable,
    SemanticTableCell,
    ValidationResult,
    ValidationStatus,
)
from src.semantic.semantic_pipeline import SemanticPipeline
from src.semantic.table_reconstructor import TableReconstructor
from src.semantic.validator import FieldValidator
from src.semantic.value_associator import ValueAssociator

__all__ = [
    "BoundaryRecord",
    "CadastralRecord",
    "FieldProvenance",
    "LandRecordDocument",
    "OwnerRecord",
    "SemanticFieldItem",
    "SemanticTable",
    "SemanticTableCell",
    "ValidationResult",
    "ValidationStatus",
    "SemanticNormalizer",
    "FieldDetector",
    "ValueAssociator",
    "FieldValidator",
    "TableReconstructor",
    "SemanticPipeline",
    "BaseSemanticEngine",
    "GeminiSemanticEngine",
    "LocalSemanticEngine",
    "RuleSemanticEngine",
    "SemanticEvidence",
    "SemanticExtractedField",
    "SemanticEngineResult",
    "EvidenceConfidenceCalculator",
]

