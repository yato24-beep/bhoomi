"""Post-processing and text normalization utilities for Land Record Digitization."""

from src.postprocessing.kannada_normalizer import (
    KANNADA_LAND_RECORD_LEXICON,
    KannadaNormalizer,
    NormalizationResult,
)
from src.postprocessing.beam_rescorer import (
    KannadaBeamRescorer,
    RescoringResult,
    CandidateInfo,
)

from src.postprocessing.multi_variant_ocr import (
    MultiVariantBeamEvaluator,
    MultiVariantResult,
    generate_crop_variants,
)

__all__ = [
    "KannadaNormalizer",
    "NormalizationResult",
    "KANNADA_LAND_RECORD_LEXICON",
    "KannadaBeamRescorer",
    "RescoringResult",
    "CandidateInfo",
    "MultiVariantBeamEvaluator",
    "MultiVariantResult",
    "generate_crop_variants",
]
