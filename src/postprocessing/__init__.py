"""Post-processing and text normalization utilities for Land Record Digitization."""

from src.postprocessing.kannada_normalizer import (
    KANNADA_LAND_RECORD_LEXICON,
    KannadaNormalizer,
    NormalizationResult,
)

__all__ = [
    "KannadaNormalizer",
    "NormalizationResult",
    "KANNADA_LAND_RECORD_LEXICON",
]
