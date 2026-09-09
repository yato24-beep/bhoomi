"""Human Correction & Active Learning module for Land Record Digitization."""

from src.correction.schemas import HumanCorrectionRecord
from src.correction.service import CorrectionService, DEFAULT_CORRECTIONS_PATH

__all__ = [
    "HumanCorrectionRecord",
    "CorrectionService",
    "DEFAULT_CORRECTIONS_PATH",
]
