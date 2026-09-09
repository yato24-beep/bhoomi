"""person-a/src/handwriting/adapter.py
Interface Contract for Person B's Handwriting OCR (TrOCR / HuggingFace).
Provides an abstract base class, dynamic registration, and a clearly labeled stub placeholder.
"""

import abc
from typing import List, Optional
import numpy as np
from pydantic import BaseModel, Field

from ..schemas import BoundingBox


class HandwritingRegionResult(BaseModel):
    """Recognized handwritten region from Person B's TrOCR engine."""
    region_id: str
    text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    page_number: int = 1
    bbox: BoundingBox
    model_version: str = "trocr-base-landrecords-v1"
    source_region: str = "handwriting_fill"
    preprocessing_applied: List[str] = Field(default_factory=list)


class HandwritingResult(BaseModel):
    """Standard collection of handwritten regions recognized across document."""
    document_id: str
    regions: List[HandwritingRegionResult] = Field(default_factory=list)
    model_version: str = "trocr-base-landrecords-v1"
    processing_time_ms: float = 0.0


class HandwritingAdapter(abc.ABC):
    """Abstract interface defining the integration contract for Person B."""

    @abc.abstractmethod
    def recognize_region(
        self,
        crop: np.ndarray,
        bbox: BoundingBox,
        page_number: int = 1,
        region_id: Optional[str] = None,
    ) -> HandwritingRegionResult:
        """Recognize text in a cropped handwritten region."""
        pass


class StubHandwritingAdapter(HandwritingAdapter):
    """Awaiting Person B fine-tuned TrOCR checkpoint."""

    def recognize_region(
        self,
        crop: np.ndarray,
        bbox: BoundingBox,
        page_number: int = 1,
        region_id: Optional[str] = None,
    ) -> HandwritingRegionResult:
        reg_id = region_id or f"hw_page_{page_number}_{int(bbox.x_min)}_{int(bbox.y_min)}"
        return HandwritingRegionResult(
            region_id=reg_id,
            text="",
            confidence=1.0,
            page_number=page_number,
            bbox=bbox,
            model_version="stub-awaiting-person-b",
            source_region="handwriting_fill",
            preprocessing_applied=["stub_placeholder"],
        )


_active_handwriting_adapter: HandwritingAdapter = StubHandwritingAdapter()


def get_handwriting_adapter() -> HandwritingAdapter:
    """Return active handwriting adapter."""
    return _active_handwriting_adapter


def register_handwriting_adapter(adapter: HandwritingAdapter) -> None:
    """Allow Person B to register their live model implementation without modifying Person A core code."""
    global _active_handwriting_adapter
    if not isinstance(adapter, HandwritingAdapter):
        raise TypeError("Custom adapter must implement HandwritingAdapter interface")
    _active_handwriting_adapter = adapter
