"""person-a/src/handwriting/__init__.py"""
from .adapter import (
    HandwritingAdapter,
    HandwritingRegionResult,
    HandwritingResult,
    StubHandwritingAdapter,
    get_handwriting_adapter,
    register_handwriting_adapter,
)

__all__ = [
    "HandwritingAdapter",
    "HandwritingRegionResult",
    "HandwritingResult",
    "StubHandwritingAdapter",
    "get_handwriting_adapter",
    "register_handwriting_adapter",
]
