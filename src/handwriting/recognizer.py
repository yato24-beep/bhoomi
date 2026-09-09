"""Abstract base interface for handwriting recognition engines.

Defines the contract for Person B handwriting OCR models (e.g. TrOCR, baseline models)
returning standardized, auditable OCRResult instances conforming to schemas.py.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from schemas import BoundingBox, OCRResult

# Supported image input types (path, PIL image, numpy array, or raw bytes)
ImageInput = Union[str, Path, Any]


class BaseHandwritingRecognizer(ABC):
    """Abstract base class defining the standard handwriting OCR interface.

    Any handwritten text recognition model (TrOCR, CRNN, vision-encoder-decoder, etc.)
    must inherit from this interface to ensure interoperability across the pipeline.
    """

    def __init__(self, model_name: str, model_version: str):
        """Initializes recognizer metadata.

        Args:
            model_name: Unique identifier for the model family/architecture.
            model_version: Version tag, checkpoint hash, or iteration descriptor.
        """
        self._model_name = model_name
        self._model_version = model_version

    @property
    def model_name(self) -> str:
        """Returns the recognizer model name."""
        return self._model_name

    @property
    def model_version(self) -> str:
        """Returns the recognizer model version/checkpoint."""
        return self._model_version

    @abstractmethod
    def recognize_handwriting(
        self,
        image: ImageInput,
        bbox: Optional[BoundingBox] = None,
        page_number: Optional[int] = None,
        preprocessing_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Recognizes handwritten text from a single image or crop.

        Args:
            image: Image input (file path, PIL Image, numpy array, etc.).
            bbox: Optional bounding box coordinates for the crop/region.
            page_number: Optional 1-indexed page number in the source document.
            preprocessing_info: Optional details on upstream image enhancements
                (e.g., deskew angle, binarization method).
            **kwargs: Implementation-specific inference parameters.

        Returns:
            Standardized OCRResult conforming to schemas.py containing recognized
            text, auditable confidence, model identification, and metadata.
        """
        pass

    def recognize_batch(
        self,
        images: Sequence[ImageInput],
        bboxes: Optional[Sequence[Optional[BoundingBox]]] = None,
        page_numbers: Optional[Sequence[Optional[int]]] = None,
        preprocessing_info_list: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        **kwargs: Any,
    ) -> List[OCRResult]:
        """Recognizes handwritten text across a batch of image inputs.

        Default implementation processes sequentially; subclasses can override
        for batched tensor inference.

        Args:
            images: Sequence of image inputs.
            bboxes: Optional sequence of bounding boxes matching images.
            page_numbers: Optional sequence of page numbers matching images.
            preprocessing_info_list: Optional sequence of preprocessing dicts.
            **kwargs: Implementation-specific inference parameters.

        Returns:
            List of OCRResult instances corresponding to the inputs.
        """
        results: List[OCRResult] = []
        n = len(images)

        for i in range(n):
            bbox = bboxes[i] if bboxes and i < len(bboxes) else None
            page_num = page_numbers[i] if page_numbers and i < len(page_numbers) else None
            prep_info = preprocessing_info_list[i] if preprocessing_info_list and i < len(preprocessing_info_list) else None

            result = self.recognize_handwriting(
                image=images[i],
                bbox=bbox,
                page_number=page_num,
                preprocessing_info=prep_info,
                **kwargs,
            )
            results.append(result)

        return results
