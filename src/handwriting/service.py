"""Service layer for handwritten text recognition in land record documents.

Provides high-level orchestration, crop batching, and integration with the
shared DocumentPage and OCRResult schemas via dependency-injected recognizers.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from schemas import BoundingBox, DocumentPage, OCRResult
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput


class HandwritingOCRService:
    """Service for orchestrating handwritten OCR tasks using pluggable recognizers.

    Uses dependency injection to decouple downstream business logic from specific
    OCR models (e.g. initial baseline vs. fine-tuned TrOCR).
    """

    def __init__(self, recognizer: BaseHandwritingRecognizer):
        """Initializes the service with an injected recognizer.

        Args:
            recognizer: An instance implementing BaseHandwritingRecognizer.

        Raises:
            TypeError: If the provided recognizer does not inherit from BaseHandwritingRecognizer.
        """
        self.set_recognizer(recognizer)

    @property
    def recognizer(self) -> BaseHandwritingRecognizer:
        """Returns the active recognizer instance."""
        return self._recognizer

    def set_recognizer(self, recognizer: BaseHandwritingRecognizer) -> None:
        """Swaps the active recognizer implementation.

        Allows dynamic switching between different model backends (e.g. baseline vs fine-tuned).

        Args:
            recognizer: New recognizer instance.
        """
        if not isinstance(recognizer, BaseHandwritingRecognizer):
            raise TypeError(
                f"Recognizer must be an instance of BaseHandwritingRecognizer, got {type(recognizer).__name__}"
            )
        self._recognizer = recognizer

    def get_model_info(self) -> Dict[str, str]:
        """Returns metadata about the active recognition model."""
        return {
            "model_name": self._recognizer.model_name,
            "model_version": self._recognizer.model_version,
            "recognizer_class": type(self._recognizer).__name__,
        }

    def process_crop(
        self,
        image: ImageInput,
        bbox: Optional[BoundingBox] = None,
        page_number: Optional[int] = None,
        preprocessing_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Processes a single handwriting crop.

        Args:
            image: Image crop input (file path, image object, or bytes).
            bbox: Optional coordinates of the bounding box.
            page_number: Optional 1-indexed page number.
            preprocessing_info: Optional details from preprocessing stage.
            **kwargs: Extra parameters passed to the recognizer.

        Returns:
            OCRResult object with recognized text and audit metadata.
        """
        return self._recognizer.recognize_handwriting(
            image=image,
            bbox=bbox,
            page_number=page_number,
            preprocessing_info=preprocessing_info,
            **kwargs,
        )

    def process_crops_batch(
        self,
        images: Sequence[ImageInput],
        bboxes: Optional[Sequence[Optional[BoundingBox]]] = None,
        page_numbers: Optional[Sequence[Optional[int]]] = None,
        preprocessing_info_list: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        **kwargs: Any,
    ) -> List[OCRResult]:
        """Processes multiple handwriting crops in batch.

        Args:
            images: Sequence of image inputs.
            bboxes: Optional matching bounding boxes.
            page_numbers: Optional matching page numbers.
            preprocessing_info_list: Optional matching preprocessing info dicts.
            **kwargs: Extra parameters passed to the recognizer.

        Returns:
            List of OCRResult objects.
        """
        return self._recognizer.recognize_batch(
            images=images,
            bboxes=bboxes,
            page_numbers=page_numbers,
            preprocessing_info_list=preprocessing_info_list,
            **kwargs,
        )

    def process_document_page_crops(
        self,
        page: DocumentPage,
        crop_images: Sequence[ImageInput],
        crop_bboxes: Optional[Sequence[Optional[BoundingBox]]] = None,
        preprocessing_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DocumentPage:
        """Processes handwriting crops for a given DocumentPage and updates its ocr_results.

        Args:
            page: The DocumentPage object to process and update.
            crop_images: Sequence of cropped image regions from the page.
            crop_bboxes: Optional bounding boxes corresponding to the cropped regions.
            preprocessing_info: Optional page-level preprocessing details.
            **kwargs: Extra parameters passed to the recognizer.

        Returns:
            Updated DocumentPage with new OCRResult entries appended.
        """
        page_numbers = [page.page_number] * len(crop_images)
        prep_info_list = [preprocessing_info] * len(crop_images) if preprocessing_info else None

        results = self.process_crops_batch(
            images=crop_images,
            bboxes=crop_bboxes,
            page_numbers=page_numbers,
            preprocessing_info_list=prep_info_list,
            **kwargs,
        )

        page.ocr_results.extend(results)
        return page
