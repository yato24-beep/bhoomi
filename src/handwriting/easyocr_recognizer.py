"""Regional script recognizer powered by EasyOCR (Kannada printed primary backend).

Provides high-accuracy printed and display Indic text recognition, specifically
excelling at complex Kannada conjunct consonants (ottaksharas) where stock CTC
models exhibit systemic representation dropouts.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from schemas import BoundingBox, OCREngineType, OCRResult
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput

logger = logging.getLogger("easyocr_recognizer")

# Lazy import handle for easyocr
_EASYOCR_READER_CACHE: Dict[str, Any] = {}


def get_easyocr_reader(languages: Sequence[str] = ("kn", "en"), gpu: bool = False) -> Any:
    """Retrieves or instantiates a cached EasyOCR reader instance for the specified languages."""
    import easyocr
    cache_key = f"{','.join(sorted(languages))}_gpu_{gpu}"
    if cache_key not in _EASYOCR_READER_CACHE:
        logger.info(f"Initializing EasyOCR reader with languages={languages}, gpu={gpu}...")
        _EASYOCR_READER_CACHE[cache_key] = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
    return _EASYOCR_READER_CACHE[cache_key]


def get_easyocr_recognizer(**kwargs: Any) -> "EasyOCRKannadaRecognizer":
    """Factory helper returning an EasyOCRKannadaRecognizer instance."""
    return EasyOCRKannadaRecognizer(**kwargs)


class EasyOCRKannadaRecognizer(BaseHandwritingRecognizer):
    """Production printed-Kannada recognizer backed by EasyOCR.

    Maintains full compatibility with BaseHandwritingRecognizer and schemas.py.
    Provides line clustering, reading order preservation, and conjunct fidelity.
    """

    def __init__(
        self,
        model_name: str = "easyocr-kannada",
        model_version: str = "1.7.2",
        languages: Sequence[str] = ("kn", "en"),
        gpu: bool = False,
        confidence_threshold: float = 0.20,
        auto_load: bool = False,
        **kwargs: Any,
    ):
        """Initializes the EasyOCR recognizer.

        Args:
            model_name: Engine identifier string.
            model_version: Underlying engine version descriptor.
            languages: Sequence of language codes (default: ('kn', 'en') for bilingual land records).
            gpu: Whether to utilize CUDA acceleration.
            confidence_threshold: Cutoff below which text segments are flagged.
            auto_load: If True, eagerly initializes the EasyOCR reader at construction.
        """
        super().__init__(model_name=model_name, model_version=model_version)
        self.languages = tuple(languages)
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        self._reader: Optional[Any] = None

        if auto_load:
            self._load_engine()

    def _load_engine(self) -> Any:
        """Loads and caches the EasyOCR reader instance."""
        if self._reader is None:
            self._reader = get_easyocr_reader(self.languages, gpu=self.gpu)
        return self._reader

    @staticmethod
    def _convert_image_to_numpy(image: ImageInput) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Converts diverse image input types to RGB numpy array and returns (array, (width, height))."""
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                pil_img = Image.fromarray(image).convert("RGB")
            elif image.ndim == 3 and image.shape[2] == 4:
                pil_img = Image.fromarray(image).convert("RGB")
            else:
                pil_img = Image.fromarray(image)
        else:
            raise TypeError(f"Unsupported image input type for EasyOCR: {type(image)}")

        arr = np.array(pil_img)
        return arr, pil_img.size

    @staticmethod
    def _sort_reading_order(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sorts detected boxes top-to-bottom, left-to-right using dynamic line clustering.

        Boxes whose vertical centers overlap within line_tolerance are treated as belonging
        to the same line and sorted left-to-right.
        """
        if not detections:
            return []

        # Compute mid-y and height for each box
        enriched = []
        for det in detections:
            bbox = det["bbox"]
            mid_y = (bbox.y_min + bbox.y_max) / 2.0
            height = max(1.0, bbox.y_max - bbox.y_min)
            enriched.append((mid_y, height, det))

        # Sort primarily by mid_y
        enriched.sort(key=lambda item: item[0])

        # Cluster into lines
        lines: List[List[Dict[str, Any]]] = []
        current_line: List[Dict[str, Any]] = []
        current_y = enriched[0][0]
        current_h = enriched[0][1]

        for mid_y, height, det in enriched:
            # If within half of average line height, group in same line
            line_tol = max(10.0, min(current_h, height) * 0.55)
            if abs(mid_y - current_y) <= line_tol:
                current_line.append(det)
            else:
                # Sort completed line left-to-right
                current_line.sort(key=lambda d: d["bbox"].x_min)
                lines.append(current_line)
                current_line = [det]
                current_y = mid_y
                current_h = height

        if current_line:
            current_line.sort(key=lambda d: d["bbox"].x_min)
            lines.append(current_line)

        # Flatten in reading order
        ordered = []
        for l in lines:
            ordered.extend(l)
        return ordered

    def recognize_handwriting(
        self,
        image: ImageInput,
        bbox: Optional[BoundingBox] = None,
        page_number: Optional[int] = None,
        preprocessing_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Executes EasyOCR recognition on an image or cropped region.

        Args:
            image: Path, PIL Image, or numpy array.
            bbox: Optional crop coordinates.
            page_number: Source document page index.
            preprocessing_info: Preprocessing metadata.
            **kwargs: Extra parameters passed to reader.readtext.

        Returns:
            Standardized OCRResult containing recognized Kannada text, confidence, and line details.
        """
        start_time = time.perf_counter()
        img_arr, (img_w, img_h) = self._convert_image_to_numpy(image)

        # Apply crop if bbox provided and not already cropped
        if bbox is not None:
            x1 = max(0, int(bbox.x_min))
            y1 = max(0, int(bbox.y_min))
            x2 = min(img_w, int(bbox.x_max))
            y2 = min(img_h, int(bbox.y_max))
            if x2 > x1 and y2 > y1 and (x2 - x1 < img_w or y2 - y1 < img_h):
                img_arr = img_arr[y1:y2, x1:x2]

        reader = self._load_engine()

        # Run EasyOCR
        try:
            raw_res = reader.readtext(img_arr, detail=1, paragraph=False)
        except Exception as exc:
            logger.error(f"EasyOCR inference error: {exc}", exc_info=True)
            return OCRResult(
                text="",
                confidence=0.0,
                model_name=self.model_name,
                model_version=self.model_version,
                metadata={
                    "error": str(exc),
                    "engine_status": "error",
                    "requires_human_review": True,
                },
            )

        # Process detections
        detections: List[Dict[str, Any]] = []
        for item in raw_res:
            poly, text, conf = item
            text_str = str(text).strip()
            if not text_str:
                continue

            # Compute axis-aligned bounding box from polygon
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            x_min = float(min(xs))
            y_min = float(min(ys))
            x_max = float(max(xs))
            y_max = float(max(ys))

            detections.append({
                "text": text_str,
                "confidence": float(conf),
                "bbox": BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                "poly": poly,
            })

        # Sort reading order
        ordered_dets = self._sort_reading_order(detections)

        # Assemble full text
        line_texts = [d["text"] for d in ordered_dets]
        merged_text = "\n".join(line_texts)

        # Compute character-weighted recognizer confidence
        valid_items = [(d["confidence"], max(1, len(d["text"]))) for d in ordered_dets]
        if valid_items:
            tot_w = sum(w for _, w in valid_items)
            mean_conf = sum(c * w for c, w in valid_items) / float(tot_w)
            recognizer_conf = round(float(mean_conf), 4)
        else:
            recognizer_conf = 0.0

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Format line details compatible with document_pipeline
        line_details = [
            {
                "line_id": f"line_{idx + 1:03d}",
                "text": d["text"],
                "confidence": d["confidence"],
                "bbox": {
                    "x_min": d["bbox"].x_min,
                    "y_min": d["bbox"].y_min,
                    "x_max": d["bbox"].x_max,
                    "y_max": d["bbox"].y_max,
                },
                "engine": OCREngineType.EASYOCR.value,
                "language": "kannada",
            }
            for idx, d in enumerate(ordered_dets)
        ]

        metadata: Dict[str, Any] = {
            "metadata": {
                "line_details": line_details,
                "engine_status": "success",
                "recognizer_confidence": recognizer_conf,
                "latency_ms": elapsed_ms,
                "languages": list(self.languages),
                "num_lines": len(ordered_dets),
                "requires_human_review": recognizer_conf < self.confidence_threshold,
            },
            "preprocessing_info": preprocessing_info or {},
            "bbox": bbox.model_dump() if bbox else None,
            "page_number": page_number,
        }

        return OCRResult(
            text=merged_text,
            confidence=recognizer_conf,
            model_name=self.model_name,
            model_version=self.model_version,
            metadata=metadata,
        )
