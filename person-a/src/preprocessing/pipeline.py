"""person-a/src/preprocessing/pipeline.py
Adaptive OpenCV Preprocessing Pipeline Orchestrator.
Analyzes quality -> applies selective deskew, denoising, CLAHE, super-res -> preserves metadata.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np

from .deskew import DocumentDeskewer
from .enhancement import ImageEnhancer
from .loader import load_document_images
from .quality import ImageQualityAnalyzer
from .super_resolution import SuperResolutionEnhancer
from ..schemas import ImageQualityAssessment


@dataclass
class PreprocessingConfig:
    target_dpi: int = 300
    enable_deskew: bool = True
    enable_denoise: bool = True
    enable_clahe: bool = True
    enable_sharpen: bool = True
    enable_super_resolution: bool = True
    sr_dpi_threshold: float = 120.0
    blur_threshold: float = 100.0


@dataclass
class PreprocessedPage:
    page_number: int
    original_image: np.ndarray
    processed_image: np.ndarray
    quality: ImageQualityAssessment
    operations_applied: List[str] = field(default_factory=list)


class PreprocessingPipeline:
    """Orchestrates quality evaluation and adaptive enhancement per page."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.quality_analyzer = ImageQualityAnalyzer(
            blur_threshold=self.config.blur_threshold,
            default_dpi=float(self.config.target_dpi),
        )
        self.deskewer = DocumentDeskewer()
        self.enhancer = ImageEnhancer()
        self.super_res = SuperResolutionEnhancer()

    def process(
        self,
        document_source: Union[str, bytes, np.ndarray, List[np.ndarray]],
    ) -> List[PreprocessedPage]:
        """Ingest document source and return list of PreprocessedPage instances."""
        if isinstance(document_source, list) and all(isinstance(x, np.ndarray) for x in document_source):
            raw_pages = document_source
        else:
            raw_pages = load_document_images(document_source, target_dpi=self.config.target_dpi)

        if not raw_pages:
            return []

        processed_pages: List[PreprocessedPage] = []
        for idx, page_img in enumerate(raw_pages, start=1):
            processed_pages.append(self._process_single_page(idx, page_img))

        return processed_pages

    def _process_single_page(self, page_num: int, original: np.ndarray) -> PreprocessedPage:
        quality = self.quality_analyzer.analyze(original)
        ops: List[str] = []

        if quality.is_blank:
            quality.preprocessing_applied = ["blank_page_detected"]
            return PreprocessedPage(
                page_number=page_num,
                original_image=original,
                processed_image=original,
                quality=quality,
                operations_applied=["blank_page_detected"],
            )

        current = original.copy()

        # 1. Deskew & Rotation
        if self.config.enable_deskew:
            deskewed, angle = self.deskewer.process(current)
            if abs(angle) >= 0.3:
                current = deskewed
                quality.skew_angle = angle
                ops.append(f"deskew_{angle:+.1f}deg")

        # 2. Super-resolution for low-DPI scans
        if self.config.enable_super_resolution and quality.dpi <= self.config.sr_dpi_threshold:
            current = self.super_res.enhance(current)
            ops.append("super_resolution_2x")

        # 3. Denoising
        if self.config.enable_denoise:
            current = self.enhancer.denoise(current)
            ops.append("bilateral_denoise")

        # 4. Contrast Enhancement (CLAHE)
        if self.config.enable_clahe and quality.contrast_score < 60.0:
            current = self.enhancer.apply_clahe(current)
            ops.append("clahe_contrast_enhancement")

        # 5. Sharpening
        if self.config.enable_sharpen and quality.is_blurry:
            current = self.enhancer.sharpen(current)
            ops.append("unsharp_sharpening")

        quality.preprocessing_applied = ops
        return PreprocessedPage(
            page_number=page_num,
            original_image=original,
            processed_image=current,
            quality=quality,
            operations_applied=ops,
        )
