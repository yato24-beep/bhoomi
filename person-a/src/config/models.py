"""person-a/src/config/models.py
Pydantic data models for state-specific configurations.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PreprocessingParams(BaseModel):
    target_dpi: int = 300
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    apply_super_resolution: bool = False
    deskew_max_angle: float = 45.0


class OCRParams(BaseModel):
    default_language: str = "en"
    supported_languages: List[str] = Field(default_factory=lambda: ["en", "hi"])
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.5
    drop_score: float = 0.4
    use_angle_cls: bool = True


class DocumentTypeRule(BaseModel):
    document_type: str
    display_name: str
    mandatory_keywords: List[str] = Field(default_factory=list)
    optional_keywords: List[str] = Field(default_factory=list)
    negative_keywords: List[str] = Field(default_factory=list)
    min_keyword_matches: int = 1
    layout_signals: List[str] = Field(default_factory=list)
    aspect_ratio_range: Optional[List[float]] = None
    confidence_weight: float = 1.0


class StateConfiguration(BaseModel):
    state_code: str
    state_name: str
    primary_language: str
    fallback_languages: List[str] = Field(default_factory=list)
    document_types: List[DocumentTypeRule] = Field(default_factory=list)
    preprocessing_overrides: PreprocessingParams = Field(default_factory=PreprocessingParams)
    ocr_overrides: OCRParams = Field(default_factory=OCRParams)
