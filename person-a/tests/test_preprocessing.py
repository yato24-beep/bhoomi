"""person-a/tests/test_preprocessing.py
Unit tests for quality analyzer, deskew, enhancement, and super-resolution.
"""

import numpy as np
import pytest

from src.preprocessing.deskew import DocumentDeskewer
from src.preprocessing.enhancement import ImageEnhancer
from src.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from src.preprocessing.quality import ImageQualityAnalyzer
from src.preprocessing.super_resolution import SuperResolutionEnhancer


def test_quality_analyzer_blur_detection():
    analyzer = ImageQualityAnalyzer()
    sharp = np.zeros((200, 200, 3), dtype=np.uint8)
    sharp[::4, ::4] = 255
    q_sharp = analyzer.analyze(sharp)
    assert q_sharp.blur_score > 100.0
    assert q_sharp.is_blurry is False

    blurry = np.full((200, 200, 3), 128, dtype=np.uint8)
    q_blur = analyzer.analyze(blurry)
    assert q_blur.blur_score < 10.0
    assert q_blur.is_blurry is True


def test_quality_analyzer_blank_page():
    analyzer = ImageQualityAnalyzer()
    blank = np.full((300, 300, 3), 255, dtype=np.uint8)
    q = analyzer.analyze(blank)
    assert q.is_blank is True


def test_deskew_upright_image():
    deskewer = DocumentDeskewer()
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    angle = deskewer.estimate_skew(img)
    assert abs(angle) < 1.0


def test_enhancement_operations():
    enhancer = ImageEnhancer()
    img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    clahe_out = enhancer.apply_clahe(img)
    assert clahe_out.shape == img.shape
    denoise_out = enhancer.denoise(img)
    assert denoise_out.shape == img.shape
    sharpen_out = enhancer.sharpen(img)
    assert sharpen_out.shape == img.shape


def test_super_resolution_execution():
    sr = SuperResolutionEnhancer(scale=2)
    small = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    enhanced = sr.enhance(small)
    assert enhanced.shape[0] == 100
    assert enhanced.shape[1] == 100


def test_preprocessing_pipeline_blank_short_circuit():
    pipeline = PreprocessingPipeline()
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    pages = pipeline.process(blank)
    assert len(pages) == 1
    assert pages[0].quality.is_blank is True
    assert "blank_page_detected" in pages[0].operations_applied
