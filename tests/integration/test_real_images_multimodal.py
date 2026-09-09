"""Real-image multimodal integration test for all 4 supported OCR scenarios:
1. Printed Kannada (PaddleOCR Kannada)
2. Handwritten Kannada (Trained TrOCR Kannada Checkpoint)
3. Printed English (PaddleOCR English)
4. Handwritten English (Pretrained TrOCR English: microsoft/trocr-small-handwritten)
"""

import os
import unittest
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from schemas import BoundingBox, OCRResult
from src.handwriting.router import LanguageScriptRouter
from src.integration.document_pipeline import DocumentProcessingPipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _ensure_sample_images(samples_dir: Path) -> dict:
    """Ensures test images for all 4 modalities exist."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    images = {
        "kannada_printed": samples_dir / "sample_kannada_crop.png",
        "kannada_handwritten": samples_dir / "sample_handwritten_crop.png",
        "english_printed": samples_dir / "sample_english_printed.png",
        "english_handwritten": samples_dir / "sample_english_handwritten.png",
    }

    # Generate English printed if missing
    if not images["english_printed"].exists():
        img = Image.new("RGB", (600, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 32)
        except Exception:
            font = ImageFont.load_default()
        draw.text((20, 30), "SURVEY NO 142 VILLAGE RECORD", fill=(0, 0, 0), font=font)
        img.save(images["english_printed"])

    # Generate English handwritten if missing
    if not images["english_handwritten"].exists():
        img = Image.new("RGB", (600, 100), color=(252, 250, 242))
        draw = ImageDraw.Draw(img)
        hw_font = None
        for f in ["C:/Windows/Fonts/segoepr.ttf", "C:/Windows/Fonts/comic.ttf", "C:/Windows/Fonts/arial.ttf"]:
            if os.path.exists(f):
                try:
                    hw_font = ImageFont.truetype(f, 32)
                    break
                except Exception:
                    pass
        if not hw_font:
            hw_font = ImageFont.load_default()
        draw.text((20, 30), "Survey Number 142", fill=(15, 25, 60), font=hw_font)
        img.save(images["english_handwritten"])

    return images


class TestRealImagesMultimodalPipeline(unittest.TestCase):
    """Integration test suite executing real models on real image crops."""

    @classmethod
    def setUpClass(cls):
        cls.samples_dir = PROJECT_ROOT / "data" / "samples"
        cls.images = _ensure_sample_images(cls.samples_dir)
        cls.router = LanguageScriptRouter()
        cls.pipeline = DocumentProcessingPipeline(router=cls.router)

    def test_scenario_1_printed_kannada_real(self):
        """Verify real printed Kannada recognition via PaddleOCR."""
        img_path = self.images["kannada_printed"]
        self.assertTrue(img_path.exists(), f"Missing test image: {img_path}")

        res = self.router.route_and_recognize(
            image=str(img_path),
            language="kannada",
            is_handwritten=False,
        )
        self.assertEqual(res.model_name, "paddleocr-kannada")
        self.assertFalse(res.is_handwritten)
        self.assertIsNotNone(res.text)
        self.assertGreater(len(res.text.strip()), 0)
        self.assertIsNotNone(res.confidence)
        self.assertGreater(res.confidence, 0.0)

    def test_scenario_2_handwritten_kannada_real(self):
        """Verify real handwritten Kannada recognition via fine-tuned TrOCR checkpoint."""
        img_path = self.images["kannada_handwritten"]
        self.assertTrue(img_path.exists(), f"Missing test image: {img_path}")

        res = self.router.route_and_recognize(
            image=str(img_path),
            language="kannada",
            is_handwritten=True,
        )
        self.assertIn("kannada", res.model_name.lower())
        self.assertTrue(res.is_handwritten)
        self.assertIsNotNone(res.text)
        self.assertGreater(len(res.text.strip()), 0)
        self.assertIsNotNone(res.confidence)
        self.assertGreater(res.confidence, 0.0)

    def test_scenario_3_printed_english_real(self):
        """Verify real printed English recognition via PaddleOCR English."""
        img_path = self.images["english_printed"]
        self.assertTrue(img_path.exists(), f"Missing test image: {img_path}")

        res = self.router.route_and_recognize(
            image=str(img_path),
            language="english",
            is_handwritten=False,
        )
        self.assertEqual(res.model_name, "paddleocr-english")
        self.assertFalse(res.is_handwritten)
        self.assertIn("SURVEY NO", res.text.upper())
        self.assertIsNotNone(res.confidence)
        self.assertGreater(res.confidence, 0.5)

    def test_scenario_4_handwritten_english_real(self):
        """Verify real handwritten English recognition via pretrained TrOCR English."""
        img_path = self.images["english_handwritten"]
        self.assertTrue(img_path.exists(), f"Missing test image: {img_path}")

        res = self.router.route_and_recognize(
            image=str(img_path),
            language="english",
            is_handwritten=True,
        )
        self.assertEqual(res.model_name, "microsoft/trocr-small-handwritten")
        self.assertTrue(res.is_handwritten)
        self.assertIsNotNone(res.text)
        self.assertGreater(len(res.text.strip()), 0)
        self.assertIsNotNone(res.confidence)
        self.assertGreater(res.confidence, 0.0)

    def test_e2e_document_pipeline_all_modalities(self):
        """Verify DocumentProcessingPipeline executes across multiple regions."""
        doc_img = Image.new("RGB", (800, 600), color=(255, 255, 255))
        regions = [
            {"bbox": (10, 10, 500, 100), "language": "english", "is_handwritten": False},
            {"bbox": (10, 150, 500, 250), "language": "english", "is_handwritten": True},
        ]
        response = self.pipeline.process_document(
            image=doc_img,
            regions=regions,
            document_id="doc_multimodal_real_001",
        )
        self.assertEqual(len(response.ordered_regions), 2)
        self.assertEqual(response.ordered_regions[0].model_name, "paddleocr-english")
        self.assertEqual(response.ordered_regions[1].model_name, "microsoft/trocr-small-handwritten")


if __name__ == "__main__":
    unittest.main()
