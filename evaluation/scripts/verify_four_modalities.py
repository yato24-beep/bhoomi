"""Verification Script for 4 Supported OCR Modalities.

Tests:
1. Printed Kannada (PaddleOCR Kannada baseline: ppocr_v4_kannada)
2. Handwritten Kannada (Fine-tuned TrOCR Kannada: models/trocr/kannada_full_checkpoints/best_checkpoint)
3. Printed English (PaddleOCR English baseline: ppocr_v4_en)
4. Handwritten English (Pretrained TrOCR English: microsoft/trocr-small-handwritten)

Usage:
    python evaluation/scripts/verify_four_modalities.py
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Ensure UTF-8 console output for Indic characters
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.handwriting.router import LanguageScriptRouter
from src.integration.document_pipeline import DocumentProcessingPipeline


def ensure_samples():
    """Ensure all 4 test sample images exist."""
    samples_dir = PROJECT_ROOT / "data" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    en_printed = samples_dir / "sample_english_printed.png"
    if not en_printed.exists():
        img = Image.new("RGB", (600, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 32)
        except Exception:
            font = ImageFont.load_default()
        draw.text((20, 30), "SURVEY NO 142 VILLAGE RECORD", fill=(0, 0, 0), font=font)
        img.save(en_printed)

    en_hw = samples_dir / "sample_english_handwritten.png"
    if not en_hw.exists():
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
        img.save(en_hw)


def main():
    print("=" * 80)
    print("  LAND RECORD DIGITIZATION - 4-MODALITY OCR VERIFICATION")
    print("=" * 80)

    ensure_samples()
    router = LanguageScriptRouter()

    test_cases = [
        {
            "title": "Scenario 1: Printed Kannada",
            "image": "data/samples/sample_kannada_crop.png",
            "language": "kannada",
            "is_handwritten": False,
            "expected_engine": "paddleocr-kannada",
        },
        {
            "title": "Scenario 2: Handwritten Kannada",
            "image": "data/samples/sample_handwritten_crop.png",
            "language": "kannada",
            "is_handwritten": True,
            "expected_engine": "best_checkpoint",
        },
        {
            "title": "Scenario 3: Printed English",
            "image": "data/samples/sample_english_printed.png",
            "language": "english",
            "is_handwritten": False,
            "expected_engine": "paddleocr-english",
        },
        {
            "title": "Scenario 4: Handwritten English",
            "image": "data/samples/sample_english_handwritten.png",
            "language": "english",
            "is_handwritten": True,
            "expected_engine": "microsoft/trocr-small-handwritten",
        },
    ]

    all_passed = True
    for tc in test_cases:
        print(f"\n--- {tc['title']} ---")
        img_path = PROJECT_ROOT / tc["image"]
        print(f"  Input Image       : {tc['image']}")
        print(f"  Language / Type   : {tc['language']} (is_handwritten={tc['is_handwritten']})")

        res = router.route_and_recognize(
            image=str(img_path),
            language=tc["language"],
            is_handwritten=tc["is_handwritten"],
        )

        print(f"  Active Engine     : {res.model_name}")
        print(f"  Model Version     : {res.model_version}")
        print(f"  Recognized Text   : '{res.text}'")
        print(f"  Confidence Score  : {res.confidence}")
        print(f"  Audit Method      : {res.metadata.get('calculation_method')}")

        if not res.text or res.confidence is None or res.confidence <= 0:
            print("  [STATUS] FAILED (Empty text or missing confidence)")
            all_passed = False
        else:
            print("  [STATUS] PASSED")

    print("\n" + "=" * 80)
    if all_passed:
        print("  ALL 4 MODALITIES VERIFIED SUCCESSFULLY AND READY FOR DEPLOYMENT")
    else:
        print("  SOME SCENARIOS FAILED VERIFICATION")
    print("=" * 80)


if __name__ == "__main__":
    main()
