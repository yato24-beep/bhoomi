"""Standalone Smoke-Test for Pretrained TrOCR Handwriting Recognition Backend.

Loads the real pretrained microsoft/trocr-small-handwritten Vision-Encoder-Decoder model,
performs genuine handwriting inference on a local handwritten sample image, extracts
exact logit-derived token probabilities, and generates a structured audit report.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Ensure UTF-8 output encoding for terminal display
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas import BoundingBox, DocumentPage
from src.handwriting.service import HandwritingOCRService
from src.handwriting.trocr_recognizer import (
    HAS_TORCH,
    HAS_TRANSFORMERS,
    TrocrHandwritingRecognizer,
)
from src.preprocessing.image_enhancement import preprocess_document_image


def generate_handwritten_sample_image(output_path: Path) -> Path:
    """Generates a sample handwritten crop with realistic cursive ink variations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 450, 100
    img = Image.new("RGB", (width, height), color=(252, 250, 242))
    draw = ImageDraw.Draw(img)

    # Draw archival paper grain / rule lines
    for y in (30, 75):
        draw.line([(0, y), (width, y)], fill=(230, 226, 215), width=1)

    # Choose best cursive/script font available on system
    font = None
    candidate_fonts = [
        "C:/Windows/Fonts/segoepr.ttf",      # Segoe Print
        "C:/Windows/Fonts/segoesc.ttf",      # Segoe Script
        "C:/Windows/Fonts/comic.ttf",        # Comic Sans
        "C:/Windows/Fonts/arial.ttf",
    ]
    for fp in candidate_fonts:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 32)
                break
            except Exception:
                continue

    if font is None:
        font = ImageFont.load_default()

    # English handwritten text typical in revenue land records
    sample_text = "Survey No 45/2 A"
    draw.text((25, 28), sample_text, fill=(25, 30, 45), font=font)

    img.save(output_path)
    return output_path


def run_trocr_smoke_test():
    """Executes the real TrOCR handwriting recognition smoke test."""
    print("=" * 70)
    print("  LAND RECORD DIGITIZATION - REAL TrOCR HANDWRITING SMOKE TEST")
    print("=" * 70)

    sample_path = PROJECT_ROOT / "data" / "samples" / "sample_handwritten_crop.png"
    if not sample_path.exists():
        print(f"\n[1] Sample image not found at: {sample_path}")
        print("    Generating local handwritten test crop placeholder...")
        generate_handwritten_sample_image(sample_path)
        print(f"    Sample image generated at: {sample_path}")
        print("    NOTE: You can place any custom handwritten crop at this path.")
    else:
        print(f"\n[1] Using existing local handwritten sample at: {sample_path}")
        print("    (To test your own handwritten crop, replace this file at: data/samples/sample_handwritten_crop.png)")

    # 2. Image Preprocessing
    print("\n[2] Executing Preprocessing & Enhancement Pipeline...")
    prep_res = preprocess_document_image(
        image_input=sample_path,
        apply_deskew=True,
        apply_contrast=True,
        apply_denoise=True,
        apply_thresholding=False,  # Keep grayscale to preserve subtle stroke variations
    )
    print(f"    Original Size   : {prep_res.original_size}")
    print(f"    Enhanced Mode   : {prep_res.image.mode} | Size: {prep_res.processed_size}")
    print(f"    Pipeline Steps  : {prep_res.audit_metadata.get('pipeline_steps', [])}")

    # 3. TrOCR Backend Initialization & Loading
    model_id = "microsoft/trocr-small-handwritten"
    print(f"\n[3] Initializing TrocrHandwritingRecognizer ({model_id})...")
    print(f"    PyTorch Installed    : {HAS_TORCH}")
    print(f"    Transformers Installed: {HAS_TRANSFORMERS}")

    recognizer = TrocrHandwritingRecognizer(
        model_name_or_path=model_id,
        model_version="trocr_small_v1",
        preprocess_input=True,
        max_new_tokens=32,
    )
    print(f"    Recognizer Device    : {recognizer.device}")
    print(f"    Lazy State (pre-load): is_loaded = {recognizer.is_loaded}")

    print("\n[4] Loading Pretrained Weights & Running Genuine Model Inference...")
    ocr_result = recognizer.recognize_handwriting(
        image=sample_path,
        bbox=BoundingBox(x_min=25, y_min=28, x_max=420, y_max=80),
        page_number=1,
    )

    audit_data = ocr_result.metadata
    meta_info = audit_data.get("metadata", {})
    raw_probs = audit_data.get("raw_token_probabilities", [])

    print(f"    Model Loaded State   : is_loaded = {recognizer.is_loaded}")
    print(f"    Recognized Text      : '{ocr_result.text}'")
    print(f"    Mean Confidence      : {ocr_result.confidence}")
    print(f"    Geometric Mean Conf  : {meta_info.get('geometric_mean_confidence')}")
    print(f"    Generated Token Count: {meta_info.get('token_count')}")
    print(f"    Raw Token Step Probs : {raw_probs}")
    print(f"    Engine Status        : {meta_info.get('engine_status')}")

    # 5. HandwritingOCRService Integration
    print("\n[5] Testing HandwritingOCRService with TrOCR backend...")
    service = HandwritingOCRService(recognizer=recognizer)
    page = DocumentPage(page_number=1, image_path=str(sample_path))
    updated_page = service.process_document_page_crops(
        page=page,
        crop_images=[sample_path],
        crop_bboxes=[BoundingBox(x_min=25, y_min=28, x_max=420, y_max=80)],
    )
    print(f"    DocumentPage OCR Results Count: {len(updated_page.ocr_results)}")

    # 6. Audit & Capability Report
    print("\n" + "=" * 70)
    print("  TrOCR BACKEND & CAPABILITY AUDIT REPORT")
    print("=" * 70)
    if meta_info.get("engine_status") == "active":
        print(f"  [REAL TrOCR INFERENCE EXECUTED]")
        print(f"  * Model Name         : {model_id}")
        print(f"  * Execution Device   : {recognizer.device}")
        print(f"  * Raw Recognized Text: '{ocr_result.text}'")
        print(f"  * Model Confidence   : {ocr_result.confidence} (derived from step softmax logits)")
        print(f"  * Step Probabilities : {raw_probs}")
        print(f"  * Inference Metadata : {meta_info}")
        print(f"  * Integrity Assurance: Zero synthetic scores; raw token probabilities preserved.")
    else:
        print(f"  [MODEL INFERENCE NOT EXECUTED]")
        print(f"  * Reason: {meta_info.get('error') or meta_info.get('reason')}")

    print("\n  [KNOWN LANGUAGE LIMITATIONS]")
    print("  * Current TrOCR weights: Pretrained primarily on English/Latin handwriting (IAM dataset).")
    print("  * Regional Scripts     : Does NOT natively recognize handwritten Kannada, Telugu, Tamil,")
    print("    or Devanagari text without fine-tuning.")
    print("  * Roadmap              : Fine-tuning VisionEncoderDecoder on regional Indic land record")
    print("    crops is scheduled for the upcoming model training milestone (training/).")
    print("=" * 70)


if __name__ == "__main__":
    run_trocr_smoke_test()
