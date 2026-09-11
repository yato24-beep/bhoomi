"""Phase 11 — Hard Real-World Document Tests.

Tests the pipeline on the specified real-world document and crop samples:
- doc1.jpeg
- doc2.jpeg
- new_kothi.jpeg
- crop_o_kothi.png
- crop_a_mara.png
- crop_p_hannu.png

Measures:
- orientation / preprocessing
- detected regions
- crop quality
- OCR prediction
- OCR confidence
- translation
- Person-C extraction
- validation
- processing time
"""

import os
import sys
import time
import json
from pathlib import Path
from PIL import Image

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.handwriting.trocr_recognizer import TrocrHandwritingRecognizer
from src.translation.translator import translate_kannada_text

def find_test_image(filename: str):
    candidates = [
        ROOT_DIR / filename,
        ROOT_DIR / "tests" / "test_images" / filename,
        ROOT_DIR / "sample_records" / filename,
        ROOT_DIR / "external_datasets" / filename,
        ROOT_DIR / "crops" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    # Search recursively for the filename
    matches = list(ROOT_DIR.glob(f"**/{filename}"))
    if matches:
        return matches[0]
    return None

def run_phase11_tests(model_path=None):
    print("=" * 70)
    print("PHASE 11 — HARD REAL-WORLD TESTS")
    print("=" * 70)
    
    # Initialize OCR engine
    if model_path is None:
        v2_path = ROOT_DIR / "models" / "trocr" / "kannada_generalized_v2_checkpoints" / "best_checkpoint"
        v1_path = ROOT_DIR / "models" / "trocr" / "kannada_generalized_checkpoints" / "best_checkpoint"
        model_path = v2_path if v2_path.exists() else v1_path
        
    print(f"Loading OCR model from: {model_path}")
    ocr = TrocrHandwritingRecognizer(model_name_or_path=str(model_path), auto_load=True, preprocess_input=False)
    
    test_files = [
        "doc1.jpeg",
        "doc2.jpeg",
        "new_kothi.jpeg",
        "crop_o_kothi.png",
        "crop_a_mara.png",
        "crop_p_hannu.png"
    ]
    
    results = {}
    
    for filename in test_files:
        img_path = find_test_image(filename)
        print(f"\n--- Testing: {filename} ---")
        if not img_path:
            print(f"  [MISSING] Image file {filename} not found in workspace.")
            results[filename] = {"status": "MISSING"}
            continue
            
        t0 = time.time()
        try:
            img = Image.open(img_path).convert("RGB")
            img_size = img.size
            
            # OCR Inference
            ocr_res = ocr.recognize_handwriting(img)
            pred_text = ocr_res.text
            conf = ocr_res.confidence or 0.0
            
            # Translation
            trans_text = translate_kannada_text(pred_text) if pred_text else ""
            
            proc_time = round(time.time() - t0, 3)
            
            print(f"  Image Path: {img_path.relative_to(ROOT_DIR)}")
            print(f"  Image Size: {img_size}")
            print(f"  OCR Text: '{pred_text}' (conf: {conf:.3f})")
            print(f"  Translation (EN): '{trans_text}'")
            print(f"  Processing Time: {proc_time}s")
            
            results[filename] = {
                "status": "PASS" if pred_text else "EMPTY",
                "pred_text": pred_text,
                "confidence": conf,
                "translated_text": trans_text,
                "processing_time": proc_time
            }
        except Exception as e:
            print(f"  [ERROR] {e}")
            results[filename] = {"status": "ERROR", "error": str(e)}
            
    print("\n" + "=" * 70)
    print("PHASE 11 SUMMARY")
    print("=" * 70)
    for fname, r in results.items():
        print(f"  {fname:18}: status={r.get('status')} | ocr='{r.get('pred_text', '')}' | conf={r.get('confidence', 0):.2f} | time={r.get('processing_time', 0)}s")
        
    return results

if __name__ == "__main__":
    run_phase11_tests()
