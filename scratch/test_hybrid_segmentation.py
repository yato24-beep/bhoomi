"""
Test Hybrid Line Segmentation, Routing and Batched TrOCR
"""
import os
import sys
import time
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEN_CKPT = r"c:\Land Record\models\trocr\kannada_generalized_checkpoints\best_checkpoint"

def test_pipeline(img_path):
    print("\n" + "="*80)
    print(f"TESTING HYBRID OCR PIPELINE ON: {img_path}")
    print("="*80)
    
    t0 = time.perf_counter()
    img = Image.open(img_path)
    
    # 1. Coarse orientation: rotate 90 CCW if portrait
    if img.height > img.width:
        oriented = img.rotate(90, expand=True)
    else:
        oriented = img
    print(f"Oriented size: {oriented.size} in {time.perf_counter() - t0:.2f}s")
    
    # 2. Run PaddleOCR with rec=True (finds all bounding boxes AND recognizes printed text in 1 shot)
    t1 = time.perf_counter()
    ocr = PaddleOCR(use_angle_cls=True, lang="ka", enable_mkldnn=False, show_log=False)
    paddle_res = ocr.ocr(np.array(oriented.convert("RGB")), cls=True)
    print(f"Paddle detection & recognition took: {time.perf_counter() - t1:.2f}s")
    
    boxes = paddle_res[0] if paddle_res and paddle_res[0] else []
    print(f"Total regions discovered: {len(boxes)}")
    
    # Separate into printed high-confidence vs handwritten / candidate for TrOCR
    printed_lines = []
    trocr_crops = []
    trocr_indices = []
    
    for idx, item in enumerate(boxes):
        box_pts = item[0]
        text, conf = item[1][0], item[1][1]
        
        xs = [pt[0] for pt in box_pts]
        ys = [pt[1] for pt in box_pts]
        x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
        x2, y2 = min(oriented.width, int(max(xs))), min(oriented.height, int(max(ys)))
        
        crop = oriented.crop((x1, y1, x2, y2))
        
        # Check if text is clearly printed English or confident printed Kannada
        is_english = any(c.isascii() and c.isalpha() for c in text)
        if is_english and conf > 0.50:
            printed_lines.append((idx, text, conf, "paddle_english", box_pts))
        elif conf > 0.85:
            printed_lines.append((idx, text, conf, "paddle_printed_kannada", box_pts))
        else:
            trocr_crops.append(crop)
            trocr_indices.append((idx, text, conf, box_pts))
            
    print(f"  • Confident Printed Lines: {len(printed_lines)}")
    print(f"  • Handwritten / Candidate Lines for TrOCR: {len(trocr_crops)}")
    
    # 3. Batched TrOCR on the remaining crops
    t2 = time.perf_counter()
    if trocr_crops:
        processor = TrOCRProcessor.from_pretrained(GEN_CKPT)
        model = VisionEncoderDecoderModel.from_pretrained(GEN_CKPT).to(DEVICE)
        model.eval()
        
        # Batch inference in chunks of 16
        BATCH_SIZE = 16
        trocr_results = []
        for b_start in range(0, len(trocr_crops), BATCH_SIZE):
            batch_imgs = [im.convert("RGB") for im in trocr_crops[b_start : b_start + BATCH_SIZE]]
            pixels = processor(images=batch_imgs, return_tensors="pt").pixel_values.to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(pixels, max_length=64, num_beams=5)
            preds = processor.batch_decode(outputs, skip_special_tokens=True)
            trocr_results.extend(preds)
            
        print(f"Batched TrOCR on {len(trocr_crops)} crops took: {time.perf_counter() - t2:.2f}s!")
        
    total_time = time.perf_counter() - t0
    print(f"\n>>> Total End-to-End Processing Time: {total_time:.2f}s <<< (previously 76.63s!)")
    
    print("\n--- Sample Recognized Lines ---")
    for idx, text, conf, eng, _ in printed_lines[:5]:
        print(f"  [Printed - {eng}] conf={conf:.2f}: '{text}'")
    if trocr_crops:
        for (idx, p_text, p_conf, _), t_text in zip(trocr_indices[:8], trocr_results[:8]):
            print(f"  [Handwritten - TrOCR] pred='{t_text}' (Paddle was: '{p_text}', conf={p_conf:.2f})")

if __name__ == "__main__":
    test_pipeline("doc1.jpeg")
