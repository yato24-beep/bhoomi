import sys
sys.path.insert(0, ".")
import src
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from paddleocr import PaddleOCR
from src.preprocessing.image_enhancement import detect_and_correct_coarse_orientation

raw = Image.open("doc2.jpeg")
oriented, _ = detect_and_correct_coarse_orientation(raw)

ocr_ka = PaddleOCR(use_angle_cls=True, lang="ka", enable_mkldnn=False)
ocr_en = PaddleOCR(use_angle_cls=False, lang="en", enable_mkldnn=False)

# Detect boxes
arr = np.array(oriented.convert("RGB"))
res = ocr_ka.ocr(arr, rec=False)
boxes = res[0] if res and res[0] else []

print(f"Total detected boxes: {len(boxes)}")

def multi_pass_recognize(crop_pil):
    passes = []
    
    # Pass 1: standard
    p1 = crop_pil.convert("RGB")
    passes.append(("p1_std", p1))
    
    # Pass 2: upscaled 2x Lanczos
    w, h = crop_pil.size
    p2 = crop_pil.resize((max(w * 2, 64), max(h * 2, 32)), Image.Resampling.LANCZOS).convert("RGB")
    passes.append(("p2_up2x", p2))
    
    # Pass 3: CLAHE on upscaled
    gray = np.array(p2.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    p3 = Image.fromarray(clahe.apply(gray)).convert("RGB")
    passes.append(("p3_clahe", p3))
    
    # Pass 4: Sharpness enhanced
    p4 = ImageEnhance.Sharpness(p2).enhance(1.4)
    passes.append(("p4_sharp", p4))
    
    best_text = ""
    best_conf = 0.0
    best_pass = "none"
    best_engine = "paddleocr-kannada"
    
    for pname, pimg in passes:
        parr = np.array(pimg)
        # Try Kannada
        ka_out = ocr_ka.ocr(parr, det=False, cls=True)
        if ka_out and ka_out[0] and ka_out[0][0]:
            t, c = ka_out[0][0]
            if c > best_conf:
                best_conf = c
                best_text = t
                best_pass = pname
                best_engine = "paddleocr-kannada"
        
        # Try English
        en_out = ocr_en.ocr(parr, det=False, cls=False)
        if en_out and en_out[0] and en_out[0][0]:
            t_en, c_en = en_out[0][0]
            # English preference if confidence is clearly higher or matches english words/numbers
            if c_en > best_conf + 0.1 or (any(char.isdigit() for char in t_en) and c_en > 0.6):
                best_conf = c_en
                best_text = t_en
                best_pass = f"{pname}_en"
                best_engine = "paddleocr-english"
                
    return best_text, best_conf, best_pass, best_engine

print("\nRunning multi-pass on first 25 boxes:")
for i, b in enumerate(boxes[:25]):
    xs = [pt[0] for pt in b]
    ys = [pt[1] for pt in b]
    x_min, x_max = max(0, int(min(xs))), min(oriented.size[0], int(max(xs)))
    y_min, y_max = max(0, int(min(ys))), min(oriented.size[1], int(max(ys)))
    if x_max <= x_min or y_max <= y_min:
        continue
    crop = oriented.crop((x_min, y_min, x_max, y_max))
    text, conf, pused, engine = multi_pass_recognize(crop)
    if text:
        print(f"Box {i:02d} [{conf:.2f}] ({pused}) [{engine}]: {ascii(text)}")
