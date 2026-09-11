import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from src.preprocessing.image_enhancement import detect_and_correct_coarse_orientation, load_image_as_pil
from paddleocr import PaddleOCR

raw = Image.open("doc2.jpeg")
oriented, meta = detect_and_correct_coarse_orientation(raw)
print("Coarse orientation:", meta)
print("Oriented size:", oriented.size)

# Test 1: Baseline recognition on oriented
ocr_ka = PaddleOCR(use_angle_cls=True, lang="ka", enable_mkldnn=False)
arr_orig = np.array(oriented.convert("RGB"))
res_orig = ocr_ka.ocr(arr_orig, cls=True)
lines_orig = res_orig[0] if res_orig and res_orig[0] else []
print(f"Original boxes count: {len(lines_orig)}")
sample_orig = [line[1][0] for line in lines_orig[:10]]
print("Sample orig text:", sample_orig)

# Test 2: Upscaled 1.6x + CLAHE + unsharp mask
w, h = oriented.size
scale = 1.6
upscaled = oriented.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
gray = np.array(upscaled.convert("L"))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_arr = clahe.apply(gray)
enhanced_pil = Image.fromarray(clahe_arr).convert("RGB")
enhanced_pil = ImageEnhance.Sharpness(enhanced_pil).enhance(1.2)

arr_enh = np.array(enhanced_pil)
res_enh = ocr_ka.ocr(arr_enh, cls=True)
lines_enh = res_enh[0] if res_enh and res_enh[0] else []
print(f"Enhanced boxes count: {len(lines_enh)}")
sample_enh = [line[1][0] for line in lines_enh[:10]]
print("Sample enhanced text:", sample_enh)

# Print comparison of high confidence texts
print("\n--- Comparing recognized Kannada lines with conf > 0.7 ---")
for line in lines_enh:
    text, conf = line[1]
    if conf > 0.7:
        print(f"[{conf:.2f}] {text}")
