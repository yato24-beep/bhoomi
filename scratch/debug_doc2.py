import sys
from pathlib import Path
from PIL import Image
import numpy as np

repo_root = Path("c:/Land Record")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

doc2_path = Path("C:/Users/achyu/Downloads/doc2.jpeg")
img = Image.open(doc2_path)
arr_orig = np.array(img.convert("RGB"))

from src.preprocessing.image_enhancement import preprocess_document_image
prep_res = preprocess_document_image(image_input=img, apply_thresholding=False)
prep_img = prep_res.image
arr_prep = np.array(prep_img.convert("RGB"))

from paddleocr import PaddleOCR

print("1. Running PaddleOCR(lang='ka').ocr(arr_orig)...", flush=True)
ocr_ka = PaddleOCR(lang="ka")
res_raw = ocr_ka.ocr(arr_orig)
print(f"Result raw count: {len(res_raw) if res_raw else 0}", flush=True)
print(f"Result raw repr: {res_raw}", flush=True)

print("2. Running PaddleOCR(lang='ka').ocr(arr_prep)...", flush=True)
res_prep = ocr_ka.ocr(arr_prep)
print(f"Result prep count: {len(res_prep) if res_prep else 0}", flush=True)
print(f"Result prep repr: {res_prep}", flush=True)

print("3. Running PaddleOCR(lang='en').ocr(arr_orig)...", flush=True)
ocr_en = PaddleOCR(lang="en")
res_en = ocr_en.ocr(arr_orig)
print(f"Result en count: {len(res_en) if res_en else 0}", flush=True)
print(f"Result en repr: {res_en}", flush=True)

print("4. Testing what paddleocr.predict() returns...", flush=True)
pred_ka = ocr_ka.predict(arr_orig)
print(f"Predict ka type: {type(pred_ka)}, len: {len(pred_ka) if pred_ka else 0}", flush=True)
for item in (pred_ka or []):
    print("  item keys:", item.keys() if isinstance(item, dict) else dir(item), flush=True)
    if hasattr(item, "keys"):
        print("  rec_text:", item.get("rec_text"), "rec_score:", item.get("rec_score"))
