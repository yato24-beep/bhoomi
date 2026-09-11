import os
import sys

os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from pathlib import Path
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="ka")

doc2_path = Path("C:/Users/achyu/Downloads/doc2.jpeg")
img = Image.open(doc2_path).convert("RGB")
arr_orig = np.array(img)

res = ocr.ocr(arr_orig)
r = res[0]
texts = r["rec_texts"]
scores = r["rec_scores"]
boxes = r.get("rec_boxes", r.get("dt_polys", []))

output_lines = []
output_lines.append(f"TOTAL DETECTED LINES: {len(texts)}")
for i, (txt, score) in enumerate(zip(texts, scores)):
    box = boxes[i] if i < len(boxes) else None
    output_lines.append(f"Line {i+1:02d} [conf={score:.4f}]: {txt} | Box: {box}")

output_text = "\n".join(output_lines)
with open("c:/Land Record/scratch/doc2_kannada_ocr_output.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print("Saved output to c:/Land Record/scratch/doc2_kannada_ocr_output.txt")
print(f"Total lines: {len(texts)}, Mean confidence: {np.mean(scores):.4f}")
