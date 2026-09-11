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
print("Type of res:", type(res))
print("Type of res[0]:", type(res[0]))
if isinstance(res[0], dict):
    print("res[0] keys:", list(res[0].keys()))
    for k in res[0]:
        val = res[0][k]
        if isinstance(val, list):
            print(f"Key '{k}' (list of length {len(val)}): sample item = {val[:2] if val else 'empty'}")
        else:
            print(f"Key '{k}': {type(val)} = {val}")
elif hasattr(res[0], "__dict__"):
    print("res[0] dict:", res[0].__dict__)
else:
    print("res[0] repr:", repr(res[0])[:500])
