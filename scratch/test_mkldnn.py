import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_mkldnn"] = "0"

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch  # import torch first

repo_root = Path("c:/Land Record")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

doc2_path = Path("C:/Users/achyu/Downloads/doc2.jpeg")
img = Image.open(doc2_path)
arr_orig = np.array(img.convert("RGB"))

import paddle
paddle.set_flags({"FLAGS_use_mkldnn": False})

from paddleocr import PaddleOCR
ocr = PaddleOCR(lang="ka")

print("Calling ocr()...", flush=True)
try:
    res = ocr.ocr(arr_orig)
    print("SUCCESS! Output length:", len(res) if res else 0, flush=True)
    if res and res[0]:
        print("First 5 recognized lines:")
        for line in res[0][:5]:
            print("  ", line[1][0], f"({line[1][1]:.3f})")
except Exception as e:
    import traceback
    traceback.print_exc()
