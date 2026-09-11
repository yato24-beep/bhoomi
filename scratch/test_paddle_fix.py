import os
import sys

# Disable MKLDNN/OneDNN by default in PaddleX and Paddle
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# IMPORTANT: Import torch first so its C++ runtimes and DLLs load properly on Windows
import torch
print("PyTorch loaded successfully:", torch.__version__)

from pathlib import Path
from PIL import Image
import numpy as np
import paddle
print("Paddle loaded:", paddle.__version__)

from paddleocr import PaddleOCR
print("Initializing PaddleOCR with lang='ka'...")
ocr = PaddleOCR(lang="ka")

doc2_path = Path("C:/Users/achyu/Downloads/doc2.jpeg")
if not doc2_path.exists():
    print("doc2.jpeg not found at:", doc2_path)
    sys.exit(1)

img = Image.open(doc2_path).convert("RGB")
arr_orig = np.array(img)
print("Input image shape:", arr_orig.shape)

print("Running OCR inference...")
try:
    res = ocr.ocr(arr_orig)
    print("Inference completed successfully!")
    print("Result structure type:", type(res), "length:", len(res) if res else 0)
    if res and res[0]:
        print(f"Total lines detected: {len(res[0])}")
        for i, line in enumerate(res[0][:15]):
            print(f"[{i+1}] Box: {line[0]}, Text: '{line[1][0]}', Conf: {line[1][1]:.4f}")
    else:
        print("Result was empty or None:", res)
except Exception as e:
    import traceback
    print("Error during OCR inference:")
    traceback.print_exc()
