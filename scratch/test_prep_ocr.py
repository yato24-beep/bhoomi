import os
import sys
from pathlib import Path

os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from PIL import Image
import numpy as np

repo_root = Path("c:/Land Record")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.preprocessing.image_enhancement import preprocess_document_image
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="ka")

doc2_path = Path("C:/Users/achyu/Downloads/doc2.jpeg")
img = Image.open(doc2_path)

# Test with preprocessed image
prep_res = preprocess_document_image(img, apply_thresholding=False)
arr_prep = np.array(prep_res.image.convert("RGB"))

print("Preprocessed image shape:", arr_prep.shape)
res_prep = ocr.ocr(arr_prep)
texts_prep = res_prep[0]["rec_texts"]
print("Preprocessed line count:", len(texts_prep))
if texts_prep:
    print("Preprocessed first 3 lines:", texts_prep[:3])
