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

repo_root = Path("c:/Land Record")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.handwriting.paddle_recognizer import PaddleKannadaRecognizer
from src.integration.document_pipeline import DocumentProcessingPipeline
from src.integration.person_c_adapter import PersonCAdapter

doc2_path = Path("C:/Users/achyu/Downloads/doc2.jpeg")
img = Image.open(doc2_path)

print("Testing PaddleKannadaRecognizer directly...")
recognizer = PaddleKannadaRecognizer()
rec_res = recognizer.recognize_handwriting(img)
print("Text length:", len(rec_res.text))
print("Confidence:", rec_res.confidence)
print("Sample text:", rec_res.text[:200])
