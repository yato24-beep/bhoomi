import os
import sys

os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import site
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    tlib = os.path.join(sp, 'torch', 'lib')
    if os.path.isdir(tlib):
        try:
            os.add_dll_directory(tlib)
        except Exception:
            pass
        os.environ['PATH'] = tlib + ';' + os.environ.get('PATH', '')

repo_root = r"c:\Land Record"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from PIL import Image
import numpy as np

from src.handwriting.trocr_recognizer import TrocrHandwritingRecognizer
from src.handwriting.paddle_recognizer import PaddleKannadaRecognizer

doc1_path = r"c:\Land Record\storage\uploads\48afaad54543e31e_doc1.jpeg"
img = Image.open(doc1_path)

# Rotate 90 counter-clockwise
img_90 = img.rotate(90, expand=True)
print("Rotated 90 size:", img_90.size, flush=True)

# Run Paddle text detector to get bounding boxes of lines
paddle = PaddleKannadaRecognizer()
paddle_res = paddle.recognize_handwriting(img_90)
line_details = paddle_res.metadata.get("metadata", {}).get("line_details", [])
print(f"Detected {len(line_details)} line boxes", flush=True)

# Load TrOCR
trocr = TrocrHandwritingRecognizer(device="cuda")
trocr.load_model()
print("TrOCR loaded on cuda:", trocr.is_loaded, flush=True)

# Test TrOCR on first 10 line crops
for i, l in enumerate(line_details[:10]):
    bbox = l.get("bbox")
    x1, y1, x2, y2 = int(bbox["x_min"]), int(bbox["y_min"]), int(bbox["x_max"]), int(bbox["y_max"])
    pad = 4
    crop = img_90.crop((max(0, x1 - pad), max(0, y1 - pad), min(img_90.width, x2 + pad), min(img_90.height, y2 + pad)))
    trocr_res = trocr.recognize_handwriting(crop)
    p_conf = l.get("confidence") or 0.0
    t_conf = trocr_res.confidence or 0.0
    print(f"Line {i+1} [bbox: {x1},{y1}->{x2},{y2}]:", flush=True)
    print(f"  Paddle: conf={p_conf:.3f} | {repr(l.get('text'))}", flush=True)
    print(f"  TrOCR : conf={t_conf:.3f} | {repr(trocr_res.text)}", flush=True)
