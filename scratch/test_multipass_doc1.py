import os
import sys

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PIL import Image, ImageEnhance, ImageFilter

repo_root = r"c:\Land Record"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.preprocessing.line_segmentation import DocumentLineSegmenter
from src.handwriting.trocr_recognizer import TrocrHandwritingRecognizer

doc1_path = r"c:\Land Record\doc1.jpeg"
img = Image.open(doc1_path).rotate(90, expand=True)

segmenter = DocumentLineSegmenter()
lines = segmenter.segment_into_lines(img)
print(f"Segmented {len(lines)} lines", flush=True)

trocr = TrocrHandwritingRecognizer(device="cuda")
trocr.load_model()
print("TrOCR loaded", flush=True)

def multipass_recognize(trocr_model, crop):
    # Variant A: original
    res_a = trocr_model.recognize_handwriting(crop)
    best_res = res_a
    best_conf = res_a.confidence or 0.0

    # Variant B: 1.5x upscale
    crop_b = crop.resize((int(crop.width * 1.5), int(crop.height * 1.5)), Image.BICUBIC)
    res_b = trocr_model.recognize_handwriting(crop_b)
    if (res_b.confidence or 0.0) > best_conf:
        best_res, best_conf = res_b, res_b.confidence or 0.0

    # Variant C: contrast boost
    crop_c = ImageEnhance.Contrast(crop).enhance(1.3)
    res_c = trocr_model.recognize_handwriting(crop_c)
    if (res_c.confidence or 0.0) > best_conf:
        best_res, best_conf = res_c, res_c.confidence or 0.0

    # Variant D: light denoise
    crop_d = crop.filter(ImageFilter.MedianFilter(size=3))
    res_d = trocr_model.recognize_handwriting(crop_d)
    if (res_d.confidence or 0.0) > best_conf:
        best_res, best_conf = res_c, res_c.confidence or 0.0

    return best_res

out_file = r"c:\Land Record\scratch\multipass_output.txt"
with open(out_file, "w", encoding="utf-8") as f:
    f.write("=== MULTIPASS RECOGNITION OUTPUT (doc1.jpeg) ===\n")
    for idx, l in enumerate(lines[:15]):
        res = multipass_recognize(trocr, l.image_crop)
        conf = res.confidence or 0.0
        line_str = f"Line {l.line_id} (col={l.column_index}, order={l.reading_order_index}): conf={conf:.3f} | Text: {res.text}\n"
        f.write(line_str)
        f.flush()
        print(f"Recognized {l.line_id}: conf={conf:.3f}", flush=True)

print("Saved output to", out_file, flush=True)
