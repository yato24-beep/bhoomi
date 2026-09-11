import os
import sys

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PIL import Image, ImageDraw

repo_root = r"c:\Land Record"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.preprocessing.line_segmentation import DocumentLineSegmenter

doc1_path = r"c:\Land Record\doc1.jpeg"
img = Image.open(doc1_path)

# Rotate 90 counter-clockwise so it is right-side up
img_upright = img.rotate(90, expand=True)
print(f"Upright image size: {img_upright.size}", flush=True)

segmenter = DocumentLineSegmenter()
lines = segmenter.segment_into_lines(img_upright)
print(f"Total lines segmented: {len(lines)}", flush=True)

# Draw bounding boxes on debug image
debug_img = img_upright.copy().convert("RGB")
draw = ImageDraw.Draw(debug_img)

for l in lines:
    b = l.bbox
    draw.rectangle([b.x_min, b.y_min, b.x_max, b.y_max], outline="red", width=2)
    draw.text((b.x_min, max(0, b.y_min - 12)), f"{l.reading_order_index}:C{l.column_index}", fill="blue")

os.makedirs(r"c:\Land Record\storage\debug", exist_ok=True)
out_path = r"c:\Land Record\storage\debug\doc1_segmented_lines.png"
debug_img.save(out_path)
print(f"Saved debug overlay to {out_path}", flush=True)

for l in lines[:10]:
    b = l.bbox
    print(f"Line {l.line_id} (col={l.column_index}, order={l.reading_order_index}): [{b.x_min},{b.y_min} -> {b.x_max},{b.y_max}] crop size: {l.image_crop.size}", flush=True)
