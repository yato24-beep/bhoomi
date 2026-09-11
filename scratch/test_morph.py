import os
import sys
import time
from PIL import Image

repo_root = r"c:\Land Record"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.preprocessing.line_segmentation import DocumentLineSegmenter

img = Image.open(r"c:\Land Record\doc1.jpeg").rotate(90, expand=True)
start = time.perf_counter()
segmenter = DocumentLineSegmenter(use_detector_fallback=False)
lines = segmenter.segment_into_lines(img)
elapsed = time.perf_counter() - start
print(f"Morphology segmentation: {len(lines)} lines in {elapsed*1000:.1f}ms", flush=True)
for l in lines[:8]:
    b = l.bbox
    print(f"  {l.line_id} (col={l.column_index}): [{b.x_min},{b.y_min}->{b.x_max},{b.y_max}] crop={l.image_crop.size}", flush=True)
