import os, sys, site

# Fix torch DLL loading on Windows
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    tlib = os.path.join(sp, 'torch', 'lib')
    if os.path.isdir(tlib):
        try:
            os.add_dll_directory(tlib)
        except Exception:
            pass
        os.environ['PATH'] = tlib + ';' + os.environ.get('PATH', '')

try:
    import torch
except Exception as e:
    print('Warning: torch load:', e)

from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

p1 = r'c:\Land Record\storage\uploads\48afaad54543e31e_doc1.jpeg'
img = Image.open(p1)
print(f'Original doc1 image size: {img.size}', flush=True)

ocr_ka = PaddleOCR(lang='ka', use_angle_cls=True)

for rot in [0, 90, 180, 270]:
    rot_img = img.rotate(rot, expand=True) if rot != 0 else img
    arr = np.array(rot_img.convert('RGB'))
    res = ocr_ka.ocr(arr)
    lines = []
    if res and res[0]:
        for item in res[0]:
            lines.append((item[1][0], round(item[1][1], 3)))
    print(f'\n=== ROTATION {rot} deg (size {rot_img.size}) ===', flush=True)
    print(f'Total detected lines: {len(lines)}', flush=True)
    for t, c in lines[:10]:
        try:
            print(f'  [{c}] {t}', flush=True)
        except Exception:
            print(f'  [{c}] {repr(t)}', flush=True)
