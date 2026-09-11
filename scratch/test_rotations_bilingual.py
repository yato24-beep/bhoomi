import os, sys, site, json

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
    pass

from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

p1 = r'c:\Land Record\storage\uploads\48afaad54543e31e_doc1.jpeg'
img = Image.open(p1)

ocr_ka = PaddleOCR(lang='ka', use_angle_cls=True)
ocr_en = PaddleOCR(lang='en', use_angle_cls=True)

results = {}

for rot in [90, 270]:
    rot_img = img.rotate(rot, expand=True)
    arr = np.array(rot_img.convert('RGB'))
    
    # Run Kannada OCR
    res_ka = ocr_ka.ocr(arr)
    lines_ka = []
    if res_ka and res_ka[0]:
        for item in res_ka[0]:
            lines_ka.append({'text': item[1][0], 'confidence': round(float(item[1][1]), 3), 'bbox': item[0]})
            
    # Run English OCR
    res_en = ocr_en.ocr(arr)
    lines_en = []
    if res_en and res_en[0]:
        for item in res_en[0]:
            lines_en.append({'text': item[1][0], 'confidence': round(float(item[1][1]), 3), 'bbox': item[0]})

    results[f'rot_{rot}'] = {
        'img_size': rot_img.size,
        'kannada_lines_count': len(lines_ka),
        'kannada_mean_conf': round(float(np.mean([x['confidence'] for x in lines_ka])), 3) if lines_ka else 0,
        'kannada_sample': lines_ka[:15],
        'english_lines_count': len(lines_en),
        'english_mean_conf': round(float(np.mean([x['confidence'] for x in lines_en])), 3) if lines_en else 0,
        'english_sample': lines_en[:15],
    }

with open(r'c:\Land Record\scratch\rotation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print('Rotation comparison saved successfully to scratch/rotation_results.json!')
