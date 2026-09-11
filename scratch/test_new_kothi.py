import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, r"c:\Land Record")
import src
import torch
from PIL import Image, ImageOps
import numpy as np
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaTokenizer

base_ckpt = r"c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint"
ft_ckpt = r"c:\Land Record\models\trocr\personal_trial_checkpoints\best_checkpoint"

tok = XLMRobertaTokenizer.from_pretrained(base_ckpt)
proc = TrOCRProcessor.from_pretrained(base_ckpt)
model_base = VisionEncoderDecoderModel.from_pretrained(base_ckpt).cuda().eval()
model_ft = VisionEncoderDecoderModel.from_pretrained(ft_ckpt).cuda().eval()

rot90 = Image.open(r"c:\Land Record\scratch\new_kothi_rot90.png")
gray = cv2.cvtColor(np.array(rot90), cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
print("Connected components:", num_labels)
valid_boxes = []
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if 200 < area < 100000 and w < rot90.width * 0.8:
        valid_boxes.append((x, y, w, h))

if valid_boxes:
    min_x = max(0, min(b[0] for b in valid_boxes) - 20)
    min_y = max(0, min(b[1] for b in valid_boxes) - 20)
    max_x = min(rot90.width, max(b[0] + b[2] for b in valid_boxes) + 20)
    max_y = min(rot90.height, max(b[1] + b[3] for b in valid_boxes) + 20)
    
    tight_crop = rot90.crop((min_x, min_y, max_x, max_y))
    tight_crop.save(r"c:\Land Record\scratch\new_kothi_tight.png")
    print(f"Tight crop size: {tight_crop.size}, box=({min_x}, {min_y}, {max_x}, {max_y})")
    
    px = proc(tight_crop.convert("RGB"), return_tensors="pt").pixel_values.cuda()
    with torch.no_grad():
        out_base = model_base.generate(px, max_new_tokens=32, output_scores=True, return_dict_in_generate=True)
        dec_base = tok.decode(out_base.sequences[0], skip_special_tokens=True).strip()
        
        out_ft = model_ft.generate(px, max_new_tokens=32, output_scores=True, return_dict_in_generate=True)
        dec_ft = tok.decode(out_ft.sequences[0], skip_special_tokens=True).strip()
    print(f"Tight crop -> Base: '{dec_base}', FT: '{dec_ft}'")
