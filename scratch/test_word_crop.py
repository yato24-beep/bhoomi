import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
sys.path.insert(0, r"c:\Land Record")
import src
import torch
from PIL import Image
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaTokenizer

base_ckpt = r"c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint"
ft_ckpt = r"c:\Land Record\models\trocr\personal_trial_checkpoints\best_checkpoint"

tok = XLMRobertaTokenizer.from_pretrained(base_ckpt)
proc = TrOCRProcessor.from_pretrained(base_ckpt)
model_base = VisionEncoderDecoderModel.from_pretrained(base_ckpt).cuda().eval()
model_ft = VisionEncoderDecoderModel.from_pretrained(ft_ckpt).cuda().eval()

rot90 = Image.open(r"c:\Land Record\scratch\new_kothi_rot90.png")
# Crop both components with 25px margin
x1, y1 = 574 - 25, 121 - 25
x2, y2 = 729 + 25, 236 + 25
word_crop = rot90.crop((x1, y1, x2, y2))
word_crop.save(r"c:\Land Record\scratch\kothi_word_tight.png")
print("Word crop size:", word_crop.size)

for rot in [0, 90, 180, 270]:
    c = word_crop.rotate(rot, expand=True)
    px = proc(c.convert("RGB"), return_tensors="pt").pixel_values.cuda()
    with torch.no_grad():
        out_b = model_base.generate(px, max_new_tokens=32)
        out_f = model_ft.generate(px, max_new_tokens=32)
        dec_b = tok.decode(out_b[0], skip_special_tokens=True).strip()
        dec_f = tok.decode(out_f[0], skip_special_tokens=True).strip()
    print(f"Rot {rot:3d}: Base='{dec_b}', FT='{dec_f}'")
