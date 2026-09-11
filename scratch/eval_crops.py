import os
import sys
sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaTokenizer

base_ckpt = r"c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint"
ft_ckpt = r"c:\Land Record\models\trocr\personal_trial_checkpoints\best_checkpoint"

tok = XLMRobertaTokenizer.from_pretrained(base_ckpt)
proc = TrOCRProcessor.from_pretrained(base_ckpt)
model_base = VisionEncoderDecoderModel.from_pretrained(base_ckpt).cuda().eval()
model_ft = VisionEncoderDecoderModel.from_pretrained(ft_ckpt).cuda().eval()

for name in ["crop_left", "crop_right", "crop_all"]:
    p = os.path.join(r"c:\Land Record\scratch\kothi_debug", f"{name}.png")
    img = Image.open(p).convert("RGB")
    px = proc(img, return_tensors="pt").pixel_values.cuda()
    with torch.no_grad():
        out_b = model_base.generate(px, max_new_tokens=32)
        out_ft = model_ft.generate(px, max_new_tokens=32)
        dec_b = tok.decode(out_b[0], skip_special_tokens=True).strip()
        dec_ft = tok.decode(out_ft[0], skip_special_tokens=True).strip()
    print(f"{name:12s} ({img.size[0]}x{img.size[1]}) -> Base: '{dec_b}', FT: '{dec_ft}'")
