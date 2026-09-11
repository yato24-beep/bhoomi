import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import src

import cv2
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, XLMRobertaTokenizer
from paddleocr import PaddleOCR

raw_path = r"C:\Users\achyu\Downloads\i.jpeg"
im_raw = Image.open(raw_path)
print(f"Original i.jpeg size: {im_raw.size}, mode={im_raw.mode}")

# Rotate 90 CCW to make text horizontal
rot = im_raw.rotate(90, expand=True)
rot.save(r"c:\Land Record\scratch\i_rot90.png")
print(f"Rotated 90 CCW size: {rot.size}")

# Detect bounding boxes with PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
res = ocr.ocr(np.array(rot.convert("RGB")), cls=False, rec=True)
lines = res[0] if res and res[0] else []
print(f"\nDetected {len(lines)} bounding boxes in i_rot90.png:")
for idx, l in enumerate(lines):
    bbox, (text, conf) = l
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    print(f"  [{idx}] bbox=[{min(xs):.1f}, {min(ys):.1f} -> {max(xs):.1f}, {max(ys):.1f}] text={repr(text)} conf={conf:.3f}")

# Look at box 1: bbox=[587.0, 165.0 -> 1087.0, 311.0], text='aop3Monhay'
# 'ಕೋತಿ' is the Kannada word on the left of '- Monkey'
# In this line: x goes from 580 to 1100, y goes from 160 to 310
# 'ಕೋತಿ' is approximately x=[580..810], y=[160..300]

# Let's save several candidate crops of the Kannada word:
crop_tight = rot.crop((585, 160, 810, 305))
crop_tight.save(r"c:\Land Record\scratch\i_kannada_tight.png")
print(f"\nSaved tight Kannada crop: scratch/i_kannada_tight.png size={crop_tight.size}")

crop_standard = rot.crop((580, 150, 830, 310))
crop_standard.save(r"c:\Land Record\scratch\i_kannada_standard.png")
print(f"Saved standard Kannada crop: scratch/i_kannada_standard.png size={crop_standard.size}")

# Run inference on Base model and Fine-Tuned model
BASE_CKPT = r"c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint"
FT_CKPT = r"c:\Land Record\models\trocr\personal_trial_checkpoints\best_checkpoint"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nLoading models on {device}...")
tok = XLMRobertaTokenizer.from_pretrained(BASE_CKPT)
proc = TrOCRProcessor.from_pretrained(BASE_CKPT)

model_base = VisionEncoderDecoderModel.from_pretrained(BASE_CKPT).to(device)
model_base.eval()

model_ft = VisionEncoderDecoderModel.from_pretrained(FT_CKPT).to(device)
model_ft.eval()

def run_inf(model, crop_img):
    pixel_values = proc(crop_img.convert("RGB"), return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        out = model.generate(
            pixel_values,
            max_new_tokens=32,
            return_dict_in_generate=True,
            output_scores=True,
        )
    gen_ids = out.sequences[0]
    txt = tok.decode(gen_ids, skip_special_tokens=True).strip()
    
    if out.scores:
        token_probs = []
        for step_idx, step_logits in enumerate(out.scores):
            tok_id = gen_ids[step_idx + 1] if step_idx + 1 < len(gen_ids) else None
            if tok_id is not None:
                probs = torch.softmax(step_logits[0], dim=-1)
                token_probs.append(probs[tok_id].item())
        conf = float(sum(token_probs) / len(token_probs)) if token_probs else 0.50
    else:
        conf = 0.50
    return txt, round(conf, 4)

for crop_name, c_img in [("Tight Crop", crop_tight), ("Standard Crop", crop_standard)]:
    b_txt, b_conf = run_inf(model_base, c_img)
    ft_txt, ft_conf = run_inf(model_ft, c_img)
    print(f"\n=== Evaluation on {crop_name} ({c_img.size}) ===")
    print(f"  Base Model Prediction:       '{b_txt}' (confidence: {b_conf*100:.1f}%)")
    print(f"  Fine-Tuned Model Prediction: '{ft_txt}' (confidence: {ft_conf*100:.1f}%)")
