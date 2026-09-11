import os
import sys
sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaTokenizer

TRIAL_CKPT = r"c:\Land Record\models\trocr\personal_trial_checkpoints\best_checkpoint"
OUT_DIR = r"c:\Land Record\scratch\kothi_debug"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 80)
print("FOCUSED CHARACTER DIAGNOSTIC: 'ಕೋತಿ' vs 'ಕೋಲಿ'")
print("=" * 80)

# 1. Load exact checkpoint, tokenizer, and processor
tokenizer = XLMRobertaTokenizer.from_pretrained(TRIAL_CKPT)
processor = TrOCRProcessor.from_pretrained(TRIAL_CKPT)
model = VisionEncoderDecoderModel.from_pretrained(TRIAL_CKPT).cuda().eval()

# 2. Inspect Unicode code points and Token IDs
words_to_check = ["ಕೋತಿ", "ಕೋಲಿ", "ತಿ", "ಲಿ", "ಕೋ"]
print("\n[1] Tokenizer Encoding & Unicode Verification:")
print("-" * 70)
for w in words_to_check:
    cps = [f"U+{ord(c):04X} ({c})" for c in w]
    enc = tokenizer.encode(w, add_special_tokens=False)
    dec = tokenizer.decode(enc)
    tok_pieces = [tokenizer.decode([t]) for t in enc]
    print(f"Word: '{w}' -> CodePoints: {', '.join(cps)}")
    print(f"       Token IDs: {enc} -> Subword Pieces: {tok_pieces} -> Decoded: '{dec}'")

# 3. Load Corrected Crop for new_kothi.jpeg
crop_path = os.path.join(OUT_DIR, "after_tight_crop.png")
if not os.path.exists(crop_path):
    # fallback to crop_all if after_tight_crop not found
    crop_path = os.path.join(OUT_DIR, "crop_all.png")

print(f"\n[2] Loading Corrected Crop: {crop_path}")
crop_img = Image.open(crop_path).convert("RGB")
print(f"Crop Size: {crop_img.size}")

pixel_values = processor(crop_img, return_tensors="pt").pixel_values.cuda()

# 4. Step-by-Step Logits Analysis (Greedy Step Probabilities)
print("\n[3] Step-by-Step Logit Probabilities for new_kothi crop:")
print("-" * 70)
with torch.no_grad():
    encoder_outputs = model.encoder(pixel_values)
    
    input_ids = torch.tensor([[model.config.decoder_start_token_id]], device="cuda")
    
    step_records = []
    for step in range(6):
        out = model(pixel_values=pixel_values, decoder_input_ids=input_ids)
        logits = out.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)[0]
        
        # Top 10 tokens at this step
        topk_probs, topk_indices = torch.topk(probs, 10)
        
        step_info = []
        for p, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
            piece = tokenizer.decode([idx])
            step_info.append((idx, piece, p))
            
        step_records.append(step_info)
        
        best_id = topk_indices[0].item()
        input_ids = torch.cat([input_ids, torch.tensor([[best_id]], device="cuda")], dim=-1)
        if best_id == model.config.eos_token_id:
            break

for step_num, step_info in enumerate(step_records, 1):
    print(f"Step {step_num}:")
    for rank, (t_id, t_str, prob) in enumerate(step_info[:8], 1):
        print(f"   Rank {rank:2d}: Token {t_id:6d} | '{t_str:10s}' | Prob: {prob*100:6.2f}%")

# 5. Compare Decoding Strategies: Greedy vs Beam Search (beams=3, 5)
print("\n[4] Beam Search Comparisons:")
print("-" * 70)

strategies = [
    ("Greedy (num_beams=1)", {"num_beams": 1, "do_sample": False}),
    ("Beam Search (num_beams=3, num_return_sequences=3)", {"num_beams": 3, "num_return_sequences": 3, "do_sample": False}),
    ("Beam Search (num_beams=5, num_return_sequences=5)", {"num_beams": 5, "num_return_sequences": 5, "do_sample": False}),
    ("Beam Search (num_beams=5, length_penalty=1.2)", {"num_beams": 5, "length_penalty": 1.2, "num_return_sequences": 5, "do_sample": False}),
]

for name, params in strategies:
    print(f"\n--- {name} ---")
    with torch.no_grad():
        gen_out = model.generate(
            pixel_values,
            max_new_tokens=32,
            return_dict_in_generate=True,
            output_scores=True,
            **params
        )
    
    for seq_idx, seq in enumerate(gen_out.sequences):
        text = tokenizer.decode(seq, skip_special_tokens=True).strip()
        raw_ids = seq.tolist()
        print(f"  Seq {seq_idx + 1}: '{text}' | Token IDs: {raw_ids}")

# 6. Compare with Training Crop (crop_o_kothi.png)
train_crop_path = r"c:\Land Record\training\datasets\personal_trial\crops\crop_o_kothi.png"
print(f"\n[5] Comparing with Training Sample: {train_crop_path}")
train_img = Image.open(train_crop_path).convert("RGB")
px_train = processor(train_img, return_tensors="pt").pixel_values.cuda()

with torch.no_grad():
    gen_train_greedy = model.generate(px_train, max_new_tokens=32)
    gen_train_beam5 = model.generate(px_train, max_new_tokens=32, num_beams=5, num_return_sequences=5)

print(f"  Training Sample (Greedy): '{tokenizer.decode(gen_train_greedy[0], skip_special_tokens=True).strip()}'")
print("  Training Sample (Beam 5):")
for i, seq in enumerate(gen_train_beam5):
    print(f"    Seq {i+1}: '{tokenizer.decode(seq, skip_special_tokens=True).strip()}'")

# 7. Create Visual Side-by-Side Comparison Image
print("\n[6] Generating Visual Comparison Image...")
vis_w, vis_h = 1000, 500
vis_canvas = Image.new("RGB", (vis_w, vis_h), (248, 249, 252))
draw = ImageDraw.Draw(vis_canvas)

# Left Panel: new_kothi crop
draw.rectangle([30, 30, 480, 420], fill=(255, 255, 255), outline=(200, 205, 220), width=2)
draw.rectangle([30, 30, 480, 75], fill=(235, 240, 250))
draw.text((45, 40), "Unseen: new_kothi.jpeg Crop", fill=(20, 30, 50))
draw.text((45, 55), f"Size: {crop_img.size}, Pred: 'ಕೋಲಿ' (83.3%)", fill=(80, 90, 110))

# paste new_kothi image centered
scaled_new = crop_img.resize((380, int(380 * crop_img.height / crop_img.width)), Image.Resampling.BILINEAR)
new_y = 85 + (320 - scaled_new.height) // 2
vis_canvas.paste(scaled_new, (65, max(85, new_y)))

# Right Panel: training crop_o_kothi
draw.rectangle([520, 30, 970, 420], fill=(255, 255, 255), outline=(200, 205, 220), width=2)
draw.rectangle([520, 30, 970, 75], fill=(235, 240, 250))
draw.text((535, 40), "Training: crop_o_kothi.png", fill=(20, 30, 50))
draw.text((535, 55), f"Size: {train_img.size}, Pred: 'ಕೋಟಿ' / 'ಕೋತಿ'", fill=(80, 90, 110))

# paste training image centered
scaled_train = train_img.resize((380, int(380 * train_img.height / train_img.width)), Image.Resampling.BILINEAR)
train_y = 85 + (320 - scaled_train.height) // 2
vis_canvas.paste(scaled_train, (555, max(85, train_y)))

# Bottom text bar
draw.rectangle([30, 440, 970, 485], fill=(40, 50, 70))
draw.text((45, 455), "Stroke Analysis: Notice the glyph curvature difference in the 2nd character ('ತಿ' vs 'ಲಿ')", fill=(255, 255, 255))

vis_path = os.path.join(OUT_DIR, "model_input_comparison.png")
vis_canvas.save(vis_path)
print(f"Saved Visual Comparison to: {vis_path}")

print("\n[✓] Character Diagnostic Complete.")
