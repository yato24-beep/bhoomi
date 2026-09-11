import os
import sys
sys.path.insert(0, r"c:\Land Record")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaTokenizer
from scripts.evaluate_generalization import GeneralizationEvaluator

CURRENT_CKPT = r"c:\Land Record\models\trocr\personal_trial_checkpoints\best_checkpoint"
TARGETED_CKPT = r"c:\Land Record\models\trocr\kothi_targeted_checkpoints\best_checkpoint"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 80, flush=True)
print("  TARGETED CANDIDATE VS CURRENT MODEL COMPARISON TEST", flush=True)
print("=" * 80, flush=True)

print("Loading models on device:", device, flush=True)
tok_curr = XLMRobertaTokenizer.from_pretrained(CURRENT_CKPT)
proc_curr = TrOCRProcessor.from_pretrained(CURRENT_CKPT)
model_curr = VisionEncoderDecoderModel.from_pretrained(CURRENT_CKPT).to(device).eval()

tok_targ = XLMRobertaTokenizer.from_pretrained(TARGETED_CKPT)
proc_targ = TrOCRProcessor.from_pretrained(TARGETED_CKPT)
model_targ = VisionEncoderDecoderModel.from_pretrained(TARGETED_CKPT).to(device).eval()

def run_beam_search_top5(model, processor, tokenizer, img: Image.Image, num_beams=5):
    pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        beam_outputs = model.generate(
            pixel_values,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            max_new_tokens=32,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
            decoder_start_token_id=tokenizer.bos_token_id or tokenizer.cls_token_id or 0,
        )
    
    seqs = beam_outputs.sequences
    scores = beam_outputs.sequences_scores
    if scores is not None:
        probs = torch.softmax(scores, dim=-1).tolist()
    else:
        probs = [1.0 / len(seqs)] * len(seqs)

    candidates = []
    for rank, (seq, prob) in enumerate(zip(seqs, probs), 1):
        txt = tokenizer.decode(seq, skip_special_tokens=True).strip()
        candidates.append({"rank": rank, "text": txt, "prob": prob})
    return candidates

# 1. Test Unseen: new_kothi.jpeg
print("\n" + "=" * 80, flush=True)
print("TEST 1: UNSEEN HANDWRITING (new_kothi.jpeg, GT: 'ಕೋತಿ')", flush=True)
print("=" * 80, flush=True)

gen_eval = GeneralizationEvaluator(base_checkpoint=CURRENT_CKPT, finetuned_checkpoint=TARGETED_CKPT)
_, new_kothi_crop, _ = gen_eval.preprocess_and_crop(r"C:\Users\achyu\Downloads\new_kothi.jpeg")
print(f"Crop dimensions: {new_kothi_crop.size}", flush=True)

cands_curr_new = run_beam_search_top5(model_curr, proc_curr, tok_curr, new_kothi_crop)
cands_targ_new = run_beam_search_top5(model_targ, proc_targ, tok_targ, new_kothi_crop)

print("\n[A] CURRENT MODEL (personal_trial_checkpoints):", flush=True)
for c in cands_curr_new:
    print(f"  Rank #{c['rank']}: '{c['text']}' (Score/Prob: {c['prob']*100:.2f}%)", flush=True)

print("\n[B] TARGETED CANDIDATE MODEL (kothi_targeted_checkpoints):", flush=True)
for c in cands_targ_new:
    print(f"  Rank #{c['rank']}: '{c['text']}' (Score/Prob: {c['prob']*100:.2f}%)", flush=True)


# 2. Test Training Sample: crop_o_kothi.png
print("\n" + "=" * 80, flush=True)
print("TEST 2: TRAINING SAMPLE (crop_o_kothi.png, GT: 'ಕೋತಿ')", flush=True)
print("=" * 80, flush=True)

train_crop_path = r"c:\Land Record\training\datasets\personal_trial\crops\crop_o_kothi.png"
train_crop = Image.open(train_crop_path).convert("RGB")
print(f"Crop dimensions: {train_crop.size}", flush=True)

cands_curr_train = run_beam_search_top5(model_curr, proc_curr, tok_curr, train_crop)
cands_targ_train = run_beam_search_top5(model_targ, proc_targ, tok_targ, train_crop)

print("\n[A] CURRENT MODEL (personal_trial_checkpoints):", flush=True)
for c in cands_curr_train:
    print(f"  Rank #{c['rank']}: '{c['text']}' (Score/Prob: {c['prob']*100:.2f}%)", flush=True)

print("\n[B] TARGETED CANDIDATE MODEL (kothi_targeted_checkpoints):", flush=True)
for c in cands_targ_train:
    print(f"  Rank #{c['rank']}: '{c['text']}' (Score/Prob: {c['prob']*100:.2f}%)", flush=True)

print("\n" + "=" * 80, flush=True)
print("[✓] Comparison Evaluation Complete.", flush=True)
print("=" * 80, flush=True)
