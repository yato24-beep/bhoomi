import os
import sys
import json
import time

sys.path.insert(0, r"c:\Land Record")
import src
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, XLMRobertaTokenizer
from src.training.evaluate import evaluate_predictions

VAL_MANIFEST = r"c:\Land Record\training\datasets\iiit_kannada_val.jsonl"
CKPT_PREV = r"c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint"
CKPT_NEW = r"c:\Land Record\models\trocr\kannada_full_checkpoints\checkpoint-step-55390"

# Load validation samples (first 100 samples for swift evaluation)
samples = []
with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        img_p = os.path.join(r"c:\Land Record", data["image"])
        if os.path.exists(img_p):
            samples.append((img_p, data["text"]))
        if len(samples) >= 100:
            break

print(f"Loaded {len(samples)} validation samples from {VAL_MANIFEST}")

def evaluate_checkpoint(ckpt_path, label):
    print(f"\n--- Evaluating {label}: {ckpt_path} ---", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = XLMRobertaTokenizer.from_pretrained(ckpt_path)
    processor = TrOCRProcessor.from_pretrained(ckpt_path)
    model = VisionEncoderDecoderModel.from_pretrained(ckpt_path).to(device)
    model.eval()

    references = []
    hypotheses = []
    languages = []

    start = time.perf_counter()
    for img_path, ref_text in samples:
        img = Image.open(img_path).convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            gen_ids = model.generate(pixel_values, max_new_tokens=32)
        hyp_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        references.append(ref_text)
        hypotheses.append(hyp_text)
        languages.append("kannada")

    elapsed = time.perf_counter() - start
    report = evaluate_predictions(references, hypotheses, languages)
    print(f"Results for {label}:")
    print(f"  Samples: {len(samples)}")
    print(f"  CER: {report.overall_cer:.4f}")
    print(f"  WER: {report.overall_wer:.4f}")
    print(f"  Time: {elapsed:.2f}s ({len(samples)/elapsed:.1f} samples/s)")
    print(f"  Sample Predictions:")
    for i in range(min(5, len(samples))):
        print(f"    [{i+1}] Ref: {references[i]} | Hyp: {hypotheses[i]}")

    return {
        "label": label,
        "cer": report.overall_cer,
        "wer": report.overall_wer,
        "samples": len(samples),
        "time": elapsed,
        "sample_preds": [{"ref": references[i], "hyp": hypotheses[i]} for i in range(5)]
    }

res_prev = evaluate_checkpoint(CKPT_PREV, "Previous Best Checkpoint (Step 55140, Epoch 6)")
res_new = evaluate_checkpoint(CKPT_NEW, "Newly Trained Checkpoint (Step 55390, Epoch 7)")

out_summary = {
    "previous_checkpoint": res_prev,
    "new_checkpoint": res_new,
}
with open(r"c:\Land Record\scratch\eval_comparison_result.json", "w", encoding="utf-8") as f:
    json.dump(out_summary, f, indent=2, ensure_ascii=False)

print("\nSaved evaluation comparison to scratch/eval_comparison_result.json")
