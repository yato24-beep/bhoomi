import json
import os
import sys
from collections import defaultdict
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TARGETS = ["ತಿ", "ಲಿ", "ಟಿ", "ಡಿ", "ರಿ"]

def levenshtein_dist(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_dist(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def evaluate_checkpoint(ckpt_path: str, diag_val_path: str, device="cuda"):
    print(f"\n=======================================================", flush=True)
    print(f"EVALUATING CHECKPOINT: {ckpt_path}", flush=True)
    print(f"DIAGNOSTIC DATASET: {diag_val_path}", flush=True)
    print(f"=======================================================", flush=True)

    tokenizer = XLMRobertaTokenizer.from_pretrained(ckpt_path)
    processor = TrOCRProcessor.from_pretrained(ckpt_path)
    model = VisionEncoderDecoderModel.from_pretrained(ckpt_path).to(device).eval()

    with open(diag_val_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    total_samples = len(samples)
    exact_matches = 0
    total_cer_dist = 0
    total_gt_chars = 0

    per_target_total = defaultdict(int)
    per_target_correct = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    print(f"Running evaluation on {total_samples} samples...", flush=True)
    for idx, s in enumerate(samples, 1):
        rel_img = s["image"]
        img_p = os.path.join(r"c:\Land Record", rel_img)
        if not os.path.exists(img_p):
            img_p = os.path.join(r"c:\Land Record\training", rel_img)
        if not os.path.exists(img_p):
            continue

        gt_text = s["text"]
        target_char = s.get("target_char", "anchor")

        img = Image.open(img_p).convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                num_beams=5,
                max_new_tokens=32,
                early_stopping=True,
                decoder_start_token_id=tokenizer.bos_token_id or tokenizer.cls_token_id or 0,
            )
        pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        # Word level stats
        is_exact = (pred_text == gt_text)
        if is_exact:
            exact_matches += 1

        # Character error stats
        dist = levenshtein_dist(gt_text, pred_text)
        total_cer_dist += dist
        total_gt_chars += len(gt_text)

        # Target character stats
        if target_char in TARGETS:
            per_target_total[target_char] += 1
            if target_char in pred_text:
                per_target_correct[target_char] += 1
                confusion[target_char][target_char] += 1
            else:
                confused_with = []
                for other_t in TARGETS:
                    if other_t in pred_text:
                        confused_with.append(other_t)
                if confused_with:
                    for cw in confused_with:
                        confusion[target_char][cw] += 1
                else:
                    confusion[target_char]["other/omitted"] += 1
        else:
            per_target_total["anchor"] += 1
            if is_exact:
                per_target_correct["anchor"] += 1

        if idx % 30 == 0 or idx == total_samples:
            print(f"  Processed {idx}/{total_samples} samples...", flush=True)

    word_acc = (exact_matches / total_samples) * 100 if total_samples > 0 else 0
    mean_cer = (total_cer_dist / total_gt_chars) * 100 if total_gt_chars > 0 else 0

    print(f"\n--- OVERALL DIAGNOSTIC METRICS ---", flush=True)
    print(f"Total Samples: {total_samples}", flush=True)
    print(f"Exact Match Accuracy: {word_acc:.2f}% ({exact_matches}/{total_samples})", flush=True)
    print(f"Mean Character Error Rate (CER): {mean_cer:.2f}%", flush=True)

    print(f"\n--- PER-TARGET CHARACTER ACCURACY ---", flush=True)
    for t in TARGETS + ["anchor"]:
        tot = per_target_total[t]
        corr = per_target_correct[t]
        acc = (corr / tot * 100) if tot > 0 else 0
        print(f"  Character '{t}': {corr}/{tot} ({acc:.1f}%)", flush=True)

    print(f"\n--- CHARACTER CONFUSION BREAKDOWN ---", flush=True)
    for t in TARGETS:
        print(f"  Ground Truth '{t}':", flush=True)
        for pred_c, cnt in sorted(confusion[t].items(), key=lambda x: -x[1]):
            print(f"    -> '{pred_c}': {cnt} times", flush=True)

    return {
        "word_acc": word_acc,
        "mean_cer": mean_cer,
        "per_target_correct": dict(per_target_correct),
        "per_target_total": dict(per_target_total),
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else r"c:\Land Record\models\trocr\personal_trial_checkpoints\best_checkpoint"
    diag_val = sys.argv[2] if len(sys.argv) > 2 else r"c:\Land Record\training\datasets\kothi_targeted\diagnostic_val.jsonl"
    evaluate_checkpoint(ckpt, diag_val)
