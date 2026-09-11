import json
import os
import sys
from collections import Counter, defaultdict
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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

def run_character_evaluation(ckpt_path: str, diag_manifest: str, device="cuda", max_samples=None):
    print("=" * 80, flush=True)
    print(f"EVALUATING MODEL: {ckpt_path}", flush=True)
    print(f"DATASET: {diag_manifest}", flush=True)
    print("=" * 80, flush=True)

    tokenizer = XLMRobertaTokenizer.from_pretrained(ckpt_path)
    processor = TrOCRProcessor.from_pretrained(ckpt_path)
    model = VisionEncoderDecoderModel.from_pretrained(ckpt_path).to(device).eval()

    with open(diag_manifest, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    total_samples = len(samples)
    exact_matches = 0
    total_edit_distance = 0
    total_gt_chars = 0
    total_pred_chars = 0
    confidence_scores = []

    # Per-character ground truth and prediction counts for recall/precision
    gt_char_counts = Counter()
    pred_char_counts = Counter()
    correct_char_counts = Counter()

    # Confusions: (gt_char, pred_char) -> count
    confusion_pairs = Counter()

    # Category tracking
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)

    print(f"Running inference across {total_samples} samples...", flush=True)

    for idx, s in enumerate(samples, 1):
        rel_img = s.get("image", "")
        img_p = os.path.join(r"c:\Land Record", rel_img)
        if not os.path.exists(img_p):
            continue

        gt_text = s.get("text", "")
        tags = s.get("diag_tags", [])

        img = Image.open(img_p).convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                num_beams=5,
                max_new_tokens=32,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                decoder_start_token_id=tokenizer.bos_token_id or tokenizer.cls_token_id or 0,
            )

        gen_ids = outputs.sequences[0]
        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Confidence calculation
        if outputs.scores:
            step_probs = []
            for s_idx, s_logits in enumerate(outputs.scores):
                t_id = gen_ids[s_idx + 1] if s_idx + 1 < len(gen_ids) else None
                if t_id is not None:
                    p = torch.softmax(s_logits[0], dim=-1)[t_id].item()
                    step_probs.append(p)
            avg_p = sum(step_probs) / len(step_probs) if step_probs else 0.50
        else:
            avg_p = 0.50
        confidence_scores.append(avg_p)

        # Exact match
        is_exact = (pred_text == gt_text)
        if is_exact:
            exact_matches += 1

        # CER & Edit distance
        dist = levenshtein_dist(gt_text, pred_text)
        total_edit_distance += dist
        total_gt_chars += len(gt_text)
        total_pred_chars += len(pred_text)

        # Character-level matching and confusion alignment
        for ch in gt_text:
            gt_char_counts[ch] += 1
        for ch in pred_text:
            pred_char_counts[ch] += 1

        # Align characters for confusion matrix using needleman-wunsch / simplified alignment
        # Pairwise character confusion approximation
        min_len = min(len(gt_text), len(pred_text))
        for i in range(min_len):
            g_ch = gt_text[i]
            p_ch = pred_text[i]
            if g_ch == p_ch:
                correct_char_counts[g_ch] += 1
            else:
                confusion_pairs[(g_ch, p_ch)] += 1

        if len(gt_text) > len(pred_text):
            for i in range(min_len, len(gt_text)):
                confusion_pairs[(gt_text[i], "[OMITTED]")] += 1
        elif len(pred_text) > len(gt_text):
            for i in range(min_len, len(pred_text)):
                confusion_pairs[("[INSERTED]", pred_text[i])] += 1

        # Tags categorization
        for tag in tags:
            cat_total[tag] += 1
            if is_exact:
                cat_correct[tag] += 1

        if idx % 100 == 0 or idx == total_samples:
            print(f"  Evaluated {idx}/{total_samples} samples...", flush=True)

    word_acc = (exact_matches / total_samples) * 100 if total_samples else 0
    mean_cer = (total_edit_distance / total_gt_chars) * 100 if total_gt_chars else 0
    total_correct_chars = sum(correct_char_counts.values())
    char_acc = (total_correct_chars / total_gt_chars) * 100 if total_gt_chars else 0
    avg_conf = (sum(confidence_scores) / len(confidence_scores)) * 100 if confidence_scores else 0

    print("\n" + "=" * 80, flush=True)
    print("  SUMMARY BENCHMARK REPORT", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Samples:            {total_samples:,}", flush=True)
    print(f"Word Exact Match:         {word_acc:.2f}% ({exact_matches}/{total_samples})", flush=True)
    print(f"Mean Character Error Rate: {mean_cer:.2f}%", flush=True)
    print(f"Overall Character Accuracy:{char_acc:.2f}% ({total_correct_chars}/{total_gt_chars})", flush=True)
    print(f"Average Confidence:       {avg_conf:.2f}%", flush=True)

    # Per-character Recall & Precision
    per_char_stats = []
    for ch, gt_c in gt_char_counts.items():
        if gt_c < 3:
            continue
        corr_c = correct_char_counts[ch]
        p_c = pred_char_counts[ch]
        recall = (corr_c / gt_c) * 100 if gt_c else 0
        precision = (corr_c / p_c) * 100 if p_c else 0
        per_char_stats.append({
            "char": ch,
            "gt_count": gt_c,
            "pred_count": p_c,
            "correct": corr_c,
            "recall": recall,
            "precision": precision,
        })

    # Sort to find 20 worst recall characters
    per_char_stats.sort(key=lambda x: x["recall"])
    print("\n--- 20 WORST-PERFORMING CHARACTERS (LOWEST RECALL) ---", flush=True)
    print(f"  {'Char':<6} | {'GT Count':<10} | {'Correct':<10} | {'Recall':<10} | {'Precision':<10}", flush=True)
    print("-" * 60, flush=True)
    for row in per_char_stats[:20]:
        print(f"  {row['char']:<6} | {row['gt_count']:<10} | {row['correct']:<10} | {row['recall']:6.2f}%   | {row['precision']:6.2f}%", flush=True)

    # Top 20 Confusion pairs
    print("\n--- TOP 20 CHARACTER CONFUSIONS (GROUND TRUTH -> PREDICTED) ---", flush=True)
    print(f"  {'Ground Truth':<14} -> {'Predicted':<14} | {'Count':<8}", flush=True)
    print("-" * 50, flush=True)
    top_conf = [item for item in confusion_pairs.most_common(30) if item[0][0] != item[0][1]][:20]
    for (g_ch, p_ch), cnt in top_conf:
        print(f"  '{g_ch:<12}' -> '{p_ch:<12}' | {cnt:5d} times", flush=True)

    return {
        "word_acc": word_acc,
        "mean_cer": mean_cer,
        "char_acc": char_acc,
        "avg_conf": avg_conf,
        "per_char_stats": per_char_stats,
        "top_confusions": top_conf,
        "cat_correct": dict(cat_correct),
        "cat_total": dict(cat_total),
    }

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else r"c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint"
    diag = sys.argv[2] if len(sys.argv) > 2 else r"c:\Land Record\training\datasets\kannada_diagnostic_val\diagnostic_full.jsonl"
    run_character_evaluation(ckpt, diag)
