"""
Comprehensive Comparative Evaluation Script for Phase 7, Phase 8, Phase 9, Phase 10
Sequential single-model loading to ensure minimal VRAM usage.
"""
import os
import sys
import gc
import json
from collections import defaultdict, Counter
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RARE_CHARACTERS = set("ಋಐಔಙಞಢಝಠಛಘಫಣ")
MATRAS = set("ಾಿೀುೂೃೆೇೈೊೋೌ್ಂಃ")
NUMERALS = set("೦೧೨೩೪೫೬೭೮೯")

def levenshtein_ops(s1, s2):
    """
    Computes Levenshtein distance and alignment edit operations.
    Returns: (distance, list of (tag, s1_idx, s2_idx))
    where tag is 'match', 'replace', 'delete', 'insert'
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # deletion
                                   dp[i][j-1],      # insertion
                                   dp[i-1][j-1])    # replacement
                                   
    dist = dp[m][n]
    
    # Backtrack alignment ops
    i, j = m, n
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            ops.append(('match', i-1, j-1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(('replace', i-1, j-1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(('delete', i-1, -1))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ops.append(('insert', -1, j-1))
            j -= 1
        else:
            if i > 0 and j > 0:
                ops.append(('match', i-1, j-1))
                i -= 1
                j -= 1
            elif i > 0:
                ops.append(('delete', i-1, -1))
                i -= 1
            else:
                ops.append(('insert', -1, j-1))
                j -= 1
                
    ops.reverse()
    return dist, ops

def evaluate_single_checkpoint(checkpoint_path, dataset_path, desc="Model"):
    print(f"\n{'='*70}", flush=True)
    print(f"Loading {desc} from: {checkpoint_path}", flush=True)
    print(f"{'='*70}", flush=True)
    
    processor = TrOCRProcessor.from_pretrained(checkpoint_path)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_path).to(DEVICE)
    model.eval()
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
        
    print(f"Evaluating on {len(samples)} samples...", flush=True)
    
    total_words = len(samples)
    exact_matches = 0
    total_cer_sum = 0.0
    total_chars = 0
    correct_chars = 0
    conf_scores = []
    
    subgroup_stats = {
        "rare": {"total": 0, "correct": 0},
        "matras": {"total": 0, "correct": 0},
        "numerals": {"total": 0, "correct": 0},
        "conjunct_words": {"total": 0, "exact": 0}
    }
    
    char_counts = defaultdict(lambda: {"total": 0, "correct": 0})
    all_confusions = Counter()
    per_sample_results = []
    
    # Safe batch size for 6GB VRAM with beam_size=5
    BATCH_SIZE = 8
    for b_start in range(0, total_words, BATCH_SIZE):
        batch_samples = samples[b_start : b_start + BATCH_SIZE]
        if b_start % 80 == 0 or b_start + BATCH_SIZE >= total_words:
            print(f"[{desc}] Progress: {b_start}/{total_words}...", flush=True)
            
        images = []
        valid_indices = []
        for idx, s in enumerate(batch_samples):
            img_path = s["image"]
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error opening {img_path}: {e}", flush=True)
                
        if not images:
            continue
            
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                max_length=64,
                return_dict_in_generate=True,
                output_scores=True,
                num_beams=5,
                num_return_sequences=1
            )
            
        pred_texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
        
        # Transition scores
        try:
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            confs = (torch.exp(transition_scores.sum(dim=1) / torch.clamp(torch.tensor(transition_scores.shape[1]), min=1)).cpu().numpy() * 100).tolist()
        except Exception:
            confs = [0.0] * len(pred_texts)
            
        for i, idx in enumerate(valid_indices):
            s = batch_samples[idx]
            gt_text = s["text"]
            pred_text = pred_texts[i].strip()
            conf = confs[i] if i < len(confs) else 0.0
            
            conf_scores.append(conf)
            
            # Word Exact Match
            is_exact = (gt_text == pred_text)
            if is_exact:
                exact_matches += 1
                
            # Distance & Alignment
            dist, ops = levenshtein_ops(gt_text, pred_text)
            cer = dist / max(1, len(gt_text))
            total_cer_sum += cer
            
            # Character accuracy
            replaced_or_deleted = sum(1 for tag, _, _ in ops if tag in ("replace", "delete"))
            matches = max(0, len(gt_text) - replaced_or_deleted)
            total_chars += len(gt_text)
            correct_chars += matches
            
            # Subgroups
            if "್" in gt_text:
                subgroup_stats["conjunct_words"]["total"] += 1
                if is_exact:
                    subgroup_stats["conjunct_words"]["exact"] += 1
                    
            gt_deleted = {s1_i for tag, s1_i, _ in ops if tag == "delete"}
            gt_replaced = {s1_i for tag, s1_i, _ in ops if tag == "replace"}
            
            for c_idx, ch in enumerate(gt_text):
                char_counts[ch]["total"] += 1
                if c_idx not in gt_deleted and c_idx not in gt_replaced:
                    char_counts[ch]["correct"] += 1
                    
                if ch in RARE_CHARACTERS:
                    subgroup_stats["rare"]["total"] += 1
                    if c_idx not in gt_deleted and c_idx not in gt_replaced:
                        subgroup_stats["rare"]["correct"] += 1
                elif ch in MATRAS:
                    subgroup_stats["matras"]["total"] += 1
                    if c_idx not in gt_deleted and c_idx not in gt_replaced:
                        subgroup_stats["matras"]["correct"] += 1
                elif ch in NUMERALS:
                    subgroup_stats["numerals"]["total"] += 1
                    if c_idx not in gt_deleted and c_idx not in gt_replaced:
                        subgroup_stats["numerals"]["correct"] += 1
                        
            # Record confusions
            for tag, s1_i, s2_j in ops:
                if tag == "replace":
                    all_confusions[(gt_text[s1_i], pred_text[s2_j])] += 1
                elif tag == "delete":
                    all_confusions[(gt_text[s1_i], "<OMITTED>")] += 1
                elif tag == "insert":
                    all_confusions[("<INSERTED>", pred_text[s2_j])] += 1
                    
            per_sample_results.append({
                "image": s["image"],
                "gt": gt_text,
                "pred": pred_text,
                "conf": conf,
                "cer": cer,
                "exact": is_exact
            })
            
    avg_conf = sum(conf_scores) / max(1, len(conf_scores))
    human_review_count = sum(1 for r in per_sample_results if r["conf"] < 70.0 or not r["exact"])
    
    # Cleanup memory
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    summary = {
        "word_acc": (exact_matches / total_words) * 100,
        "mean_cer": (total_cer_sum / total_words) * 100,
        "char_acc": (correct_chars / total_chars) * 100,
        "avg_conf": avg_conf,
        "human_review_rate": (human_review_count / total_words) * 100,
        "rare_acc": (subgroup_stats["rare"]["correct"] / max(1, subgroup_stats["rare"]["total"])) * 100,
        "matra_acc": (subgroup_stats["matras"]["correct"] / max(1, subgroup_stats["matras"]["total"])) * 100,
        "numeral_acc": (subgroup_stats["numerals"]["correct"] / max(1, subgroup_stats["numerals"]["total"])) * 100,
        "conjunct_acc": (subgroup_stats["conjunct_words"]["exact"] / max(1, subgroup_stats["conjunct_words"]["total"])) * 100,
        "char_counts": dict(char_counts),
        "confusions": all_confusions,
        "samples": per_sample_results
    }
    return summary

def run_full_comparison(base_ckpt, gen_ckpt, val_dataset):
    base_res = evaluate_single_checkpoint(base_ckpt, val_dataset, desc="Base Model")
    gen_res = evaluate_single_checkpoint(gen_ckpt, val_dataset, desc="Generalized Candidate")
    
    # Save full raw evaluation data to json
    output_data = {
        "base_metrics": {k: v for k, v in base_res.items() if k not in ("char_counts", "confusions", "samples")},
        "gen_metrics": {k: v for k, v in gen_res.items() if k not in ("char_counts", "confusions", "samples")},
        "base_char_counts": {k: v for k, v in base_res["char_counts"].items()},
        "gen_char_counts": {k: v for k, v in gen_res["char_counts"].items()},
        "base_top_confusions": [list(k) + [v] for k, v in base_res["confusions"].most_common(50)],
        "gen_top_confusions": [list(k) + [v] for k, v in gen_res["confusions"].most_common(50)],
    }
    
    with open("diagnostic_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*80, flush=True)
    print("COMPARISON RESULTS SUMMARY TABLE", flush=True)
    print("="*80, flush=True)
    print(f"{'Metric':<25} | {'Current (Base)':<15} | {'Generalized':<15} | {'Difference':<15}", flush=True)
    print("-" * 78, flush=True)
    
    metrics_list = [
        ("Word Accuracy (%)", "word_acc"),
        ("Mean CER (%)", "mean_cer"),
        ("Character Accuracy (%)", "char_acc"),
        ("Rare Char Accuracy (%)", "rare_acc"),
        ("Matra Accuracy (%)", "matra_acc"),
        ("Numeral Accuracy (%)", "numeral_acc"),
        ("Conjunct Word Acc (%)", "conjunct_acc"),
        ("Average Confidence (%)", "avg_conf"),
        ("Human Review Rate (%)", "human_review_rate"),
    ]
    
    for label, key in metrics_list:
        b_val = base_res[key]
        g_val = gen_res[key]
        diff = g_val - b_val
        diff_str = f"{diff:+.2f}%"
        print(f"{label:<25} | {b_val:<14.2f}% | {g_val:<14.2f}% | {diff_str:<15}", flush=True)
        
    # Calculate Per-Character Recall Deltas
    char_deltas = []
    all_chars = set(base_res["char_counts"].keys()) | set(gen_res["char_counts"].keys())
    for ch in all_chars:
        b_tot = base_res["char_counts"].get(ch, {}).get("total", 0)
        b_cor = base_res["char_counts"].get(ch, {}).get("correct", 0)
        b_rec = (b_cor / b_tot * 100) if b_tot > 0 else 0.0
        
        g_tot = gen_res["char_counts"].get(ch, {}).get("total", 0)
        g_cor = gen_res["char_counts"].get(ch, {}).get("correct", 0)
        g_rec = (g_cor / g_tot * 100) if g_tot > 0 else 0.0
        
        if b_tot >= 3:
            char_deltas.append({
                "char": ch,
                "total": b_tot,
                "base_rec": b_rec,
                "gen_rec": g_rec,
                "delta": g_rec - b_rec
            })
            
    improved = sorted([c for c in char_deltas if c["delta"] > 0], key=lambda x: x["delta"], reverse=True)[:20]
    regressed = sorted([c for c in char_deltas if c["delta"] < 0], key=lambda x: x["delta"])[:20]
    
    print("\n" + "="*80, flush=True)
    print("TOP 20 IMPROVED CHARACTERS", flush=True)
    print("="*80, flush=True)
    print(f"{'Char':<6} | {'Samples':<8} | {'Base Recall':<12} | {'Gen Recall':<12} | {'Improvement':<12}", flush=True)
    print("-" * 60, flush=True)
    for c in improved:
        print(f"{c['char']:<6} | {c['total']:<8} | {c['base_rec']:<11.1f}% | {c['gen_rec']:<11.1f}% | {c['delta']:+.1f}%", flush=True)
        
    print("\n" + "="*80, flush=True)
    print("TOP 20 REGRESSED CHARACTERS", flush=True)
    print("="*80, flush=True)
    print(f"{'Char':<6} | {'Samples':<8} | {'Base Recall':<12} | {'Gen Recall':<12} | {'Regression':<12}", flush=True)
    print("-" * 60, flush=True)
    for c in regressed:
        print(f"{c['char']:<6} | {c['total']:<8} | {c['base_rec']:<11.1f}% | {c['gen_rec']:<11.1f}% | {c['delta']:+.1f}%", flush=True)

    print("\n" + "="*80, flush=True)
    print("TOP 20 CHARACTER CONFUSIONS IN GENERALIZED MODEL", flush=True)
    print("="*80, flush=True)
    print(f"{'Ground Truth':<15} -> {'Prediction':<15} : {'Count'}", flush=True)
    print("-" * 50, flush=True)
    for (gt_c, pr_c), count in gen_res["confusions"].most_common(20):
        print(f"{gt_c:<15} -> {pr_c:<15} : {count}", flush=True)

if __name__ == "__main__":
    base_ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/trocr/kannada_full_checkpoints/best_checkpoint"
    gen_ckpt = sys.argv[2] if len(sys.argv) > 2 else "models/trocr/kannada_generalized_checkpoints/best_checkpoint"
    val_data = sys.argv[3] if len(sys.argv) > 3 else "training/datasets/kannada_diagnostic_val/diagnostic_full.jsonl"
    run_full_comparison(base_ckpt, gen_ckpt, val_data)
