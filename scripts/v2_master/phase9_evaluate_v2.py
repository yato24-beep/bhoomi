"""Phase 9 — Comprehensive V2 Evaluation.

Compares Base, V1, and V2 on the SAME untouched diagnostic benchmark.
Computes per-character recall, top confusions, matra/virama/conjunct/numeral accuracy.
"""

import collections
import json
import re
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KANNADA_CONSONANTS = set("ಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹಳ")
KANNADA_MATRAS = set("ಾಿೀುೂೃೆೇೈೊೋೌ")
KANNADA_VIRAMA = "್"
KANNADA_NUMERALS = set("೦೧೨೩೪೫೬೭೮೯")


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def levenshtein_ops(ref, hyp):
    """Return list of (op, ref_char, hyp_char) for edit operations."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    
    # Backtrace
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            ops.append(("match", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("substitute", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("insert", "", hyp[j - 1]))
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("delete", ref[i - 1], ""))
            i -= 1
        else:
            break
    ops.reverse()
    return ops


def compute_cer(ref, hyp):
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def evaluate_model(model_path, records, model_label):
    """Run a model on records and return per-sample predictions."""
    print(f"\n  Loading {model_label} from: {model_path}")
    from src.handwriting.trocr_recognizer import TrocrHandwritingRecognizer
    
    recognizer = TrocrHandwritingRecognizer(
        model_name_or_path=str(model_path),
        auto_load=True,
        preprocess_input=False,  # Raw crop evaluation
    )
    
    if not recognizer.is_loaded:
        print(f"    [ERROR] Failed to load model: {recognizer._load_error}")
        return None
    
    print(f"    Device: {recognizer.device}, FP16: {recognizer.is_fp16}")
    
    predictions = []
    start = time.time()
    for idx, r in enumerate(records):
        img_path = PROJECT_ROOT / r["image"]
        if not img_path.exists():
            predictions.append("")
            continue
        try:
            res = recognizer.recognize_handwriting(img_path, preprocess_input=False)
            predictions.append(res.text)
        except Exception as e:
            predictions.append("")
        
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"    Progress: {idx + 1}/{len(records)} ({elapsed:.1f}s)", flush=True)
    
    elapsed = time.time() - start
    print(f"    Completed: {len(records)} samples in {elapsed:.1f}s ({len(records)/elapsed:.1f} samples/s)")
    return predictions


def compute_metrics(references, hypotheses):
    """Compute comprehensive metrics."""
    # Overall CER
    total_ref_chars = sum(len(r) for r in references)
    total_edits = 0
    word_matches = 0
    
    # Per-character tracking
    char_correct = collections.Counter()
    char_total = collections.Counter()
    confusion_pairs = collections.Counter()
    
    # Category tracking
    matra_correct = 0
    matra_total = 0
    virama_correct = 0
    virama_total = 0
    numeral_correct = 0
    numeral_total = 0
    
    # Conjunct tracking
    conjunct_pattern = re.compile(f"[{''.join(KANNADA_CONSONANTS)}]{KANNADA_VIRAMA}[{''.join(KANNADA_CONSONANTS)}]")
    conjunct_word_correct = 0
    conjunct_word_total = 0
    
    # Rare character tracking (chars appearing < 50 times in dataset)
    rare_chars = set()  # Will be populated below
    rare_correct = 0
    rare_total = 0
    
    for ref, hyp in zip(references, hypotheses):
        cer = compute_cer(ref, hyp)
        total_edits += int(cer * len(ref))
        
        if ref == hyp:
            word_matches += 1
        
        # Per-character analysis via alignment
        ops = levenshtein_ops(ref, hyp)
        for op_type, ref_ch, hyp_ch in ops:
            if ref_ch and "\u0c80" <= ref_ch <= "\u0cff":
                char_total[ref_ch] += 1
                
                if op_type == "match":
                    char_correct[ref_ch] += 1
                elif op_type == "substitute":
                    confusion_pairs[(ref_ch, hyp_ch)] += 1
                elif op_type == "delete":
                    confusion_pairs[(ref_ch, "<DEL>")] += 1
                
                # Matra tracking
                if ref_ch in KANNADA_MATRAS:
                    matra_total += 1
                    if op_type == "match":
                        matra_correct += 1
                
                # Virama tracking
                if ref_ch == KANNADA_VIRAMA:
                    virama_total += 1
                    if op_type == "match":
                        virama_correct += 1
                
                # Numeral tracking
                if ref_ch in KANNADA_NUMERALS:
                    numeral_total += 1
                    if op_type == "match":
                        numeral_correct += 1
        
        # Conjunct word accuracy
        if conjunct_pattern.search(ref):
            conjunct_word_total += 1
            if ref == hyp:
                conjunct_word_correct += 1
    
    # Identify rare characters (< 50 total in eval set)
    for ch, count in char_total.items():
        if count < 50:
            rare_chars.add(ch)
    
    for ch in rare_chars:
        rare_total += char_total[ch]
        rare_correct += char_correct.get(ch, 0)
    
    overall_cer = total_edits / total_ref_chars if total_ref_chars > 0 else 0
    char_acc = 1.0 - overall_cer
    word_acc = word_matches / len(references) if references else 0
    
    return {
        "char_accuracy": round(char_acc * 100, 2),
        "cer": round(overall_cer * 100, 2),
        "word_exact_match": round(word_acc * 100, 2),
        "matra_accuracy": round(matra_correct / matra_total * 100, 2) if matra_total > 0 else 0,
        "virama_accuracy": round(virama_correct / virama_total * 100, 2) if virama_total > 0 else 0,
        "numeral_accuracy": round(numeral_correct / numeral_total * 100, 2) if numeral_total > 0 else 0,
        "conjunct_word_accuracy": round(conjunct_word_correct / conjunct_word_total * 100, 2) if conjunct_word_total > 0 else 0,
        "rare_char_accuracy": round(rare_correct / rare_total * 100, 2) if rare_total > 0 else 0,
        "total_samples": len(references),
        "total_ref_chars": total_ref_chars,
        "matra_total": matra_total,
        "virama_total": virama_total,
        "numeral_total": numeral_total,
        "conjunct_word_total": conjunct_word_total,
        "rare_total": rare_total,
        "char_correct": dict(char_correct),
        "char_total": dict(char_total),
        "confusion_pairs": {f"{k[0]}→{k[1]}": v for k, v in confusion_pairs.most_common(50)},
    }


def main():
    print("=" * 70)
    print("  PHASE 9 — COMPREHENSIVE V2 EVALUATION")
    print("=" * 70)

    # Load diagnostic benchmark (untouched)
    diag_path = PROJECT_ROOT / "training" / "datasets" / "kannada_diagnostic_val" / "diagnostic_full.jsonl"
    records = load_jsonl(diag_path)
    references = [r["text"] for r in records]
    print(f"Diagnostic benchmark: {len(records)} samples")

    results = {}

    # V1 Model
    v1_path = PROJECT_ROOT / "models" / "trocr" / "kannada_generalized_checkpoints" / "best_checkpoint"
    if v1_path.exists():
        v1_preds = evaluate_model(v1_path, records, "V1 (Generalized)")
        if v1_preds:
            results["V1"] = compute_metrics(references, v1_preds)

    # V2 Model
    v2_path = PROJECT_ROOT / "models" / "trocr" / "kannada_generalized_v2_checkpoints" / "best_checkpoint"
    if v2_path.exists():
        v2_preds = evaluate_model(v2_path, records, "V2 (New)")
        if v2_preds:
            results["V2"] = compute_metrics(references, v2_preds)
    else:
        print(f"\n  [SKIP] V2 checkpoint not found at: {v2_path}")
        print(f"  Run Phase 8 training first.")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  MODEL COMPARISON RESULTS")
    print(f"{'='*80}")
    
    metrics_list = ["char_accuracy", "cer", "word_exact_match", "rare_char_accuracy", 
                    "matra_accuracy", "virama_accuracy", "numeral_accuracy", "conjunct_word_accuracy"]
    metric_labels = {
        "char_accuracy": "Character Accuracy %",
        "cer": "CER %",
        "word_exact_match": "Word Exact Match %",
        "rare_char_accuracy": "Rare Character Acc %",
        "matra_accuracy": "Matra Accuracy %",
        "virama_accuracy": "Virama Accuracy %",
        "numeral_accuracy": "Numeral Accuracy %",
        "conjunct_word_accuracy": "Conjunct Word Acc %",
    }
    
    header = f"  {'Metric':<25}"
    for model_name in results:
        header += f" | {model_name:>10}"
    if len(results) >= 2:
        header += f" | {'Delta':>8}"
    print(header)
    print("  " + "-" * len(header))
    
    for m in metrics_list:
        row = f"  {metric_labels[m]:<25}"
        vals = []
        for model_name in results:
            val = results[model_name].get(m, 0)
            vals.append(val)
            row += f" | {val:>10.2f}"
        if len(vals) >= 2:
            delta = vals[-1] - vals[0]
            sign = "+" if delta > 0 else ""
            row += f" | {sign}{delta:>7.2f}"
        print(row)

    # Per-character recall comparison
    if len(results) >= 1:
        print(f"\n{'='*80}")
        print(f"  PER-CHARACTER RECALL (sorted by worst V2 recall)")
        print(f"{'='*80}")
        
        last_model = list(results.keys())[-1]
        last_metrics = results[last_model]
        
        char_rows = []
        for ch, total in last_metrics["char_total"].items():
            correct = last_metrics["char_correct"].get(ch, 0)
            recall = correct / total * 100 if total > 0 else 0
            
            v1_recall = None
            if "V1" in results:
                v1_total = results["V1"]["char_total"].get(ch, 0)
                v1_correct = results["V1"]["char_correct"].get(ch, 0)
                v1_recall = v1_correct / v1_total * 100 if v1_total > 0 else 0
            
            char_rows.append((ch, total, v1_recall, recall))
        
        char_rows.sort(key=lambda x: x[3])  # Sort by last model recall
        
        print(f"  {'Char':<6} | {'Samples':>8} | {'V1 Recall':>10} | {f'{last_model} Recall':>10} | {'Delta':>8}")
        print("  " + "-" * 55)
        for ch, total, v1_r, v2_r in char_rows[:40]:
            v1_str = f"{v1_r:.1f}%" if v1_r is not None else "N/A"
            delta = v2_r - v1_r if v1_r is not None else 0
            sign = "+" if delta > 0 else ""
            print(f"  {ch:<6} | {total:>8} | {v1_str:>10} | {v2_r:>9.1f}% | {sign}{delta:>7.1f}")

    # Top confusions
    if results:
        last_model = list(results.keys())[-1]
        print(f"\n{'='*80}")
        print(f"  TOP 30 CONFUSIONS ({last_model})")
        print(f"{'='*80}")
        confusions = results[last_model].get("confusion_pairs", {})
        for i, (pair, count) in enumerate(sorted(confusions.items(), key=lambda x: -x[1])[:30], 1):
            print(f"  {i:>2}. {pair}: {count}")

    # 90%/95% Gate Check
    print(f"\n{'='*80}")
    print(f"  PHASE 10 — ACCURACY GATE CHECK")
    print(f"{'='*80}")
    
    if results:
        last_model = list(results.keys())[-1]
        char_acc = results[last_model]["char_accuracy"]
        print(f"  {last_model} Character Accuracy: {char_acc:.2f}%")
        
        if char_acc >= 95:
            print(f"  ★ STATUS: EXCELLENT (≥95%)")
        elif char_acc >= 90:
            print(f"  ★ STATUS: ACCEPTABLE / PROMOTE (≥90%)")
        else:
            print(f"  ✗ STATUS: BELOW TARGET (<90%)")
            print(f"    Deficit: {90 - char_acc:.2f}% needed to reach 90%")
        
        # Check sub-criteria
        if "V1" in results:
            v1_metrics = results["V1"]
            v2_metrics = results[last_model]
            
            checks = {
                "Matra does not regress": v2_metrics["matra_accuracy"] >= v1_metrics["matra_accuracy"] - 1.0,
                "Virama does not regress": v2_metrics["virama_accuracy"] >= v1_metrics["virama_accuracy"] - 1.0,
                "Numerals ≥ 95%": v2_metrics["numeral_accuracy"] >= 95.0,
                "Conjuncts improve": v2_metrics["conjunct_word_accuracy"] >= v1_metrics["conjunct_word_accuracy"] - 1.0,
                "Rare chars improve": v2_metrics["rare_char_accuracy"] >= v1_metrics["rare_char_accuracy"] - 1.0,
                "CER improves": v2_metrics["cer"] <= v1_metrics["cer"],
            }
            
            print(f"\n  Sub-criteria:")
            for check_name, passed in checks.items():
                status = "PASS" if passed else "FAIL"
                print(f"    [{status}] {check_name}")

    # Save full report
    report_path = PROJECT_ROOT / "evaluation" / "v2_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Full report saved: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
