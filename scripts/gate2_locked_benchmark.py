"""GATE 2: LOCKED OCR BENCHMARK FOR IIT BOMBAY INDIC-TROCR V0.0.2

Tests:
1. IIIT Kannada word samples (from smoke_kannada_val.jsonl)
2. Real archival word crops (6 samples from personal_trial)
3. Real archival line crops (7 samples from doc1_lines)

Measures & Saves:
- Ground truth and raw predictions
- CER and WER independently calculated via jiwer
- Exact-match accuracy
- Empty predictions
- Latin hallucinations
- Unique-prediction count
- Most-common-prediction frequency
- Disaggregated word-level and line-level results
"""

import json
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
import jiwer
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "trocr" / "experimental" / "iitb_kannada_v002"

WORD_LEVEL_ARCHIVAL = [
    {"image": PROJECT_ROOT / "training" / "datasets" / "personal_trial" / "crops" / "crop_a_mara.png", "gt": "ಮರ", "subset": "archival_word"},
    {"image": PROJECT_ROOT / "training" / "datasets" / "personal_trial" / "crops" / "crop_o_kothi.png", "gt": "ಕೋತಿ", "subset": "archival_word"},
    {"image": PROJECT_ROOT / "training" / "datasets" / "personal_trial" / "crops" / "crop_p_hannu.png", "gt": "ಹಣ್ಣು", "subset": "archival_word"},
    {"image": PROJECT_ROOT / "training" / "datasets" / "personal_trial" / "crops" / "a_kannada_raw.png", "gt": "ಅ", "subset": "archival_word"},
    {"image": PROJECT_ROOT / "training" / "datasets" / "personal_trial" / "crops" / "o_kannada_raw.png", "gt": "ಒ", "subset": "archival_word"},
    {"image": PROJECT_ROOT / "training" / "datasets" / "personal_trial" / "crops" / "p_kannada_raw.png", "gt": "ಪ", "subset": "archival_word"},
]

LINE_LEVEL_ARCHIVAL = [
    {"image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_04.png", "gt": "125", "subset": "archival_line"},
    {"image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_05.png", "gt": "125 1 ರ", "subset": "archival_line"},
    {"image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_06.png", "gt": "ಶ್ರೀ ಕರಿದಿರ್ N. ಮದನಗೌಡರ ಕಂದಾಯ", "subset": "archival_line"},
    {"image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_07.png", "gt": "ಶ್ರೀ ಕರಿದಿರ್ N. ಮದನಗೌಡರ ತಂದೆ ಲಿಂಗಯ್ಯ 1 ನೇ ಹೆಂಡತಿ ಲಿಂಗಮ್ಮ ಇವರ ಮಕ್ಕಳು", "subset": "archival_line"},
    {"image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_08.png", "gt": "148 0-15 ಇವರ ಹೆಸರಿಗೆ ರಾಜೀನಾಮೆ ಗ್ರಾಮ ಪಂಚಾಯಿತಿ ವತಿಯಿಂದ ಬಂದಂತೆ", "subset": "archival_line"},
    {"image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_09.png", "gt": "ದಿನಾಂಕ 20/09/2005 ರ ಪ್ರಕಾರ ಪಹಣಿ ಮತ್ತು ಹಕ್ಕು ದಾಖಲೆ ನಮೂದಿಸಲಾಗಿದೆ", "subset": "archival_line"},
    {"image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_10.png", "gt": "ಖಾತೆ ಬದಲಾವಣೆ ಮಂಜೂರಾಗಿದೆ ತಹಶೀಲ್ದಾರ್ ಆದೇಶ ಸಂಖ್ಯೆ", "subset": "archival_line"},
]

def load_iiit_samples(manifest_path: Path, max_samples: int = 50) -> List[Dict]:
    samples = []
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    for line in lines[:max_samples]:
        if not line.strip():
            continue
        entry = json.loads(line)
        img_path = Path(entry["image"])
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path
        samples.append({
            "image": img_path,
            "gt": entry["text"].strip(),
            "subset": "iiit_word"
        })
    return samples

def compute_cer(gt: str, pred: str) -> float:
    return jiwer.cer(gt, pred)

def compute_wer(gt: str, pred: str) -> float:
    # If gt has no spaces (single word), standard wer is 0 if equal else 1
    gt_words = gt.strip().split()
    pred_words = pred.strip().split()
    if not gt_words and not pred_words:
        return 0.0
    if not gt_words or not pred_words:
        return 1.0
    return jiwer.wer(gt, pred)

def has_latin(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", text))

def run_locked_benchmark():
    print("=" * 80)
    print("GATE 2: LOCKED OCR BENCHMARK — IIT BOMBAY INDIC-TROCR V0.0.2")
    print(f"Model: {MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load Model and Processor
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    processor = AutoImageProcessor.from_pretrained(str(MODEL_PATH))
    model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_PATH)).to(device)
    model.eval()

    # 2. Build Benchmark Test Suite
    iiit_manifest = PROJECT_ROOT / "training" / "datasets" / "smoke_kannada_val.jsonl"
    iiit_samples = load_iiit_samples(iiit_manifest, max_samples=50)

    all_samples = []
    all_samples.extend(WORD_LEVEL_ARCHIVAL)
    all_samples.extend(LINE_LEVEL_ARCHIVAL)
    all_samples.extend(iiit_samples)

    print(f"\nTotal test samples: {len(all_samples)}")
    print(f"  - Real archival word crops: {len(WORD_LEVEL_ARCHIVAL)}")
    print(f"  - Real archival line crops: {len(LINE_LEVEL_ARCHIVAL)}")
    print(f"  - IIIT Kannada word samples: {len(iiit_samples)}")

    results = []
    preds_counter = Counter()

    for idx, sample in enumerate(all_samples):
        img_path = sample["image"]
        gt_raw = sample["gt"]
        gt = unicodedata.normalize("NFC", gt_raw).strip()
        subset = sample["subset"]

        if not Path(img_path).exists():
            print(f"  [ERROR] Image not found: {img_path}")
            continue

        im = Image.open(img_path).convert("RGB")
        pixel_values = processor(im, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                pixel_values,
                max_length=64,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                decoder_start_token_id=model.config.decoder_start_token_id or tokenizer.bos_token_id,
            )

        pred_raw = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        pred = unicodedata.normalize("NFC", pred_raw).strip()

        preds_counter[pred] += 1

        cer = compute_cer(gt, pred)
        wer = compute_wer(gt, pred)
        is_exact = (pred == gt)
        is_empty = (len(pred) == 0)
        latin_hallucination = has_latin(pred)

        results.append({
            "index": idx + 1,
            "image": str(img_path),
            "subset": subset,
            "gt": gt,
            "pred": pred,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "exact_match": is_exact,
            "is_empty": is_empty,
            "has_latin": latin_hallucination,
        })

    # Summary Metrics across subsets
    def summarize_subset(items, name):
        if not items:
            return {}
        total = len(items)
        cer_list = [it["cer"] for it in items]
        wer_list = [it["wer"] for it in items]
        exact_count = sum(1 for it in items if it["exact_match"])
        empty_count = sum(1 for it in items if it["is_empty"])
        latin_count = sum(1 for it in items if it["has_latin"])

        preds = [it["pred"] for it in items]
        counts = Counter(preds)
        unique_preds = len(counts)
        most_common_pred, most_common_cnt = counts.most_common(1)[0] if counts else ("", 0)
        most_common_freq = (most_common_cnt / total) * 100.0 if total else 0.0

        return {
            "name": name,
            "total_samples": total,
            "mean_cer": round(float(sum(cer_list) / total), 4),
            "mean_wer": round(float(sum(wer_list) / total), 4),
            "exact_match_count": exact_count,
            "exact_match_accuracy": round((exact_count / total) * 100.0, 2),
            "empty_predictions": empty_count,
            "latin_hallucinations": latin_count,
            "unique_predictions": unique_preds,
            "most_common_prediction": most_common_pred,
            "most_common_frequency_pct": round(most_common_freq, 2),
        }

    archival_words = [it for it in results if it["subset"] == "archival_word"]
    archival_lines = [it for it in results if it["subset"] == "archival_line"]
    iiit_words = [it for it in results if it["subset"] == "iiit_word"]
    all_words = [it for it in results if "word" in it["subset"]]

    overall_summary = summarize_subset(results, "Overall")
    archival_word_summary = summarize_subset(archival_words, "Real Archival Words")
    archival_line_summary = summarize_subset(archival_lines, "Real Archival Lines")
    iiit_word_summary = summarize_subset(iiit_words, "IIIT Words")
    all_words_summary = summarize_subset(all_words, "All Words Combined")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 80)
    for s in [overall_summary, archival_word_summary, archival_line_summary, iiit_word_summary]:
        print(f"\n--- {s['name']} (N={s['total_samples']}) ---")
        print(f"  Mean CER: {s['mean_cer'] * 100:.2f}%")
        print(f"  Mean WER: {s['mean_wer'] * 100:.2f}%")
        print(f"  Exact Match Accuracy: {s['exact_match_accuracy']}% ({s['exact_match_count']}/{s['total_samples']})")
        print(f"  Empty Predictions: {s['empty_predictions']}")
        print(f"  Latin Hallucinations: {s['latin_hallucinations']}")
        print(f"  Unique Predictions: {s['unique_predictions']}")
        print(f"  Most Common Pred: '{s['most_common_prediction']}' ({s['most_common_frequency_pct']}%)")

    # Save complete report
    benchmark_report = {
        "model_name": "IIT Bombay Indic-TrOCR v0.0.2",
        "model_path": str(MODEL_PATH),
        "device": device,
        "overall": overall_summary,
        "subsets": {
            "archival_word": archival_word_summary,
            "archival_line": archival_line_summary,
            "iiit_word": iiit_word_summary,
            "all_words": all_words_summary,
        },
        "samples": results,
    }

    out_path = PROJECT_ROOT / "scratch" / "locked_ocr_benchmark_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(benchmark_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nLocked OCR Benchmark Report saved to: {out_path}")

    return benchmark_report

if __name__ == "__main__":
    run_locked_benchmark()
