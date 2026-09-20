"""Benchmark script comparing Method A (Whole-Line TrOCR) vs Method B (Line Word Decomposition -> Word TrOCR).

Evaluates archival lines line_04 through line_10 on:
- CER (Character Error Rate)
- WER (Word Error Rate)
- Exact Match
- Empty Predictions
- Latin Hallucinations
- Segmentation Failures
- Latency (ms)

Strict constraints:
- NO training or weight modifications
- Preserves locked benchmark baseline
"""

import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas import BoundingBox
from src.handwriting.trocr_recognizer import get_iitb_kannada_recognizer
from src.preprocessing.line_word_decomposer import LineWordDecomposer

# Archival line dataset with verbatim ground truth
ARCHIVAL_LINES = [
    {
        "id": "line_04",
        "image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_04.png",
        "gt": "125",
    },
    {
        "id": "line_05",
        "image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_05.png",
        "gt": "125 1 ರ",
    },
    {
        "id": "line_06",
        "image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_06.png",
        "gt": "ಶ್ರೀ ಕರಿದಿರ್ N. ಮದನಗೌಡರ ಕಂದಾಯ",
    },
    {
        "id": "line_07",
        "image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_07.png",
        "gt": "ಶ್ರೀ ಕರಿದಿರ್ N. ಮದನಗೌಡರ ತಂದೆ ಲಿಂಗಯ್ಯ 1 ನೇ ಹೆಂಡತಿ ಲಿಂಗಮ್ಮ ಇವರ ಮಕ್ಕಳು",
    },
    {
        "id": "line_08",
        "image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_08.png",
        "gt": "148 0-15 ಇವರ ಹೆಸರಿಗೆ ರಾಜೀನಾಮೆ ಗ್ರಾಮ ಪಂಚಾಯಿತಿ ವತಿಯಿಂದ ಬಂದಂತೆ",
    },
    {
        "id": "line_09",
        "image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_09.png",
        "gt": "ದಿನಾಂಕ 20/09/2005 ರ ಪ್ರಕಾರ ಪಹಣಿ ಮತ್ತು ಹಕ್ಕು ದಾಖಲೆ ನಮೂದಿಸಲಾಗಿದೆ",
    },
    {
        "id": "line_10",
        "image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_10.png",
        "gt": "ಖಾತೆ ಬದಲಾವಣೆ ಮಂಜೂರಾಗಿದೆ ತಹಶೀಲ್ದಾರ್ ಆದೇಶ ಸಂಖ್ಯೆ",
    },
]


def levenshtein_distance(s1: Sequence, s2: Sequence) -> int:
    """Computes minimum edit distance between two sequences."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return round(float(levenshtein_distance(pred, gt) / max(1, len(gt))), 4)


def compute_wer(pred: str, gt: str) -> float:
    gt_words = gt.strip().split()
    pred_words = pred.strip().split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    return round(float(levenshtein_distance(pred_words, gt_words) / max(1, len(gt_words))), 4)


def count_latin_hallucinations(pred: str, gt: str) -> int:
    """Counts unexpected Latin characters that do not appear in ground truth."""
    gt_latin = set(re.findall(r"[A-Za-z]", gt))
    pred_latin = re.findall(r"[A-Za-z]", pred)
    unexpected = [ch for ch in pred_latin if ch not in gt_latin]
    return len(unexpected)


def run_benchmark():
    print("=" * 80)
    print("HANDWRITTEN ARCHIVAL LINE OCR BENCHMARK: METHOD A vs METHOD B")
    print("Model: IIT Bombay Indic-TrOCR v0.0.2 (FROZEN - NO TRAINING)")
    print("=" * 80)

    recognizer = get_iitb_kannada_recognizer(auto_load=True)
    decomposer = LineWordDecomposer(enable_debug_viz=True, debug_output_dir=PROJECT_ROOT / "scratch" / "line_debug_viz")

    results_a = []
    results_b = []

    for item in ARCHIVAL_LINES:
        line_id = item["id"]
        img_path = item["image"]
        gt = item["gt"]

        if not img_path.exists():
            print(f"Skipping {line_id}: file not found at {img_path}")
            continue

        pil_img = Image.open(img_path).convert("RGB")

        # -------------------------------------------------------------
        # METHOD A: Whole-Line Direct TrOCR
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        ocr_res_a = recognizer.recognize_handwriting(pil_img)
        t_ms_a = round((time.perf_counter() - t0) * 1000.0, 2)
        pred_a = (ocr_res_a.text or "").strip()

        cer_a = compute_cer(pred_a, gt)
        wer_a = compute_wer(pred_a, gt)
        em_a = (pred_a == gt)
        empty_a = (len(pred_a) == 0)
        latin_halluc_a = count_latin_hallucinations(pred_a, gt)

        res_entry_a = {
            "id": line_id,
            "ground_truth": gt,
            "prediction": pred_a,
            "cer": cer_a,
            "wer": wer_a,
            "exact_match": em_a,
            "is_empty": empty_a,
            "latin_hallucinations": latin_halluc_a,
            "segmentation_failure": False,
            "failure_reason": None,
            "latency_ms": t_ms_a,
        }
        results_a.append(res_entry_a)

        # -------------------------------------------------------------
        # METHOD B: Line -> Word Decomposition -> Word TrOCR
        # -------------------------------------------------------------
        t0_b = time.perf_counter()
        decomp = decomposer.decompose(pil_img, line_id=line_id)

        if not decomp.is_valid:
            # Segmentation failure fallback: preserve original line crop, needs_review=True
            t_ms_b = round((time.perf_counter() - t0_b) * 1000.0, 2)
            res_entry_b = {
                "id": line_id,
                "ground_truth": gt,
                "prediction": "",  # Honest fallback: unverified text suppressed
                "cer": 1.0,
                "wer": 1.0,
                "exact_match": False,
                "is_empty": True,
                "latin_hallucinations": 0,
                "segmentation_failure": True,
                "failure_reason": decomp.failure_reason,
                "diagnostics": decomp.diagnostics,
                "word_count": 0,
                "latency_ms": t_ms_b,
                "needs_review": True,
            }
        else:
            word_crops = [c.crop for c in decomp.word_candidates]
            word_preds = recognizer.recognize_batch(word_crops)
            pred_words = [p.text.strip() for p in word_preds if p.text.strip()]
            reconstructed_text = " ".join(pred_words)
            t_ms_b = round((time.perf_counter() - t0_b) * 1000.0, 2)

            cer_b = compute_cer(reconstructed_text, gt)
            wer_b = compute_wer(reconstructed_text, gt)
            em_b = (reconstructed_text == gt)
            empty_b = (len(reconstructed_text) == 0)
            latin_halluc_b = count_latin_hallucinations(reconstructed_text, gt)

            res_entry_b = {
                "id": line_id,
                "ground_truth": gt,
                "prediction": reconstructed_text,
                "per_word_predictions": [
                    {"index": i, "text": p.text, "confidence": p.confidence}
                    for i, p in enumerate(word_preds)
                ],
                "cer": cer_b,
                "wer": wer_b,
                "exact_match": em_b,
                "is_empty": empty_b,
                "latin_hallucinations": latin_halluc_b,
                "segmentation_failure": False,
                "failure_reason": None,
                "diagnostics": decomp.diagnostics,
                "word_count": len(decomp.word_candidates),
                "latency_ms": t_ms_b,
                "needs_review": False,
            }
        results_b.append(res_entry_b)

    # -------------------------------------------------------------
    # Aggregate Metrics Calculation
    # -------------------------------------------------------------
    mean_cer_a = round(float(np.mean([r["cer"] for r in results_a])), 4)
    mean_wer_a = round(float(np.mean([r["wer"] for r in results_a])), 4)
    exact_match_a = sum(1 for r in results_a if r["exact_match"])
    empty_a_count = sum(1 for r in results_a if r["is_empty"])
    total_latin_a = sum(r["latin_hallucinations"] for r in results_a)
    mean_latency_a = round(float(np.mean([r["latency_ms"] for r in results_a])), 2)

    mean_cer_b = round(float(np.mean([r["cer"] for r in results_b])), 4)
    mean_wer_b = round(float(np.mean([r["wer"] for r in results_b])), 4)
    exact_match_b = sum(1 for r in results_b if r["exact_match"])
    empty_b_count = sum(1 for r in results_b if r["is_empty"])
    total_latin_b = sum(r["latin_hallucinations"] for r in results_b)
    seg_failures_b = sum(1 for r in results_b if r["segmentation_failure"])
    mean_latency_b = round(float(np.mean([r["latency_ms"] for r in results_b])), 2)

    print("\n" + "=" * 80)
    print(f"{'Line ID':<10} | {'Method A Pred':<30} | {'Method B Pred':<30}")
    print("-" * 80)
    for ra, rb in zip(results_a, results_b):
        p_a = (ra["prediction"][:27] + "...") if len(ra["prediction"]) > 27 else ra["prediction"]
        p_b = (rb["prediction"][:27] + "...") if len(rb["prediction"]) > 27 else rb["prediction"]
        if rb["segmentation_failure"]:
            p_b = f"[SEG FAIL: {rb['failure_reason'][:15]}...]"
        print(f"{ra['id']:<10} | {p_a:<30} | {p_b:<30}")

    print("\n" + "=" * 80)
    print("AGGREGATE METRICS SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Method A (Whole-Line)':<22} | {'Method B (Decomposition)':<22}")
    print("-" * 80)
    print(f"{'Mean CER':<30} | {mean_cer_a:<22.4f} | {mean_cer_b:<22.4f}")
    print(f"{'Mean WER':<30} | {mean_wer_a:<22.4f} | {mean_wer_b:<22.4f}")
    print(f"{'Exact Match Count':<30} | {f'{exact_match_a}/{len(results_a)}':<22} | {f'{exact_match_b}/{len(results_b)}':<22}")
    print(f"{'Empty Predictions':<30} | {empty_a_count:<22} | {empty_b_count:<22}")
    print(f"{'Latin Hallucinations':<30} | {total_latin_a:<22} | {total_latin_b:<22}")
    print(f"{'Segmentation Failures':<30} | {'0 (N/A)':<22} | {seg_failures_b:<22}")
    print(f"{'Mean Latency (ms)':<30} | {mean_latency_a:<22.1f} | {mean_latency_b:<22.1f}")
    print("=" * 80)

    report_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "IITB Indic-TrOCR v0.0.2 (kannada)",
        "sample_count": len(results_a),
        "method_a_whole_line": {
            "mean_cer": mean_cer_a,
            "mean_wer": mean_wer_a,
            "exact_match_count": exact_match_a,
            "empty_count": empty_a_count,
            "latin_hallucinations": total_latin_a,
            "mean_latency_ms": mean_latency_a,
            "per_sample": results_a,
        },
        "method_b_line_word_decomposition": {
            "mean_cer": mean_cer_b,
            "mean_wer": mean_wer_b,
            "exact_match_count": exact_match_b,
            "empty_count": empty_b_count,
            "latin_hallucinations": total_latin_b,
            "segmentation_failures": seg_failures_b,
            "mean_latency_ms": mean_latency_b,
            "per_sample": results_b,
        },
    }

    out_path = PROJECT_ROOT / "scratch" / "line_decomposition_benchmark_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetailed benchmark report saved to: {out_path}")


if __name__ == "__main__":
    run_benchmark()
