"""Real Inference Test Script for Fine-Tuned Kannada TrOCR Checkpoint.

Evaluates at least 10 real Kannada handwritten word crops from the untouched
IIIT validation dataset using the saved production checkpoint:
models/trocr/kannada_full_checkpoints/best_checkpoint/

Reports:
- Ground Truth (Kannada)
- Predicted Kannada Text
- Confidence Score (step-softmax mean)
- Character Error Rate (CER via Levenshtein distance)
- Inference Time (ms)
- Aggregate CER across the evaluated batch
"""

import json
import os
from pathlib import Path
import sys
import time

# Ensure UTF-8 output encoding for terminal display of Indic characters on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.handwriting.trocr_recognizer import TrocrHandwritingRecognizer
from src.training.evaluate import compute_cer, levenshtein_distance


def main():
    checkpoint_dir = PROJECT_ROOT / "models" / "trocr" / "kannada_full_checkpoints" / "best_checkpoint"
    val_manifest = PROJECT_ROOT / "training" / "datasets" / "iiit_kannada_val.jsonl"

    print("=" * 95, flush=True)
    print("  LAND RECORD DIGITIZATION — REAL KANNADA CHECKPOINT INFERENCE TEST", flush=True)
    print("=" * 95, flush=True)
    print(f"  Checkpoint Path : {checkpoint_dir}", flush=True)
    print(f"  Validation Set  : {val_manifest}", flush=True)

    if not checkpoint_dir.exists():
        print(f"\n[!] Error: Checkpoint directory not found at {checkpoint_dir}", flush=True)
        sys.exit(1)

    if not val_manifest.exists():
        print(f"\n[!] Error: Validation manifest not found at {val_manifest}", flush=True)
        sys.exit(1)

    # 1. Load samples from untouched validation set
    samples = []
    with open(val_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    rel_img = data.get("image") or data.get("image_path")
                    full_img_path = PROJECT_ROOT / rel_img
                    if full_img_path.exists() and data.get("text", "").strip():
                        samples.append({
                            "image_path": full_img_path,
                            "rel_path": rel_img,
                            "text": data["text"].strip(),
                        })
                except Exception:
                    continue
            if len(samples) >= 12:
                break

    if len(samples) < 10:
        print(f"[!] Warning: Only found {len(samples)} valid validation samples with existing images.", flush=True)

    print(f"  Loaded {len(samples)} authentic validation crop samples.", flush=True)
    print("\n  Initializing TrocrHandwritingRecognizer on production checkpoint...", flush=True)

    # 2. Instantiate and load model
    load_start = time.perf_counter()
    recognizer = TrocrHandwritingRecognizer(
        model_name_or_path=str(checkpoint_dir),
        language="kannada",
        script="Kannada",
        auto_load=True,
    )
    load_time = time.perf_counter() - load_start

    print(f"  Model Loaded in : {load_time:.2f}s | Device: {recognizer.device} | Version: {recognizer.model_version}", flush=True)
    print("=" * 95, flush=True)
    print(f" {'#':<3} | {'Ground Truth (Kannada)':<22} | {'Predicted Text':<22} | {'Conf':<6} | {'CER':<6} | {'Time (ms)':<9} | {'Match'}", flush=True)
    print("-" * 95, flush=True)

    total_ref_chars = 0
    total_edit_distance = 0
    inference_times = []
    results = []

    for idx, sample in enumerate(samples, start=1):
        gt_text = sample["text"]
        img_path = sample["image_path"]

        # Run real inference
        ocr_res = recognizer.recognize_handwriting(image=img_path)
        pred_text = ocr_res.text

        # Compute metrics
        sample_cer = compute_cer(gt_text, pred_text)
        edit_dist = levenshtein_distance(list(gt_text), list(pred_text))
        total_ref_chars += len(gt_text)
        total_edit_distance += edit_dist

        meta = ocr_res.metadata.get("metadata", {})
        inf_time_ms = meta.get("inference_time_ms", 0.0)
        inference_times.append(inf_time_ms)
        conf_val = ocr_res.confidence
        conf_str = f"{conf_val:.3f}" if conf_val is not None else "N/A"
        match_str = "EXACT" if pred_text == gt_text else ("PARTIAL" if sample_cer < 0.50 else "MISMATCH")

        print(f" {idx:<3} | {gt_text:<22} | {pred_text:<22} | {conf_str:<6} | {sample_cer:<6.3f} | {inf_time_ms:<9.1f} | {match_str}", flush=True)

        results.append({
            "sample_index": idx,
            "image": sample["rel_path"],
            "ground_truth": gt_text,
            "predicted": pred_text,
            "confidence": conf_val,
            "cer": sample_cer,
            "inference_time_ms": inf_time_ms,
            "match": match_str,
        })

    # 3. Aggregate metrics
    aggregate_cer = (total_edit_distance / total_ref_chars) if total_ref_chars > 0 else 0.0
    avg_inf_time = sum(inference_times) / len(inference_times) if inference_times else 0.0

    print("=" * 95, flush=True)
    print("  SUMMARY AUDIT REPORT", flush=True)
    print("=" * 95, flush=True)
    print(f"  Samples Evaluated   : {len(samples)}", flush=True)
    print(f"  Total Characters    : {total_ref_chars}", flush=True)
    print(f"  Total Edit Distance : {total_edit_distance}", flush=True)
    print(f"  Aggregate CER       : {aggregate_cer:.4f} ({aggregate_cer * 100:.2f}%)", flush=True)
    print(f"  Mean Inference Time : {avg_inf_time:.2f} ms / crop", flush=True)
    print(f"  Active Device       : {recognizer.device}", flush=True)
    print("=" * 95, flush=True)

    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.exit(1)
