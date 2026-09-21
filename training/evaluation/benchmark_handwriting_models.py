"""Comprehensive Real Kannada Handwriting Benchmark Suite.

Evaluates and benchmarks multiple TrOCR checkpoints across authentic handwritten samples:
- Baseline TrOCR (`kannada_generalized_v2`)
- Retrained Synthetic Smoke-Test Checkpoint (`smoke_test_best_checkpoint`)
- Future Real-Data Retrained Checkpoints

Evaluates separately:
1. Word-level real crops (isolated handwritten words)
2. Line-level real crops (multi-word archival land record lines)

Metrics computed:
- Character Error Rate (CER) via jiwer
- Word Error Rate (WER) via jiwer
- Exact Match Rate (% identical string matches)
- Repetition / Hallucination rate
- Line-by-line prediction diffs
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import jiwer
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# UTF-8 encoding configuration
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("handwriting_benchmark")

# Ground truth benchmarks
WORD_LEVEL_BENCHMARKS = [
    {"image": PROJECT_ROOT / "training" / "datasets" / "personal_trial" / "crops" / "crop_a_mara.png", "gt": "ಮರ"},
    {"image": PROJECT_ROOT / "training" / "datasets" / "personal_trial" / "crops" / "crop_o_kothi.png", "gt": "ಕೋತಿ"},
    {"image": PROJECT_ROOT / "training" / "datasets" / "personal_trial" / "crops" / "crop_p_hannu.png", "gt": "ಹಣ್ಣು"},
    {"image": PROJECT_ROOT / "training" / "datasets" / "personal_trial" / "crops" / "a_kannada_raw.png", "gt": "ಅ"},
    {"image": PROJECT_ROOT / "training" / "datasets" / "personal_trial" / "crops" / "o_kannada_raw.png", "gt": "ಒ"},
    {"image": PROJECT_ROOT / "training" / "datasets" / "personal_trial" / "crops" / "p_kannada_raw.png", "gt": "ಪ"},
]

LINE_LEVEL_BENCHMARKS = [
    {"image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_04.png", "gt": "125"},
    {"image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_05.png", "gt": "125 1 ರ"},
    {"image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_06.png", "gt": "ಶ್ರೀ ಕರಿದಿರ್ N. ಮದನಗೌಡರ ಕಂದಾಯ"},
    {"image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_07.png", "gt": "ಶ್ರೀ ಕರಿದಿರ್ N. ಮದನಗೌಡರ ತಂದೆ ಲಿಂಗಯ್ಯ 1 ನೇ ಹೆಂಡತಿ ಲಿಂಗಮ್ಮ ಇವರ ಮಕ್ಕಳು"},
    {"image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_08.png", "gt": "148 0-15 ಇವರ ಹೆಸರಿಗೆ ರಾಜೀನಾಮೆ ಗ್ರಾಮ ಪಂಚಾಯಿತಿ ವತಿಯಿಂದ ಬಂದಂತೆ"},
    {"image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_09.png", "gt": "ದಿನಾಂಕ 20/09/2005 ರ ಪ್ರಕಾರ ಪಹಣಿ ಮತ್ತು ಹಕ್ಕು ದಾಖಲೆ ನಮೂದಿಸಲಾಗಿದೆ"},
    {"image": PROJECT_ROOT / "scratch" / "doc1_lines" / "line_10.png", "gt": "ಖಾತೆ ಬದಲಾವಣೆ ಮಂಜೂರಾಗಿದೆ ತಹಶೀಲ್ದಾರ್ ಆದೇಶ ಸಂಖ್ಯೆ"},
]

DEFAULT_MODELS = [
    {
        "name": "baseline_kannada_generalized_v2",
        "path": r"c:\Users\akars\OneDrive\Documents\Land_Record_Models_FULL\models\trocr\kannada_generalized_v2_checkpoints\best_checkpoint",
        "type": "word_trained_legacy_baseline",
    },
    {
        "name": "retrained_smoke_test_v1",
        "path": str(PROJECT_ROOT / "models" / "trocr" / "smoke_test_best_checkpoint"),
        "type": "synthetic_smoke_test_checkpoint",
    },
]


def load_model_safely(model_path: str) -> Optional[Tuple[TrOCRProcessor, VisionEncoderDecoderModel]]:
    """Loads a TrOCR model and processor from checkpoint directory."""
    p = Path(model_path)
    if not p.exists():
        logger.warning(f"Checkpoint path does not exist: {model_path}")
        return None
    try:
        processor = TrOCRProcessor.from_pretrained(str(p))
        model = VisionEncoderDecoderModel.from_pretrained(str(p))
        model.eval()
        return processor, model
    except Exception as exc:
        logger.error(f"Failed to load checkpoint {model_path}: {exc}")
        return None


def run_trocr_prediction(processor: TrOCRProcessor, model: VisionEncoderDecoderModel, image_path: Path) -> str:
    """Runs TrOCR inference on a single image."""
    try:
        img = Image.open(image_path).convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values
        with torch.no_grad():
            gen_ids = model.generate(
                pixel_values,
                max_length=64,
                no_repeat_ngram_size=getattr(model.config, "no_repeat_ngram_size", 3),
                early_stopping=True,
            )
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        return text
    except Exception as exc:
        logger.error(f"Inference error on {image_path}: {exc}")
        return ""


def detect_repetition_or_loop(text: str) -> bool:
    """Detects whether text exhibits token runaway or repetitive loops."""
    if not text:
        return False
    # Check 3-gram repetitions
    words = text.split()
    if len(words) >= 4:
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return True
    # Check character n-gram loops
    for l in [2, 3, 4, 5]:
        for i in range(len(text) - l * 3):
            sub = text[i:i + l]
            if sub * 3 in text:
                return True
    return False


def evaluate_model_on_dataset(
    model_name: str,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    benchmarks: List[Dict[str, Any]],
    granularity: str,
) -> Dict[str, Any]:
    """Runs evaluation across benchmark samples and computes metrics."""
    preds: List[str] = []
    gts: List[str] = []
    sample_results: List[Dict[str, Any]] = []
    exact_matches = 0
    repetitions_detected = 0

    for item in benchmarks:
        img_p = Path(item["image"])
        gt = item["gt"]
        if not img_p.exists():
            continue

        pred = run_trocr_prediction(processor, model, img_p)
        preds.append(pred)
        gts.append(gt)

        is_exact = (pred == gt)
        if is_exact:
            exact_matches += 1

        has_loop = detect_repetition_or_loop(pred)
        if has_loop:
            repetitions_detected += 1

        sample_cer = jiwer.cer(gt, pred) if pred else 1.0
        sample_wer = jiwer.wer(gt, pred) if pred else 1.0

        sample_results.append({
            "image": img_p.name,
            "gt": gt,
            "prediction": pred,
            "cer": round(sample_cer, 4),
            "wer": round(sample_wer, 4),
            "exact_match": is_exact,
            "has_repetition_loop": has_loop,
        })

    if not preds:
        return {"error": "no_valid_samples"}

    overall_cer = jiwer.cer(gts, preds)
    overall_wer = jiwer.wer(gts, preds)
    exact_ratio = exact_matches / len(preds)

    return {
        "granularity": granularity,
        "sample_count": len(preds),
        "overall_cer": round(overall_cer, 4),
        "overall_wer": round(overall_wer, 4),
        "exact_match_rate": round(exact_ratio, 4),
        "repetition_loop_count": repetitions_detected,
        "samples": sample_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Real Kannada Handwriting Benchmark")
    parser.add_argument("--additional-model", type=str, default=None, help="Path to an extra checkpoint directory")
    args = parser.parse_args()

    models_to_test = list(DEFAULT_MODELS)
    if args.additional_model:
        models_to_test.append({
            "name": Path(args.additional_model).name,
            "path": args.additional_model,
            "type": "custom_checkpoint",
        })

    print("=" * 80)
    print("KANNADA HANDWRITING OCR STANDARDIZED BENCHMARK")
    print("=" * 80)

    summary_results: Dict[str, Any] = {}

    for m_info in models_to_test:
        m_name = m_info["name"]
        m_path = m_info["path"]
        print(f"\nEvaluating Checkpoint: {m_name}")
        print(f"  Path: {m_path}")

        loaded = load_model_safely(m_path)
        if not loaded:
            print("  [SKIPPED] Model could not be loaded.")
            continue
        proc, mod = loaded

        # Word-level evaluation
        word_metrics = evaluate_model_on_dataset(m_name, proc, mod, WORD_LEVEL_BENCHMARKS, granularity="word")
        print(f"  [Word-Level] CER = {word_metrics['overall_cer']:.2%}, WER = {word_metrics['overall_wer']:.2%}, Exact = {word_metrics['exact_match_rate']:.2%}, Loops = {word_metrics['repetition_loop_count']}")

        # Line-level evaluation
        line_metrics = evaluate_model_on_dataset(m_name, proc, mod, LINE_LEVEL_BENCHMARKS, granularity="line")
        print(f"  [Line-Level] CER = {line_metrics['overall_cer']:.2%}, WER = {line_metrics['overall_wer']:.2%}, Exact = {line_metrics['exact_match_rate']:.2%}, Loops = {line_metrics['repetition_loop_count']}")

        summary_results[m_name] = {
            "model_type": m_info["type"],
            "path": m_path,
            "word_level": word_metrics,
            "line_level": line_metrics,
        }

    # Save benchmark report
    out_file = PROJECT_ROOT / "training" / "evaluation" / "handwriting_benchmark_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Standardized benchmark report saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
