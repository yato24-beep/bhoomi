"""Real-Data TrOCR Evaluation Experiment.

Evaluates the smoke_test_best_checkpoint on the 13 locally available real
archival Kannada handwriting crops from doc1.jpeg and personal_trial.

This is an EVALUATION-ONLY run — 13 samples is not enough for training.
Training is BLOCKED pending IIIT-INDIC-HW-WORDS image download.

Reports:
  - Per-sample: decoded text, CER, WER, exact match
  - Aggregate: mean CER, WER, exact match rate
  - Qualitative: repetition/hallucination examples
  - Checkpoint path used

Does NOT promote any checkpoint to production.
"""

import json
import logging
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8")

import jiwer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("real_data_trocr_experiment")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def is_repetitive(text: str, threshold: float = 0.40) -> bool:
    """Detects autoregressive repetition loops: a token repeated >threshold fraction of total."""
    if not text:
        return False
    tokens = text.split()
    if len(tokens) < 3:
        return False
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    max_freq = max(counts.values())
    return max_freq / len(tokens) > threshold


def load_manifest(path: Path) -> List[Dict]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        samples.append(d)
    return samples


def run_experiment(
    checkpoint_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
):
    # ------------------------------------------------------------------ #
    # Checkpoint selection                                                #
    # ------------------------------------------------------------------ #
    candidate_checkpoints = [
        checkpoint_path,
        "models/trocr/kannada_retrained_combined/best_checkpoint",
        "models/trocr/smoke_test_best_checkpoint",
        r"C:\Users\akars\Downloads\kannada_trocr_best_checkpoint\best_checkpoint",
        r"C:\Users\akars\OneDrive\Documents\Land_Record_Models_FULL\models\trocr\kannada_generalized_v2_checkpoints\best_checkpoint",
    ]
    local_ckpt = None
    for c in candidate_checkpoints:
        if c and Path(c).exists():
            local_ckpt = Path(c)
            break

    if not local_ckpt:
        logger.error(
            f"No valid checkpoint found in candidates: {candidate_checkpoints}\n"
            "Specify a checkpoint path with --checkpoint."
        )
        sys.exit(1)

    logger.info(f"Loading checkpoint: {local_ckpt}")

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(str(local_ckpt))
    model = VisionEncoderDecoderModel.from_pretrained(str(local_ckpt))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Model loaded on {device}")

    # ------------------------------------------------------------------ #
    # Load manifest                                                       #
    # ------------------------------------------------------------------ #
    if manifest_path and Path(manifest_path).exists():
        manifest_file = Path(manifest_path)
    elif Path("training/datasets/combined_handwriting/held_out_archival_benchmark.jsonl").exists():
        manifest_file = Path("training/datasets/combined_handwriting/held_out_archival_benchmark.jsonl")
    else:
        manifest_file = Path("training/datasets/real_handwriting/archival_lines/test.jsonl")

    all_samples = load_manifest(manifest_file)
    logger.info(f"Loaded {len(all_samples)} samples to evaluate from: {manifest_file}")


    # ------------------------------------------------------------------ #
    # Build Latin token suppression list                                  #
    # ------------------------------------------------------------------ #
    import re
    latin_pattern = re.compile(r"[a-zA-Z]")
    suppress_ids = [
        tid
        for token, tid in processor.tokenizer.get_vocab().items()
        if tid not in processor.tokenizer.all_special_ids and latin_pattern.search(token)
    ]
    logger.info(f"Built Latin token suppression list: {len(suppress_ids)} token IDs")

    # ------------------------------------------------------------------ #
    # Per-sample evaluation                                               #
    # ------------------------------------------------------------------ #
    results = []
    total_cer_raw, total_wer_raw = 0.0, 0.0
    total_cer_supp, total_wer_supp = 0.0, 0.0
    exact_matches_raw, exact_matches_supp = 0, 0
    repetition_count_raw, repetition_count_supp = 0, 0

    print("\n" + "=" * 80)
    print("REAL-DATA TROCR EVALUATION — RAW vs LATIN TOKEN SUPPRESSION")
    print(f"Checkpoint: {local_ckpt}")
    print(f"Suppression token count: {len(suppress_ids)}")
    print("=" * 80)

    for i, sample in enumerate(all_samples):
        img_path = Path(sample["image"])
        ref_text = nfc(sample["text"])
        split = sample.get("split", "unknown")

        # Check image exists
        if not img_path.exists():
            logger.warning(f"[{i}] Image missing: {img_path} — SKIPPING")
            continue

        # Load and process image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"[{i}] Cannot open {img_path}: {e}")
            continue

        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

        # 1. Unconstrained generation (baseline)
        with torch.no_grad():
            gen_raw = model.generate(
                pixel_values,
                max_length=64,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        pred_raw = nfc(processor.batch_decode(gen_raw, skip_special_tokens=True)[0])

        # 2. Constrained generation (Latin tokens suppressed)
        with torch.no_grad():
            gen_supp = model.generate(
                pixel_values,
                max_length=64,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
                suppress_tokens=suppress_ids,
            )
        pred_supp = nfc(processor.batch_decode(gen_supp, skip_special_tokens=True)[0])

        # Metrics for raw
        ref_for_metric = ref_text if ref_text else " "
        cer_raw = jiwer.cer(ref_for_metric, pred_raw) if ref_for_metric else 1.0
        wer_raw = jiwer.wer(ref_for_metric, pred_raw) if ref_for_metric else 1.0
        exact_raw = pred_raw == ref_text
        repetitive_raw = is_repetitive(pred_raw)

        # Metrics for suppressed
        cer_supp = jiwer.cer(ref_for_metric, pred_supp) if ref_for_metric else 1.0
        wer_supp = jiwer.wer(ref_for_metric, pred_supp) if ref_for_metric else 1.0
        exact_supp = pred_supp == ref_text
        repetitive_supp = is_repetitive(pred_supp)

        total_cer_raw += cer_raw
        total_wer_raw += wer_raw
        if exact_raw:
            exact_matches_raw += 1
        if repetitive_raw:
            repetition_count_raw += 1

        total_cer_supp += cer_supp
        total_wer_supp += wer_supp
        if exact_supp:
            exact_matches_supp += 1
        if repetitive_supp:
            repetition_count_supp += 1

        print(f"\n[{i:02d}] split={split} | {img_path.name}")
        print(f"  REF        : {ref_text!r}")
        print(f"  PRED (RAW) : {pred_raw!r}  (CER={cer_raw:.3f}, WER={wer_raw:.3f})")
        print(f"  PRED (SUPP): {pred_supp!r}  (CER={cer_supp:.3f}, WER={wer_supp:.3f})")

        results.append({
            "index": i,
            "split": split,
            "image": str(img_path),
            "reference": ref_text,
            "predicted_raw": pred_raw,
            "cer_raw": round(cer_raw, 4),
            "wer_raw": round(wer_raw, 4),
            "predicted_suppressed": pred_supp,
            "cer_suppressed": round(cer_supp, 4),
            "wer_suppressed": round(wer_supp, 4),
            "exact_match_suppressed": exact_supp,
            "is_repetitive": repetitive_supp,
        })

    # ------------------------------------------------------------------ #
    # Aggregate metrics                                                   #
    # ------------------------------------------------------------------ #
    n = max(len(results), 1)
    mean_cer_raw = total_cer_raw / n
    mean_wer_raw = total_wer_raw / n
    mean_cer_supp = total_cer_supp / n
    mean_wer_supp = total_wer_supp / n

    print("\n" + "=" * 80)
    print("AGGREGATE COMPARISON RESULTS (13 Real Archival Crops)")
    print(f"  Evaluated samples   : {len(results)} / {len(all_samples)}")
    print(f"  Baseline Mean CER   : {mean_cer_raw * 100:.2f}%")
    print(f"  Suppressed Mean CER : {mean_cer_supp * 100:.2f}%")
    print(f"  CER Delta           : {(mean_cer_supp - mean_cer_raw) * 100:+.2f}%")
    print(f"  Baseline Mean WER   : {mean_wer_raw * 100:.2f}%")
    print(f"  Suppressed Mean WER : {mean_wer_supp * 100:.2f}%")
    print(f"  WER Delta           : {(mean_wer_supp - mean_wer_raw) * 100:+.2f}%")
    print(f"  Exact Match Rate    : {exact_matches_supp / n * 100:.1f}%")
    print(f"  Repetition Loops    : {repetition_count_supp}")
    print(f"  Checkpoint          : {local_ckpt}")
    print("=" * 80)

    # ------------------------------------------------------------------ #
    # IIIT data availability report                                       #
    # ------------------------------------------------------------------ #
    iiit_image_dir = Path("external_datasets/iiit_indic_hw_words/train/images")
    print("\n--- IIIT-INDIC-HW-WORDS Availability ---")
    if iiit_image_dir.exists():
        n_imgs = len(list(iiit_image_dir.glob("*.jpg")))
        print(f"  IIIT train images found: {n_imgs}")
        print("  Status: AVAILABLE — run training/train_trocr_kannada_gpu.py on GPU")
    else:
        print("  IIIT images: NOT ON DISK")
        print("  Status: BLOCKED — training requires IIIT-INDIC-HW-WORDS download")
        print("  Download: https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data")
        print("  Registration required. After download, place at:")
        print("    external_datasets/iiit_indic_hw_words/train/images/*.jpg")
        print("    (and val/, test/ equivalents)")

    # ------------------------------------------------------------------ #
    # Save report                                                         #
    # ------------------------------------------------------------------ #
    out_dir = Path("tests/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(local_ckpt),
        "device": device,
        "total_samples": len(all_samples),
        "evaluated_samples": len(results),
        "baseline_mean_cer": round(mean_cer_raw, 4),
        "baseline_mean_wer": round(mean_wer_raw, 4),
        "suppressed_mean_cer": round(mean_cer_supp, 4),
        "suppressed_mean_wer": round(mean_wer_supp, 4),
        "cer_delta": round(mean_cer_supp - mean_cer_raw, 4),
        "wer_delta": round(mean_wer_supp - mean_wer_raw, 4),
        "exact_match_rate": round(exact_matches_supp / n * 100, 2),
        "repetition_loop_count": repetition_count_supp,
        "iiit_images_available": iiit_image_dir.exists(),
        "training_status": "BLOCKED_NO_IIIT_IMAGES" if not iiit_image_dir.exists() else "READY_FOR_GPU",
        "per_sample": results,
    }
    out_file = out_dir / "real_data_handwriting_eval_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {out_file}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate TrOCR on held-out handwriting benchmark")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint folder")
    parser.add_argument("--manifest", type=str, default=None, help="Path to evaluation manifest jsonl")
    args = parser.parse_args()
    run_experiment(checkpoint_path=args.checkpoint, manifest_path=args.manifest)

