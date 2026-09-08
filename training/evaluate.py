"""Evaluation Entry Point for Handwriting OCR Models.

Computes exact CER and WER metrics globally and per-language across test manifests.

Usage:
    # Evaluate a model/checkpoint on a test manifest:
    python training/evaluate.py --config training/configs/kannada.yaml --manifest training/datasets/test_kannada.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure UTF-8 output encoding for terminal display of Indic characters
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import (
    HandwritingDataset,
    HandwritingTrainer,
    TrainingConfig,
    evaluate_predictions,
    load_training_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Handwriting Recognition Models.")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/kannada.yaml",
        help="Path to training configuration YAML file.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to test dataset JSONL manifest (overrides config test_manifest).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to saved model checkpoint directory.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Filter evaluation to a specific language (e.g. 'kannada', 'telugu').",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Optional path to save JSON evaluation report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    print("=" * 70)
    print("  LAND RECORD DIGITIZATION - OCR EVALUATION PIPELINE")
    print("=" * 70)

    config = TrainingConfig.from_yaml(config_path) if config_path.exists() else TrainingConfig()

    manifest_path_str = args.manifest or config.dataset.test_manifest or config.dataset.val_manifest
    if not manifest_path_str:
        print("[!] Error: No evaluation manifest specified. Provide --manifest path.")
        sys.exit(1)

    manifest_path = Path(manifest_path_str)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    if not manifest_path.exists():
        print(f"\n[!] Notice: Test manifest not found at: {manifest_path}")
        print("    To evaluate, create a JSONL manifest formatted as:")
        print('    {"image": "path/to/crop.png", "text": "transcription", "language": "kannada", "script": "Kannada"}')
        print("\n[EVALUATION PIPELINE READY] Evaluation utilities configured successfully.")
        return

    print(f"Loading evaluation dataset from: {manifest_path}")
    dataset = HandwritingDataset(
        manifest_path=manifest_path,
        root_dir=PROJECT_ROOT,
        validate_images=False,
    )

    if args.language:
        dataset = dataset.filter_by_language(args.language)
        print(f"Filtered by language '{args.language}': {len(dataset)} samples remaining.")

    print(f"Total Evaluation Samples: {len(dataset)}")
    print(f"Dataset Languages       : {dataset.get_languages()}")

    # Collect ground truths
    references = [s.text for s in dataset.samples]
    languages = [s.language for s in dataset.samples]

    # Run inference or generate predictions
    model_name = config.model.model_name_or_path
    if args.checkpoint:
        model_name = args.checkpoint
        try:
            meta, _ = load_training_checkpoint(args.checkpoint)
            print(f"Loaded checkpoint metadata from: {args.checkpoint} (Epoch: {meta.get('epoch')}, Step: {meta.get('step')})")
        except Exception as e:
            print(f"Notice: Could not load metadata: {e}")

    print(f"\nEvaluating predictions using model: {model_name}...")

    # Use TrocrHandwritingRecognizer from src.handwriting if available
    from src.handwriting.trocr_recognizer import TrocrHandwritingRecognizer

    recognizer = TrocrHandwritingRecognizer(
        model_name_or_path=model_name,
        auto_load=True,
    )

    hypotheses = []
    print("Running OCR inference on evaluation batch...")
    for idx, sample in enumerate(dataset.samples):
        if not sample.image_path.exists():
            hypotheses.append("")
            continue
        try:
            res = recognizer.recognize_handwriting(sample.image_path)
            hypotheses.append(res.text)
        except Exception as exc:
            print(f"Warning: Inference error on {sample.image_path}: {exc}")
            hypotheses.append("")

    # Calculate metrics
    report = evaluate_predictions(
        references=references,
        hypotheses=hypotheses,
        languages=languages,
    )

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS REPORT")
    print("=" * 70)
    print(f"  Total Samples Evaluated : {report.total_samples}")
    print(f"  Total Characters        : {report.total_reference_characters}")
    print(f"  Total Words             : {report.total_reference_words}")
    print(f"  Overall CER             : {report.overall_cer:.4f} ({report.overall_cer * 100:.2f}%)")
    print(f"  Overall WER             : {report.overall_wer:.4f} ({report.overall_wer * 100:.2f}%)")
    print("\n  Per-Language Breakdown:")
    for lang, metrics in report.per_language.items():
        print(f"  - [{lang.upper()}] Samples: {metrics['sample_count']} | CER: {metrics['cer']:.4f} | WER: {metrics['wer']:.4f}")
    print("=" * 70)

    if args.output_report:
        out_path = Path(args.output_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
