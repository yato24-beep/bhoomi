"""CLI script for importing official IIIT Indic Handwriting Datasets.

Parses official ground-truth files, resolves and validates image paths,
preserves Unicode transcriptions, generates JSONL manifests for train/val/test splits,
runs full manifest validation, and performs dataset integrity verification.

Usage:
    # Import all splits (train, val, test) for Kannada
    python training/scripts/import_iiit_indic_hw.py \
        --dataset-root "external_datasets/iiit_indic_hw_words" \
        --language kannada \
        --script Kannada \
        --all

    # Import single split
    python training/scripts/import_iiit_indic_hw.py \
        --dataset-root "external_datasets/iiit_indic_hw_words" \
        --language kannada \
        --split train
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure UTF-8 output encoding for terminal display of Indic characters
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.data_preparation import validate_manifest
from src.training.importers.iiit_indic_hw import IIITIndicHWImporter, ImportReport
from src.training.dataset import MultilingualHandwritingSample


def parse_args():
    parser = argparse.ArgumentParser(
        description="Official IIIT Indic Handwriting Dataset Importer"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="external_datasets/iiit_indic_hw_words",
        help="Path to the extracted IIIT Indic dataset root containing train/, val/, test/ folders.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="kannada",
        help="Language name (e.g. kannada, telugu, tamil, hindi, malayalam).",
    )
    parser.add_argument(
        "--script",
        type=str,
        default=None,
        help="Script name (e.g. Kannada, Telugu, Tamil, Devanagari). Inferred if omitted.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default=None,
        help="Import an individual split ('train', 'val', or 'test').",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Import all standard splits (train, val, test).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/datasets",
        help="Directory where output JSONL manifests will be stored.",
    )
    parser.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip per-sample image file existence checks during import (not recommended).",
    )
    parser.add_argument(
        "--deep-image-verify",
        action="store_true",
        help="Run PIL image decoding verification on all images during manifest validation pass.",
    )
    return parser.parse_args()



def print_summary_table(
    split_results: Dict[str, Tuple[List[MultilingualHandwritingSample], ImportReport]],
    manifest_paths: Dict[str, Path],
    validation_reports: Dict[str, Any],
):
    """Prints a detailed tabular report of dataset import and validation results."""
    print("\n" + "=" * 90)
    print(" IIIT INDIC HANDWRITING DATASET IMPORT REPORT")
    print("=" * 90)
    
    header = (
        f"{'Split':<8} | {'Total GT':<9} | {'Imported':<9} | {'Missing Img':<11} | "
        f"{'Malformed':<9} | {'Empty Text':<10} | {'Duplicates':<10} | {'Unicode Err':<11}"
    )
    print(header)
    print("-" * 90)

    total_gt = 0
    total_imported = 0
    total_missing = 0
    total_malformed = 0
    total_empty = 0
    total_duplicates = 0
    total_unicode = 0

    for split_name, (samples, report) in split_results.items():
        total_gt += report.total_lines
        total_imported += report.imported_count
        total_missing += report.missing_images_count
        total_malformed += report.malformed_lines_count
        total_empty += report.empty_text_count
        total_duplicates += report.duplicate_paths_count
        total_unicode += report.unicode_issues_count

        row = (
            f"{split_name:<8} | {report.total_lines:<9} | {report.imported_count:<9} | "
            f"{report.missing_images_count:<11} | {report.malformed_lines_count:<9} | "
            f"{report.empty_text_count:<10} | {report.duplicate_paths_count:<10} | "
            f"{report.unicode_issues_count:<11}"
        )
        print(row)

    print("-" * 90)
    totals_row = (
        f"{'TOTAL':<8} | {total_gt:<9} | {total_imported:<9} | {total_missing:<11} | "
        f"{total_malformed:<9} | {total_empty:<10} | {total_duplicates:<10} | "
        f"{total_unicode:<11}"
    )
    print(totals_row)
    print("=" * 90)

    print("\n[MANIFEST OUTPUTS]")
    for split_name, path in manifest_paths.items():
        val_rep = validation_reports.get(split_name)
        val_status = "VALID (0 errors)" if val_rep and val_rep.is_valid else "INVALID / WARNINGS"
        print(f"  - {split_name:<6} -> {path} [{val_status}]")


def run_integrity_checks(
    importer: IIITIndicHWImporter,
    split_results: Dict[str, Tuple[List[MultilingualHandwritingSample], ImportReport]],
) -> bool:
    """Runs and logs all dataset integrity checks."""
    print("\n" + "=" * 90)
    print(" DATASET INTEGRITY & SANITY CHECKS")
    print("=" * 90)

    integrity = importer.verify_dataset_integrity(split_results)
    all_passed = True

    # 1. Verify Sample 1
    print("\n1. First Sample Ground-Truth Verification (1.jpg mapping):")
    for split, details in integrity["sample_1_verified"].items():
        print(f"   [{split}] Image: {details['image']} | Text: '{details['text']}'")

    # 2. Check overlap across splits
    print("\n2. Cross-Split Independence Check:")
    if integrity["zero_split_overlap"]:
        print("   [PASS] Zero path overlap between train, val, and test splits.")
    else:
        print("   [FAIL] Detected overlapping sample paths across splits:")
        for pair, count in integrity["overlap_details"].items():
            print(f"     - {pair}: {count} shared images")
        all_passed = False

    # 3. Check every manifest image path exists on disk
    print("\n3. Disk Existence Verification:")
    if integrity["all_images_exist"]:
        print("   [PASS] 100% of manifest image paths exist and are accessible on disk.")
    else:
        print("   [FAIL] One or more referenced image paths do not exist on disk.")
        all_passed = False

    # 4. Check all imported transcriptions are non-empty
    print("\n4. Label Non-Empty Verification:")
    if integrity["all_transcriptions_non_empty"]:
        print("   [PASS] 100% of imported samples contain valid, non-empty Unicode transcriptions.")
    else:
        print("   [FAIL] Found empty transcriptions in imported samples.")
        all_passed = False

    print("=" * 90)
    return all_passed


def main():
    args = parse_args()

    if not args.all and not args.split:
        print("[ERROR] Please specify either --all or --split <train|val|test>")
        sys.exit(1)

    splits_to_import = ["train", "val", "test"] if args.all else [args.split]

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root

    if not dataset_root.exists():
        print(f"[ERROR] Dataset root directory not found: '{dataset_root}'")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    importer = IIITIndicHWImporter(
        dataset_root=dataset_root,
        language=args.language,
        script=args.script,
        project_root=PROJECT_ROOT,
        normalize_unicode=True,
    )

    print(f"Starting IIIT Indic handwriting import for language '{importer.language}' (script: '{importer.script}')...")
    print(f"Dataset Root : {importer.dataset_root}")
    print(f"Output Dir   : {output_dir}")
    print(f"Splits       : {splits_to_import}")

    split_results: Dict[str, Tuple[List[MultilingualHandwritingSample], ImportReport]] = {}
    manifest_paths: Dict[str, Path] = {}
    validation_reports: Dict[str, Any] = {}

    validate_images = not args.skip_image_validation

    for split_name in splits_to_import:
        print(f"\nProcessing split '{split_name}'...")
        samples, report = importer.import_split(split_name, validate_images=validate_images)
        split_results[split_name] = (samples, report)
        print(f"  -> Successfully imported {report.imported_count:,}/{report.total_lines:,} samples (issues: {len(report.issues)}).")

        # Output manifest path e.g. training/datasets/iiit_kannada_train.jsonl
        manifest_filename = f"iiit_{importer.language}_{split_name}.jsonl"
        manifest_file = output_dir / manifest_filename
        importer.save_manifest(samples, manifest_file)
        manifest_paths[split_name] = manifest_file
        print(f"  -> Saved manifest to: {manifest_file}")

        # Run manifest validation pipeline on generated file
        print(f"  -> Running manifest validation pipeline...")
        _, val_report = validate_manifest(
            manifest_path=manifest_file,
            root_dir=PROJECT_ROOT,
            verify_images=args.deep_image_verify,
            allow_duplicates=False,
        )
        validation_reports[split_name] = val_report
        print(f"  -> Manifest validation status: {'VALID' if val_report.is_valid else 'INVALID'} (errors: {val_report.error_count}, warnings: {val_report.warning_count})")


    # Print summary
    print_summary_table(split_results, manifest_paths, validation_reports)

    # Run integrity checks
    integrity_passed = run_integrity_checks(importer, split_results)

    if not integrity_passed:
        print("[WARNING] One or more dataset integrity checks failed. Please review the issues above.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Dataset import and validation completed successfully.")


if __name__ == "__main__":
    main()
