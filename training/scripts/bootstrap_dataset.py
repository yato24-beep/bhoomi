"""Dataset Bootstrap and Annotation Template CLI Script.

Provides commands to:
1. Initialize missing dataset directories for Indic languages.
2. Scan for available raw crops and detect annotated vs unannotated files.
3. Generate annotation templates without fabricating transcriptions.
4. Output a clear dataset readiness summary.

Usage:
    python training/scripts/bootstrap_dataset.py --all
    python training/scripts/bootstrap_dataset.py --language kannada
    python training/scripts/bootstrap_dataset.py --scan
"""

import argparse
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.bootstrap import (
    LANGUAGE_SCRIPT_MAP,
    bootstrap_multilingual_datasets,
    ensure_dataset_directories,
    scan_language_directory,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dataset Bootstrap and Annotation Workflow Manager"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan and bootstrap all supported Indic languages (Kannada, Hindi, Tamil, Telugu, Malayalam).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Target specific language (e.g. kannada, hindi, tamil, telugu, malayalam).",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Run discovery scan without writing new templates or manifests.",
    )
    parser.add_argument(
        "--no-templates",
        action="store_true",
        help="Disable automatic annotation template generation for unannotated images.",
    )
    return parser.parse_args()


def print_banner():
    print("=" * 70)
    print("  LAND RECORD DIGITIZATION - DATASET BOOTSTRAP & ANNOTATION WORKFLOW")
    print("=" * 70)


def main():
    args = parse_args()
    print_banner()

    default_languages = ["kannada", "hindi", "tamil", "telugu", "malayalam"]

    if args.language:
        target_languages = [args.language.lower().strip()]
    elif args.all or args.scan:
        target_languages = default_languages
    else:
        target_languages = ["kannada"]  # Default primary language

    print(f"Target Languages : {', '.join([l.capitalize() for l in target_languages])}")
    print(f"Project Root     : {PROJECT_ROOT}")
    print(f"Execution Mode   : {'SCAN ONLY' if args.scan else 'BOOTSTRAP & TEMPLATE GENERATION'}\n")

    summary = bootstrap_multilingual_datasets(
        project_root=PROJECT_ROOT,
        languages=target_languages,
        generate_templates=not (args.scan or args.no_templates),
    )

    print("----------------------------------------------------------------------")
    print("LANGUAGE INVENTORY & ANNOTATION BREAKDOWN")
    print("----------------------------------------------------------------------")

    for lang in target_languages:
        res = summary.results_by_language.get(lang)
        if not res:
            continue

        print(f"\n[ {lang.upper()} ({res.script} Script) ]")
        print(f"  Directory       : {res.directory_path.relative_to(PROJECT_ROOT)}")
        print(f"  Total Images    : {res.total_images_found}")
        print(f"  Annotated Pairs : {res.annotated_count}")
        print(f"  Unannotated     : {res.unannotated_count}")

        if res.source_type_counts:
            breakdown = [f"{k}: {v}" for k, v in res.source_type_counts.items() if v > 0]
            if breakdown:
                print(f"  Source Types    : {', '.join(breakdown)}")

        if res.unannotated_count > 0 and not args.scan and not args.no_templates:
            tmpl_path = f"data/annotations/{lang}_template.jsonl"
            print(f"  Generated Template: {tmpl_path} ({res.unannotated_count} entries to label)")

    print("\n----------------------------------------------------------------------")
    print("SUMMARY")
    print("----------------------------------------------------------------------")
    print(f"Total Discovered Images : {summary.total_images}")
    print(f"Verified Annotated Pairs: {summary.total_annotated}")
    print(f"Unannotated Images      : {summary.total_unannotated}")

    if summary.templates_generated:
        print("\nAnnotation Templates Ready for Labeling:")
        for t in summary.templates_generated:
            print(f"  -> {t}")

    print("\nNext Steps:")
    print("1. Place cropped handwriting/document images into data/raw/<language>/")
    print("2. Run 'python training/scripts/bootstrap_dataset.py --all' to generate labeling templates.")
    print("3. Add ground truth transcriptions into the generated JSONL templates.")
    print("4. Validate and split using src/training/data_preparation.py & src/training/splitter.py.")
    print("=" * 70)


if __name__ == "__main__":
    main()
