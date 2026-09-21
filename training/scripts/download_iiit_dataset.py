"""IIIT-INDIC-HW-WORDS Dataset Download Instructions and Validator.

This script does NOT auto-download (registration required).
It explains how to obtain the data and validates structure if present.
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

INSTRUCTIONS = """
================================================================================
IIIT-INDIC-HW-WORDS DATASET — DOWNLOAD INSTRUCTIONS
================================================================================

The IIIT-INDIC-HW-WORDS dataset contains ~115,000 handwritten word images
across 12 Indic scripts including Kannada. It is released by CVIT, IIIT Hyderabad.

OPTION A: Direct Download via Kaggle (Fastest & No Waiting for Email)
  URL: https://www.kaggle.com/datasets/santhoshinigongidi/iit-indic-hw-words
  - Log in to Kaggle and download the archive (or via Kaggle CLI):
      kaggle datasets download -d santhoshinigongidi/iit-indic-hw-words
  - Size: ~3.5 GB (all scripts) or extract just the Kannada folder (~350 MB)

OPTION B: Official CVIT Portal Registration
  URL: https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data
  - Fill out the registration form (Name, Affiliation, Purpose)
  - You will receive a direct download link via email (typically 24-48 hours)

DIRECTORY SETUP:
  Extract the Kannada split to:

    land-record-digitization/
      external_datasets/
        iiit_indic_hw_words/
          train/
            images/        ← .jpg word crop images (e.g. 1.jpg, 2.jpg ...)
            labels.txt     ← tab-separated: filename\\ttext
          val/
            images/
            labels.txt
          test/
            images/
            labels.txt

STEP 3: Re-run the dataset preparation
  python training/scripts/prepare_real_handwriting_dataset.py

STEP 4: Run GPU training (Colab or Kaggle)
  python training/train_trocr_kannada_gpu.py \\
    --train-manifest training/datasets/real_handwriting/archival_lines/train.jsonl \\
    --val-manifest training/datasets/real_handwriting/archival_lines/val.jsonl \\
    --base-model microsoft/trocr-small-handwritten \\
    --output-dir models/trocr/kannada_retrained_real \\
    --epochs 10 --batch-size 4 --early-stopping-patience 3

================================================================================
ALTERNATIVE: ICDAR 2025 IHDR
================================================================================
  Competition page: https://icdar2025.com (check track listings for Indic scripts)
  Training data release: typically announced 3-4 months before competition deadline

================================================================================
"""


def validate_iiit_structure():
    base = Path("external_datasets/iiit_indic_hw_words")
    if not base.exists():
        print("STATUS: NOT FOUND")
        print(f"Expected at: {base.resolve()}")
        print(INSTRUCTIONS)
        return False

    print(f"Found: {base}")
    all_ok = True
    for split in ["train", "val", "test"]:
        split_dir = base / split / "images"
        labels_file = base / split / "labels.txt"
        imgs = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")) if split_dir.exists() else []
        print(f"  {split}/images: {len(imgs)} images | labels.txt: {labels_file.exists()}")
        if not imgs or not labels_file.exists():
            all_ok = False

    if all_ok:
        print("\nSTATUS: READY — run prepare_real_handwriting_dataset.py")
    else:
        print("\nSTATUS: INCOMPLETE — some splits missing images or labels")
        print(INSTRUCTIONS)
    return all_ok


if __name__ == "__main__":
    validate_iiit_structure()
