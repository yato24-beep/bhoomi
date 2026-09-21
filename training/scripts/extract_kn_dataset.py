"""Extraction and wiring script for IIIT-INDIC-HW-WORDS Kannada dataset (kn.zip).

Extracts nested train.zip, val.zip, test.zip, vocab.txt, and label manifests
from C:\\Users\\akars\\Downloads\\kn.zip into external_datasets/iiit_indic_hw_words/
matching the exact paths expected by existing manifests and prepare_real_handwriting_dataset.py.
"""

import io
import os
from pathlib import Path
import sys
import time
import zipfile
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ZIP_PATH = (
    Path("/content/kn.zip") if Path("/content/kn.zip").exists()
    else Path(r"C:\Users\akars\Downloads\kn.zip")
)
TARGET_BASE = PROJECT_ROOT / "external_datasets" / "iiit_indic_hw_words"


def extract_kn_dataset(zip_path: Path = DEFAULT_ZIP_PATH):
    print("=" * 80)
    print("EXTRACTING IIIT-INDIC-HW-WORDS KANNADA DATASET (kn.zip)")
    print(f"Source: {zip_path}")
    print(f"Target: {TARGET_BASE}")
    print("=" * 80)

    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found!")
        sys.exit(1)

    TARGET_BASE.mkdir(parents=True, exist_ok=True)

    start_total = time.time()

    with zipfile.ZipFile(zip_path, "r") as outer_zip:
        # 1. Extract text and vocabulary files first
        text_files = ["train.txt", "val.txt", "test.txt", "vocab.txt"]
        for txt_name in text_files:
            if txt_name in outer_zip.namelist():
                out_path = TARGET_BASE / txt_name
                print(f"Extracting {txt_name} -> {out_path} ...")
                with outer_zip.open(txt_name) as f_in, open(out_path, "wb") as f_out:
                    f_out.write(f_in.read())

        # Also create labels.txt inside each split folder for compatibility
        vocab_lines = (TARGET_BASE / "vocab.txt").read_text(encoding="utf-8").splitlines()
        for split in ["train", "val", "test"]:
            split_txt = TARGET_BASE / f"{split}.txt"
            if split_txt.exists():
                split_dir = TARGET_BASE / split
                split_dir.mkdir(parents=True, exist_ok=True)
                labels_file = split_dir / "labels.txt"
                print(f"Generating {labels_file} from {split_txt.name} ...")
                with open(labels_file, "w", encoding="utf-8") as f_out:
                    for line in split_txt.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 2:
                            fname = Path(parts[0]).name
                            try:
                                v_idx = int(parts[1])
                                word = vocab_lines[v_idx] if v_idx < len(vocab_lines) else ""
                                f_out.write(f"{fname}\t{word}\n")
                            except ValueError:
                                pass

        # 2. Extract inner zip archives: val.zip, test.zip, train.zip
        for split in ["val", "test", "train"]:
            inner_zip_name = f"{split}.zip"
            if inner_zip_name not in outer_zip.namelist():
                print(f"WARNING: {inner_zip_name} not found in {KN_ZIP_PATH}!")
                continue

            target_images_dir = TARGET_BASE / split / "images"
            target_images_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nProcessing {inner_zip_name} ...")
            t0 = time.time()
            with outer_zip.open(inner_zip_name) as inner_stream:
                inner_bytes = io.BytesIO(inner_stream.read())
                with zipfile.ZipFile(inner_bytes) as inner_zip:
                    members = [m for m in inner_zip.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png"))]
                    print(f"  Extracting {len(members)} images into {target_images_dir} ...")
                    extracted_count = 0
                    for m in members:
                        fname = Path(m).name
                        dest_file = target_images_dir / fname
                        with inner_zip.open(m) as f_in, open(dest_file, "wb") as f_out:
                            f_out.write(f_in.read())
                        extracted_count += 1
                        if extracted_count % 15000 == 0 or extracted_count == len(members):
                            print(f"    Extracted {extracted_count}/{len(members)} images ({time.time()-t0:.1f}s)")

            print(f"Completed {inner_zip_name} in {time.time()-t0:.1f}s.")

    print(f"\nTotal extraction completed in {time.time()-start_total:.1f}s.")

    # 3. Verification & Spot Check
    print("\n" + "=" * 80)
    print("VERIFICATION & SPOT CHECKS")
    print("=" * 80)
    spot_checks = [
        ("train", "1.jpg", "ನಾಜೂಕಾಗಿರುವುದರಿಂದ"),
        ("train", "2.jpg", "ಗುರುತಿಸಿಕೊಂಡಮೇಲೆ"),
        ("val", "1.jpg", None),
        ("val", "2.jpg", None),
        ("test", "1.jpg", None),
        ("test", "2.jpg", None),
    ]

    all_valid = True
    for split, filename, expected_text in spot_checks:
        img_path = TARGET_BASE / split / "images" / filename
        if not img_path.exists():
            print(f"FAIL: {img_path} does not exist!")
            all_valid = False
            continue

        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                w, h = img.size
                mode = img.mode
            print(f"PASS: {img_path.relative_to(PROJECT_ROOT)} | size=({w}x{h}), mode={mode} | expected_text={expected_text}")
        except Exception as e:
            print(f"FAIL: {img_path} corrupted: {e}")
            all_valid = False

    if all_valid:
        print("\nAll spot checks PASSED. Real handwriting images are valid and non-corrupt.")
    else:
        print("\nSome spot checks FAILED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract kn.zip dataset")
    parser.add_argument(
        "--zip-path",
        type=str,
        default=str(DEFAULT_ZIP_PATH),
        help="Path to kn.zip archive",
    )
    args = parser.parse_args()
    extract_kn_dataset(Path(args.zip_path))
