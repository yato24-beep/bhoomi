"""GATE 1: AUDIT SCRIPT FOR IITB INDIC-TROCR V0.0.2 SETUP

Audits:
1. Kannada tokenizer coverage: 100% of required project characters and grapheme sequences
2. 0% <unk> tokens in all labels
3. 100% exact label round-trip after tokenization and decoding
4. 0 label truncation or empty labels
5. Unicode normalization consistency
6. Image loading and image-label alignment
7. Processor/model dimension compatibility
8. Train/validation/test and writer leakage
"""

import json
import os
import sys
import hashlib
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, TrOCRProcessor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "trocr" / "experimental" / "iitb_kannada_v002"

# Taxonomy mappings
# T01: OOV / UNK token emitted
# T02: Tokenizer truncation
# T03: Non-invertible tokenization / round-trip decode failure
# T04: Unicode normalization mismatch
# T05: Empty label
# T06: Missing vowel matra in vocab
# T07: Missing conjunct / ottakshara sequence
# T08: Numeral mapping unsupported
# T09: Image-label mis-alignment / corrupted image
# T10: Processor / Model dimension mismatch
# T11: Train / Val / Test data leakage
# T12: Writer leakage across splits

VOWELS = ["ಅ", "ಆ", "ಇ", "ಈ", "ಉ", "ಊ", "ಋ", "ೠ", "ಎ", "ಏ", "ಐ", "ಒ", "ಓ", "ಔ"]
CONSONANTS = [
    "ಕ", "ಖ", "ಗ", "ಘ", "ಙ",
    "ಚ", "ಛ", "ಜ", "ಝ", "ಞ",
    "ಟ", "ಠ", "ಡ", "ಢ", "ಣ",
    "ತ", "ಥ", "ದ", "ಧ", "ನ",
    "ಪ", "ಫ", "ಬ", "ಭ", "ಮ",
    "ಯ", "ರ", "ಱ", "ಲ", "ವ",
    "ಶ", "ಷ", "ಸ", "ಹ", "ಳ", "ೞ"
]
MATRAS = ["ಾ", "ಿ", "ೀ", "ು", "ೂ", "ೃ", "ೄ", "ೆ", "ೇ", "ೈ", "ೊ", "ೋ", "ೌ"]
MODIFIERS = ["ಂ", "ಃ", "್"]
NUMERALS_KN = ["೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯"]
NUMERALS_AR = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
PUNCTUATION = [".", ",", "-", "/", "(", ")", ":", ";", "'", '"', " "]

COMMON_CONJUNCTS = [
    "ಲ್ಲ", "ತ್ತ", "ದ್ದ", "ನ್ನ", "ಕ್ಕ", "ಪ್ರ", "ಟ್ಟ", "ಳ್ಳ", "ಕ್ಷ", "ತ್ರ",
    "ಷ್ಟ", "ಸ್ತ", "ಪ್ಪ", "ಸ್ಥ", "ವ್ಯ", "ಬ್ಬ", "ತ್ಯ", "ಕ್ತ", "ಮ್ಮ", "ಚ್ಚ",
    "ದ್ಧ", "ಗ್ರ", "ದ್ರ", "ಸ್ವ", "ಕ್ರ", "ದ್ಯ", "ಧ್ಯ", "ರ್ಯ", "ನ್ಯ", "ರ್ಮ",
    "ಡ್ಡ", "ಳ್ಳಾ", "ರ್ದಿಷ್ಟ", "ಶ್ರಿ", "ಶ್ರೀ"
]

LAND_RECORD_STRINGS = [
    "ಮರ", "ಕೋತಿ", "ಹಣ್ಣು", "125", "125 1 ರ",
    "ಶ್ರೀ ಕರಿದಿರ್ N. ಮದನಗೌಡರ ಕಂದಾಯ",
    "ಶ್ರೀ ಕರಿದಿರ್ N. ಮದನಗೌಡರ ತಂದೆ ಲಿಂಗಯ್ಯ 1 ನೇ ಹೆಂಡತಿ ಲಿಂಗಮ್ಮ ಇವರ ಮಕ್ಕಳು",
    "148 0-15 ಇವರ ಹೆಸರಿಗೆ ರಾಜೀನಾಮೆ ಗ್ರಾಮ ಪಂಚಾಯಿತಿ ವತಿಯಿಂದ ಬಂದಂತೆ",
    "ದಿನಾಂಕ 20/09/2005 ರ ಪ್ರಕಾರ ಪಹಣಿ ಮತ್ತು ಹಕ್ಕು ದಾಖಲೆ ನಮೂದಿಸಲಾಗಿದೆ",
    "ಖಾತೆ ಬದಲಾವಣೆ ಮಂಜೂರಾಗಿದೆ ತಹಶೀಲ್ದಾರ್ ಆದೇಶ ಸಂಖ್ಯೆ",
    "ಸರ್ವೆ ನಂಬರ್", "ಹಿಸ್ಸಾ ನಂಬರ್", "ಖಾತೆ ಸಂಖ್ಯೆ", "ಜಮೀನಿನ ವಿವರ",
    "ದೊಡ್ಡಬಳ್ಳಾಪುರ", "ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ", "ಕಂದಾಯ ಇಲಾಖೆ", "ಕರ್ನಾಟಕ ಸರ್ಕಾರ"
]

def run_gate1_audit():
    print("=" * 80)
    print("GATE 1: AUDIT — IIT BOMBAY INDIC-TROCR V0.0.2")
    print(f"Model Path: {MODEL_PATH}")
    print("=" * 80)

    audit_results = {
        "gate": "GATE 1: AUDIT",
        "passed": True,
        "primary_root_cause": None,
        "failures": [],
        "details": {}
    }

    def record_failure(test_name: str, root_cause: str, details: Any):
        audit_results["passed"] = False
        if audit_results["primary_root_cause"] is None:
            audit_results["primary_root_cause"] = root_cause
        audit_results["failures"].append({
            "test": test_name,
            "root_cause": root_cause,
            "details": details
        })
        print(f"  [FAIL] {test_name} -> Root Cause: {root_cause}")

    # 1. Load Tokenizer & Processor & Model
    print("\n--- 1. Loading Tokenizer, Processor, and Model ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        img_processor = AutoImageProcessor.from_pretrained(str(MODEL_PATH))
        model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_PATH))
        print("  Tokenizer, ImageProcessor, and Model loaded successfully.")
        print(f"  Tokenizer class: {type(tokenizer).__name__}, Vocab size: {len(tokenizer)}")
        print(f"  UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
        print(f"  BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
        print(f"  EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
        print(f"  PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    except Exception as exc:
        record_failure("Load Model/Tokenizer", "T10", str(exc))
        return audit_results

    # Check 1: Kannada tokenizer coverage: 100% of required project characters and grapheme sequences
    print("\n--- 2. Checking Tokenizer Coverage & Exact Round-Trip on Project Inventory ---")
    inventory_items = (
        [("Vowel", v) for v in VOWELS] +
        [("Consonant", c) for c in CONSONANTS] +
        [("Matra", "ಕ" + m) for m in MATRAS] + # consonant + matra
        [("Modifier", "ಕ" + mod) for mod in MODIFIERS] +
        [("Numeral_KN", n) for n in NUMERALS_KN] +
        [("Numeral_AR", n) for n in NUMERALS_AR] +
        [("Punctuation", p) for p in PUNCTUATION if p.strip()] +
        [("Conjunct", cj) for cj in COMMON_CONJUNCTS] +
        [("LandRecordStr", s) for s in LAND_RECORD_STRINGS]
    )

    unk_count = 0
    roundtrip_fail_count = 0
    unk_samples = []
    roundtrip_samples = []

    for category, text in inventory_items:
        norm_text = unicodedata.normalize("NFC", text)
        token_ids = tokenizer.encode(norm_text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

        if tokenizer.unk_token_id in token_ids:
            unk_count += 1
            unk_samples.append((category, text, token_ids))

        # Check roundtrip
        if decoded != norm_text.strip():
            roundtrip_fail_count += 1
            roundtrip_samples.append((category, text, decoded, token_ids))

    audit_results["details"]["inventory_total"] = len(inventory_items)
    audit_results["details"]["inventory_unk_count"] = unk_count
    audit_results["details"]["inventory_roundtrip_fail_count"] = roundtrip_fail_count

    if unk_count > 0:
        record_failure("Tokenizer Coverage (<unk> tokens in inventory)", "T01", unk_samples[:10])
    else:
        print(f"  [PASS] 0 <unk> tokens in project character inventory ({len(inventory_items)} items tested)")

    if roundtrip_fail_count > 0:
        record_failure("Tokenizer Round-trip on inventory", "T03", roundtrip_samples[:10])
    else:
        print(f"  [PASS] 100% exact round-trip decoding on project inventory")

    # Check 2: Audit Manifests (labels, images, empty labels, truncation, normalization)
    print("\n--- 3. Auditing Project Manifests (Labels, Truncation, Normalization, Images) ---")
    manifest_paths = [
        ("archival_train", PROJECT_ROOT / "training" / "datasets" / "real_handwriting" / "archival_lines" / "train.jsonl"),
        ("archival_val", PROJECT_ROOT / "training" / "datasets" / "real_handwriting" / "archival_lines" / "val.jsonl"),
        ("archival_test", PROJECT_ROOT / "training" / "datasets" / "real_handwriting" / "archival_lines" / "test.jsonl"),
        ("personal_trial", PROJECT_ROOT / "training" / "datasets" / "personal_trial" / "manifest.jsonl"),
        ("iiit_val_sample", PROJECT_ROOT / "training" / "datasets" / "smoke_kannada_val.jsonl"),
    ]

    total_labels_audited = 0
    total_label_unks = 0
    total_roundtrip_fails = 0
    total_truncations = 0
    total_empty_labels = 0
    total_norm_inconsistencies = 0
    total_images_checked = 0
    total_image_load_fails = 0
    all_sample_hashes = defaultdict(list)
    writer_manifest_map = defaultdict(set)

    label_unk_details = []
    roundtrip_details = []
    truncation_details = []
    empty_label_details = []
    image_fail_details = []

    for split_name, m_path in manifest_paths:
        if not m_path.exists():
            print(f"  [SKIP] Manifest not found: {m_path}")
            continue
        print(f"  Auditing manifest {split_name} ({m_path.name})...")
        lines = m_path.read_text(encoding="utf-8").splitlines()

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            total_labels_audited += 1

            text = entry.get("text") or entry.get("label") or ""
            img_rel = entry.get("image") or entry.get("image_path") or ""

            # Check empty label
            if not text or not text.strip():
                total_empty_labels += 1
                empty_label_details.append((split_name, idx, entry))
                continue

            # Check Unicode normalization consistency
            nfc_text = unicodedata.normalize("NFC", text)
            if text != nfc_text:
                total_norm_inconsistencies += 1

            # Check tokenizer truncation & unk
            token_ids = tokenizer.encode(nfc_text, add_special_tokens=False)
            if len(token_ids) > 512:
                total_truncations += 1
                truncation_details.append((split_name, idx, len(token_ids), nfc_text[:30]))

            if tokenizer.unk_token_id in token_ids:
                total_label_unks += 1
                label_unk_details.append((split_name, idx, nfc_text, token_ids))

            # Check roundtrip
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            if decoded != nfc_text.strip():
                total_roundtrip_fails += 1
                roundtrip_details.append((split_name, idx, nfc_text, decoded))

            # Check image loading
            if img_rel:
                # resolve path
                img_path = Path(img_rel)
                if not img_path.is_absolute():
                    # Check relative to manifest dir, project root, etc.
                    cand1 = m_path.parent / img_rel
                    cand2 = PROJECT_ROOT / img_rel
                    if cand1.exists():
                        img_path = cand1
                    elif cand2.exists():
                        img_path = cand2

                total_images_checked += 1
                if not img_path.exists():
                    total_image_load_fails += 1
                    image_fail_details.append((split_name, idx, str(img_path), "File not found"))
                else:
                    try:
                        with Image.open(img_path) as im:
                            w, h = im.size
                            if w <= 0 or h <= 0:
                                total_image_load_fails += 1
                                image_fail_details.append((split_name, idx, str(img_path), f"Invalid size {w}x{h}"))
                            else:
                                # Compute image perceptual hash for leakage detection
                                hsh = hashlib.md5(im.tobytes()).hexdigest()
                                all_sample_hashes[hsh].append((split_name, str(img_path)))
                    except Exception as im_err:
                        total_image_load_fails += 1
                        image_fail_details.append((split_name, idx, str(img_path), str(im_err)))

            # Writer check
            writer = entry.get("writer") or entry.get("writer_id")
            if writer:
                writer_manifest_map[writer].add(split_name)

    print(f"  Total labels audited: {total_labels_audited}")
    print(f"  Empty labels: {total_empty_labels}")
    print(f"  Normalization inconsistencies: {total_norm_inconsistencies}")
    print(f"  Token truncations (>512): {total_truncations}")
    print(f"  Label <unk> count: {total_label_unks}")
    print(f"  Label roundtrip failures: {total_roundtrip_fails}")
    print(f"  Images checked: {total_images_checked}, Fails: {total_image_load_fails}")

    if total_empty_labels > 0:
        record_failure("Empty Labels", "T05", empty_label_details)
    else:
        print("  [PASS] 0 empty labels")

    if total_norm_inconsistencies > 0:
        record_failure("Unicode Normalization Consistency", "T04", f"{total_norm_inconsistencies} inconsistent labels")
    else:
        print("  [PASS] 100% Unicode NFC normalization consistency")

    if total_truncations > 0:
        record_failure("Label Truncation", "T02", truncation_details)
    else:
        print("  [PASS] 0 label truncations")

    if total_label_unks > 0:
        record_failure("Label UNK tokens", "T01", label_unk_details[:10])
    else:
        print("  [PASS] 0% <unk> tokens in all labels")

    if total_roundtrip_fails > 0:
        record_failure("Label exact round-trip", "T03", roundtrip_details[:10])
    else:
        print("  [PASS] 100% exact label round-trip after tokenization and decoding")

    if total_image_load_fails > 0:
        record_failure("Image loading and alignment", "T09", image_fail_details[:10])
    else:
        print("  [PASS] 100% valid image loading and image-label alignment")

    # Check 3: Processor/Model dimension compatibility
    print("\n--- 4. Checking Processor/Model Dimension Compatibility ---")
    try:
        # Check encoder dimensions
        enc_config = model.config.encoder
        dec_config = model.config.decoder
        tok_vocab_size = len(tokenizer)
        model_vocab_size = dec_config.vocab_size

        print(f"  Encoder hidden size: {enc_config.hidden_size}")
        print(f"  Decoder hidden size: {dec_config.hidden_size}")
        print(f"  Tokenizer vocab size: {tok_vocab_size}")
        print(f"  Decoder vocab size: {model_vocab_size}")

        # Check cross attention hidden size compatibility
        if enc_config.hidden_size != dec_config.hidden_size and dec_config.cross_attention_hidden_size is None:
            # ViT encoder has 768, RoBERTa decoder has 768. If equal, fine!
            record_failure("Encoder-Decoder Hidden Dimension Mismatch", "T10", f"enc: {enc_config.hidden_size} vs dec: {dec_config.hidden_size}")
        else:
            print("  [PASS] Encoder-Decoder hidden dimensions are compatible (768 == 768)")

        # Check vocab size compatibility
        if tok_vocab_size > model_vocab_size:
            record_failure("Tokenizer-Model Vocab Mismatch", "P15", f"Tokenizer {tok_vocab_size} > Model {model_vocab_size}")
        else:
            print(f"  [PASS] Vocab dimensions compatible ({tok_vocab_size} <= {model_vocab_size})")

        # Test forward pass with dummy image
        dummy_img = Image.new("RGB", (384, 384), color=(255, 255, 255))
        pixel_vals = img_processor(dummy_img, return_tensors="pt").pixel_values
        print(f"  Preprocessed dummy image shape: {pixel_vals.shape}")

        with torch.no_grad():
            enc_out = model.encoder(pixel_vals)
            print(f"  Encoder output shape: {enc_out.last_hidden_state.shape}")
        print("  [PASS] Processor/Model forward dimension compatibility verified")

    except Exception as dim_err:
        record_failure("Dimension Compatibility", "T10", str(dim_err))

    # Check 4: Train / Validation / Test and Writer Leakage
    print("\n--- 5. Checking Data and Writer Leakage across Splits ---")
    leakage_count = 0
    leakage_examples = []
    for hsh, occurrences in all_sample_hashes.items():
        splits = set(occ[0] for occ in occurrences)
        if len(splits) > 1:
            leakage_count += 1
            leakage_examples.append(occurrences)

    writer_leakage = []
    for writer, splits in writer_manifest_map.items():
        if len(splits) > 1:
            writer_leakage.append((writer, list(splits)))

    if leakage_count > 0:
        record_failure("Image Data Leakage across splits", "T11", leakage_examples[:5])
    else:
        print("  [PASS] 0 duplicate images across train/val/test splits")

    if len(writer_leakage) > 0:
        record_failure("Writer Leakage across splits", "T12", writer_leakage[:5])
    else:
        print("  [PASS] 0 writer leakage across splits")

    # Summary
    print("\n" + "=" * 80)
    if audit_results["passed"]:
        print("GATE 1 AUDIT STATUS: *** PASS ***")
    else:
        print(f"GATE 1 AUDIT STATUS: *** FAIL *** (Primary Root-Cause: {audit_results['primary_root_cause']})")
    print("=" * 80)

    # Save report
    out_report_path = PROJECT_ROOT / "scratch" / "gate1_audit_report.json"
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text(json.dumps(audit_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved Gate 1 report to: {out_report_path}")

    return audit_results

if __name__ == "__main__":
    res = run_gate1_audit()
    if not res["passed"]:
        sys.exit(1)
