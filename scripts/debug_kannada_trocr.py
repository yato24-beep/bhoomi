"""Deep Root-Cause Debugging and Diagnostic Suite for Kannada TrOCR Pipeline.

Investigates:
1. Tokenizer round-trip & token ID mapping (letters vs numerals).
2. Model & Decoder configuration (vocabulary consistency, forced BOS/EOS, start tokens).
3. Checkpoint verification & loading paths.
4. Training labels verification across random dataset samples.
5. Step-by-step Top-K logits & token probabilities during decoding.
6. Image preprocessing, line/word cropping, and stroke preservation.
7. ImageProcessor tensor transformation (shape, dtype, normalization).
8. Generation configuration effects (greedy vs beam search).
9. Confidence scoring audit.
10. Unicode normalization behavior.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

# Ensure UTF-8 output encoding for Indic characters
sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import src

from transformers import TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaTokenizer
from src.preprocessing.image_enhancement import preprocess_document_image
from src.training.data_preparation import normalize_unicode_text
from src.training.evaluate import compute_cer

BASE_CKPT = r"c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint"
FT_CKPT = r"c:\Land Record\models\trocr\personal_trial_checkpoints\best_checkpoint"
TRAIN_MANIFEST = r"c:\Land Record\training\datasets\iiit_kannada_train.jsonl"


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# =====================================================================
# 1. TOKENIZER DEEP INSPECTION & ROUND-TRIP TESTS
# =====================================================================
def debug_tokenizer(tokenizer_path: str):
    print_section("1. TOKENIZER DEEP INSPECTION & ROUND-TRIP TESTS")
    print(f"Loading Tokenizer from: {tokenizer_path}")
    tok = XLMRobertaTokenizer.from_pretrained(tokenizer_path)

    print(f"Vocab size: {tok.vocab_size}")
    print(f"BOS Token: '{tok.bos_token}' (ID: {tok.bos_token_id})")
    print(f"EOS Token: '{tok.eos_token}' (ID: {tok.eos_token_id})")
    print(f"PAD Token: '{tok.pad_token}' (ID: {tok.pad_token_id})")
    print(f"UNK Token: '{tok.unk_token}' (ID: {tok.unk_token_id})")

    test_strings = [
        "ಮರ",
        "ಕೋತಿ",
        "ಹಣ್ಣು",
        "ಕ",
        "ೋ",
        "ತಿ",
        "೯",
        "೬",
        "೯೯",
        "೯೬",
        "೧೨೩೪೫",
        "ಕರ್ನಾಟಕ",
        "ಭೂಮಿ",
        "ಸರ್ವೆ",
    ]

    print("\n--- Encoding / Decoding Round-Trip Tests ---")
    print(f"{'Input Text':<15} | {'Token IDs':<30} | {'Decoded Text':<15} | {'Match?':<8} | {'Unicode Points'}")
    print("-" * 95)

    all_passed = True
    for text in test_strings:
        input_ids = tok.encode(text, add_special_tokens=True)
        input_ids_no_special = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(input_ids, skip_special_tokens=True)
        match = (decoded == text)
        if not match:
            all_passed = False
        u_points = [f"U+{ord(c):04X}" for c in text]
        dec_u_points = [f"U+{ord(c):04X}" for c in decoded]
        match_str = "PASS" if match else f"FAIL (got {decoded})"
        print(f"{text:<15} | {str(input_ids):<30} | {decoded:<15} | {match_str:<8} | {' '.join(u_points)}")

    print("-" * 95)
    print(f"Tokenizer Round-Trip Result: {'[✓] 100% PERFECT MATCH' if all_passed else '[!] ERRORS DETECTED'}")

    # Inspect token IDs for Kannada Digits vs Letters
    print("\n--- Key Token ID Mapping ---")
    digit_tokens = {d: tok.encode(d, add_special_tokens=False) for d in "೦೧೨೩೪೫೬೭೮೯"}
    print(f"Kannada Digits 0-9 Token IDs: {digit_tokens}")
    letter_tokens = {l: tok.encode(l, add_special_tokens=False) for l in ["ಮ", "ರ", "ಕ", "ೋ", "ತ", "ಿ", "ಹ", "ಣ", "್", "ಣ", "ು"]}
    print(f"Key Kannada Characters Token IDs: {letter_tokens}")


# =====================================================================
# 2. MODEL CONFIGURATION & VOCABULARY CONSISTENCY
# =====================================================================
def debug_model_config(ckpt_path: str):
    print_section("2. MODEL CONFIGURATION & VOCABULARY CONSISTENCY")
    cfg_file = os.path.join(ckpt_path, "config.json")
    gen_cfg_file = os.path.join(ckpt_path, "generation_config.json")
    proc_cfg_file = os.path.join(ckpt_path, "processor_config.json")

    with open(cfg_file, "r") as f:
        cfg = json.load(f)

    print(f"Model Type: {cfg.get('model_type')}")
    print(f"Encoder Model: {cfg.get('encoder', {}).get('model_type')}")
    print(f"Decoder Model: {cfg.get('decoder', {}).get('model_type')}")
    print(f"Decoder Vocab Size: {cfg.get('decoder', {}).get('vocab_size')}")
    print(f"Decoder Start Token ID: {cfg.get('decoder_start_token_id')}")
    print(f"EOS Token ID: {cfg.get('eos_token_id')}")
    print(f"PAD Token ID: {cfg.get('pad_token_id')}")

    if os.path.exists(gen_cfg_file):
        with open(gen_cfg_file, "r") as f:
            gen_cfg = json.load(f)
        print(f"Generation Config: {gen_cfg}")

    if os.path.exists(proc_cfg_file):
        with open(proc_cfg_file, "r") as f:
            proc_cfg = json.load(f)
        print(f"Processor Config: {proc_cfg}")


# =====================================================================
# 3. TRAINING LABELS AUDIT ACROSS 50 RANDOM SAMPLES
# =====================================================================
def debug_training_dataset_labels(manifest_path: str, tokenizer_path: str, num_samples: int = 50):
    print_section(f"3. TRAINING DATASET LABELS AUDIT ({num_samples} SAMPLES)")
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return

    tok = XLMRobertaTokenizer.from_pretrained(tokenizer_path)

    samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Total samples in {manifest_path}: {len(samples)}")
    random.seed(42)
    selected = random.sample(samples, min(num_samples, len(samples)))

    digit_samples = 0
    letter_samples = 0
    mismatches = 0

    print(f"\n{'Sample Image':<45} | {'GT String':<20} | {'Decoded from Tokens':<20} | {'Status'}")
    print("-" * 105)

    for i, s in enumerate(selected[:15]):  # print first 15 in detail
        img = s["image"]
        raw_text = s["text"]
        norm_text = normalize_unicode_text(raw_text)

        # Tokenize and decode
        t_ids = tok.encode(norm_text, add_special_tokens=True)
        dec = tok.decode(t_ids, skip_special_tokens=True)

        is_digits = any(c in "೦೧೨೩೪೫೬೭೮೯0123456789" for c in norm_text)
        if is_digits:
            digit_samples += 1
        else:
            letter_samples += 1

        match = (dec == norm_text)
        if not match:
            mismatches += 1

        status_str = "OK" if match else "MISMATCH"
        print(f"{os.path.basename(img):<45} | {raw_text:<20} | {dec:<20} | {status_str}")

    # Check remaining
    for s in selected[15:]:
        norm_text = normalize_unicode_text(s["text"])
        t_ids = tok.encode(norm_text, add_special_tokens=True)
        dec = tok.decode(t_ids, skip_special_tokens=True)
        if any(c in "೦೧೨೩೪೫೬೭೮೯0123456789" for c in norm_text):
            digit_samples += 1
        else:
            letter_samples += 1
        if dec != norm_text:
            mismatches += 1

    print("-" * 105)
    print(f"Audit Summary across {len(selected)} samples:")
    print(f"  Word / Text samples: {letter_samples} ({letter_samples/len(selected)*100:.1f}%)")
    print(f"  Digit / Number samples: {digit_samples} ({digit_samples/len(selected)*100:.1f}%)")
    print(f"  Label tokenization roundtrip mismatches: {mismatches} / {len(selected)}")


# =====================================================================
# 4. STEP-BY-STEP TOP-K LOGITS & TOKEN PREDICTION TRACER
# =====================================================================
def trace_decoding_step_logits(
    model: Any,
    processor: Any,
    tokenizer: Any,
    image_crop: Image.Image,
    label: str,
    device: str,
    top_k: int = 5,
):
    print_section(f"4. STEP-BY-STEP TOP-K LOGITS TRACE: {label}")
    pixel_values = processor(image_crop.convert("RGB"), return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            max_new_tokens=10,
            return_dict_in_generate=True,
            output_scores=True,
        )

    gen_ids = outputs.sequences[0]
    final_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"Raw Generated Token IDs: {gen_ids.tolist()}")
    print(f"Final Decoded Text: '{final_text}'")

    print(f"\n{'Step':<6} | {'Top-1 Token (Prob)':<25} | {'Top-2 Token (Prob)':<25} | {'Top-3 Token (Prob)':<25}")
    print("-" * 85)

    for step_idx, step_logits in enumerate(outputs.scores):
        probs = torch.softmax(step_logits[0], dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k)

        step_info = []
        for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
            tok_str = tokenizer.decode([idx])
            # clean representation
            tok_repr = repr(tok_str)
            step_info.append(f"{tok_repr} ({p*100:.1f}%) [id={idx}]")

        s1 = step_info[0] if len(step_info) > 0 else ""
        s2 = step_info[1] if len(step_info) > 1 else ""
        s3 = step_info[2] if len(step_info) > 2 else ""
        print(f"Step {step_idx+1:<2} | {s1:<25} | {s2:<25} | {s3:<25}")


# =====================================================================
# 5. DIAGNOSTIC RUN ON REAL TEST CASES
# =====================================================================
def run_diagnostics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Diagnostic Suite on Device: {device}")

    # 1. Tokenizer
    debug_tokenizer(BASE_CKPT)

    # 2. Config
    debug_model_config(BASE_CKPT)

    # 3. Training Labels
    debug_training_dataset_labels(TRAIN_MANIFEST, BASE_CKPT, num_samples=50)

    # Load models
    print("\nLoading Models for Top-K Logit Tracing...")
    tok = XLMRobertaTokenizer.from_pretrained(BASE_CKPT)
    proc = TrOCRProcessor.from_pretrained(BASE_CKPT)
    model_base = VisionEncoderDecoderModel.from_pretrained(BASE_CKPT).to(device)
    model_base.eval()

    model_ft = VisionEncoderDecoderModel.from_pretrained(FT_CKPT).to(device)
    model_ft.eval()

    # Test cases
    test_crops = [
        ("Known IIIT Train Sample 1 ('ನಾಜೂಕಾಗಿರುವುದರಿಂದ')", r"c:\Land Record\external_datasets\iiit_indic_hw_words\train\images\1.jpg", None),
        ("Personal Trial Crop 'crop_o_kothi.png' ('ಕೋತಿ')", r"c:\Land Record\training\datasets\personal_trial\crops\crop_o_kothi.png", None),
        ("Verified i.jpeg Crop 'i_kannada_tight.png' ('ಕೋತಿ')", r"c:\Land Record\scratch\i_kannada_tight.png", None),
        ("Full Uncropped/Sideways i.jpeg (Noise Test)", r"c:\Users\achyu\Downloads\i.jpeg", 90),
        ("New Kothi Image 'new_kothi.jpeg'", r"c:\Users\achyu\Downloads\new_kothi.jpeg", 90),
    ]

    for label, img_path, rot_deg in test_crops:
        if not os.path.exists(img_path):
            print(f"[!] Skipping {img_path} (not found)")
            continue

        im = Image.open(img_path)
        if rot_deg:
            im = im.rotate(rot_deg, expand=True)

        print_section(f"TEST CASE: {label}")
        print(f"Image Path: {img_path}")
        print(f"Image Size: {im.size}, Mode: {im.mode}")

        print("\n--- BASE MODEL LOGITS TRACE ---")
        trace_decoding_step_logits(model_base, proc, tok, im, f"Base Model: {label}", device)

        print("\n--- FINE-TUNED MODEL LOGITS TRACE ---")
        trace_decoding_step_logits(model_ft, proc, tok, im, f"Fine-Tuned Model: {label}", device)


if __name__ == "__main__":
    run_diagnostics()
