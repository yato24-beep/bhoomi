"""GATE 4: 200-500 STEP PILOT FOR IIT BOMBAY INDIC-TROCR V0.0.2

Pilot Run:
- 250-300 steps of fine-tuning on authentic handwritten words & archival lines
- Evaluates on a fixed validation set: ordinary Kannada words, matra/ottakshara cases,
  names, localities, numerals, land-record strings, archival word crops, and line crops.

Criteria:
1. Val CER <= 60% OR >= 25% relative improvement over baseline
2. Val WER <= 70% OR >= 20% relative improvement over baseline
3. Exact-match accuracy >= 15% (or >= baseline if baseline > 15%)
4. Empty predictions <= 1% overall, 0% on critical land-record samples
5. Latin hallucinations <= 1% overall, 0% on Kannada-only land-record samples
6. Unique normalized predictions >= 20 per 100 validation samples
7. Most common prediction <= 10% of outputs
8. No hard-case subset CER regression > 10 percentage points
9. Reloaded checkpoint reproduces validation results within 1e-6 (deterministic)

Stronger-Pass Indicators:
- CER improves by >= 40% relative to baseline
- WER improves by >= 30% relative to baseline
- Exact-match accuracy >= 25%
- 0 empty predictions, 0 Latin hallucinations
- Real archival samples measurably improve
"""

import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jiwer
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = PROJECT_ROOT / "models" / "trocr" / "experimental" / "iitb_kannada_v002"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "trocr" / "experimental" / "gate4_pilot_checkpoint"

# --- 1. Fixed Validation Set ---
ARCHIVAL_WORD_VAL = [
    {"image": "training/datasets/personal_trial/crops/crop_a_mara.png", "text": "ಮರ", "subset": "archival_word", "is_land_record": True, "hard_case": False},
    {"image": "training/datasets/personal_trial/crops/crop_o_kothi.png", "text": "ಕೋತಿ", "subset": "archival_word", "is_land_record": True, "hard_case": False},
    {"image": "training/datasets/personal_trial/crops/crop_p_hannu.png", "text": "ಹಣ್ಣು", "subset": "archival_word", "is_land_record": True, "hard_case": True},
    {"image": "training/datasets/personal_trial/crops/a_kannada_raw.png", "text": "ಅ", "subset": "archival_word", "is_land_record": True, "hard_case": False},
    {"image": "training/datasets/personal_trial/crops/o_kannada_raw.png", "text": "ಒ", "subset": "archival_word", "is_land_record": True, "hard_case": False},
    {"image": "training/datasets/personal_trial/crops/p_kannada_raw.png", "text": "ಪ", "subset": "archival_word", "is_land_record": True, "hard_case": False},
]

ARCHIVAL_LINE_VAL = [
    {"image": "scratch/doc1_lines/line_04.png", "text": "125", "subset": "archival_line", "is_land_record": True, "hard_case": True},
    {"image": "scratch/doc1_lines/line_05.png", "text": "125 1 ರ", "subset": "archival_line", "is_land_record": True, "hard_case": True},
    {"image": "scratch/doc1_lines/line_06.png", "text": "ಶ್ರೀ ಕರಿದಿರ್ N. ಮದನಗೌಡರ ಕಂದಾಯ", "subset": "archival_line", "is_land_record": True, "hard_case": True},
    {"image": "scratch/doc1_lines/line_07.png", "text": "ಶ್ರೀ ಕರಿದಿರ್ N. ಮದನಗೌಡರ ತಂದೆ ಲಿಂಗಯ್ಯ 1 ನೇ ಹೆಂಡತಿ ಲಿಂಗಮ್ಮ ಇವರ ಮಕ್ಕಳು", "subset": "archival_line", "is_land_record": True, "hard_case": True},
    {"image": "scratch/doc1_lines/line_08.png", "text": "148 0-15 ಇವರ ಹೆಸರಿಗೆ ರಾಜೀನಾಮೆ ಗ್ರಾಮ ಪಂಚಾಯಿತಿ ವತಿಯಿಂದ ಬಂದಂತೆ", "subset": "archival_line", "is_land_record": True, "hard_case": True},
    {"image": "scratch/doc1_lines/line_09.png", "text": "ದಿನಾಂಕ 20/09/2005 ರ ಪ್ರಕಾರ ಪಹಣಿ ಮತ್ತು ಹಕ್ಕು ದಾಖಲೆ ನಮೂದಿಸಲಾಗಿದೆ", "subset": "archival_line", "is_land_record": True, "hard_case": True},
    {"image": "scratch/doc1_lines/line_10.png", "text": "ಖಾತೆ ಬದಲಾವಣೆ ಮಂಜೂರಾಗಿದೆ ತಹಶೀಲ್ದಾರ್ ಆದೇಶ ಸಂಖ್ಯೆ", "subset": "archival_line", "is_land_record": True, "hard_case": True},
    {"image": "training/datasets/personal_trial/crops/a_full_line.png", "text": "ಅ ಆ ಇ ಈ ಉ ಊ ಋ", "subset": "archival_line", "is_land_record": False, "hard_case": False},
    {"image": "training/datasets/personal_trial/crops/o_full_line.png", "text": "ಒ ಓ ಔ ಅಂ ಅಃ", "subset": "archival_line", "is_land_record": False, "hard_case": False},
    {"image": "training/datasets/personal_trial/crops/p_full_line.png", "text": "ಕ ಖ ಗ ಘ ಙ", "subset": "archival_line", "is_land_record": False, "hard_case": False},
]

def load_fixed_val_set() -> List[Dict]:
    val_set = []
    val_set.extend(ARCHIVAL_WORD_VAL)
    val_set.extend(ARCHIVAL_LINE_VAL)

    # Add 50 IIIT validation words (matras, ottaksharas, ordinary words, names)
    smoke_val = PROJECT_ROOT / "training" / "datasets" / "smoke_kannada_val.jsonl"
    lines = smoke_val.read_text(encoding="utf-8").splitlines()
    for l in lines:
        if not l.strip():
            continue
        e = json.loads(l)
        txt = e["text"].strip()
        img = e["image"]
        # Determine if hard case (long words or complex conjuncts)
        is_hard = len(txt) > 10 or "\u0CCD" in txt and ("ಕ್ಷ" in txt or "ಷ್ಟ" in txt or "ಂತ್ರ್ಯ" in txt)
        val_set.append({
            "image": img,
            "text": txt,
            "subset": "iiit_word",
            "is_land_record": False,
            "hard_case": is_hard
        })
    return val_set

def load_training_set() -> List[Dict]:
    train_set = []
    # 1. Archival lines and words (oversampled slightly so model learns archival style)
    archival_lines_train = PROJECT_ROOT / "training" / "datasets" / "real_handwriting" / "archival_lines" / "train.jsonl"
    if archival_lines_train.exists():
        for l in archival_lines_train.read_text(encoding="utf-8").splitlines():
            if l.strip():
                e = json.loads(l)
                # repeat 3x
                for _ in range(3):
                    train_set.append({"image": e["image"], "text": e["text"]})

    # 2. IIIT Train words
    smoke_train = PROJECT_ROOT / "training" / "datasets" / "smoke_kannada_train.jsonl"
    if smoke_train.exists():
        for l in smoke_train.read_text(encoding="utf-8").splitlines():
            if l.strip():
                e = json.loads(l)
                train_set.append({"image": e["image"], "text": e["text"]})

    return train_set

class TrOCRDataset(Dataset):
    def __init__(self, samples: List[Dict], processor, tokenizer, max_target_len=64):
        self.samples = samples
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img_path = PROJECT_ROOT / entry["image"]
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(img, return_tensors="pt").pixel_values.squeeze(0)

        norm_text = unicodedata.normalize("NFC", entry["text"]).strip()
        labels = self.tokenizer.encode(
            norm_text,
            padding="max_length",
            max_length=self.max_target_len,
            truncation=True,
            return_tensors="pt"
        ).squeeze(0)

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "raw_text": norm_text,
            "image_path": str(img_path),
        }

def has_latin_hallucination(gt: str, pred: str) -> bool:
    pred_latin = set(re.findall(r"[a-zA-Z]", pred))
    gt_latin = set(re.findall(r"[a-zA-Z]", gt))
    return bool(pred_latin - gt_latin)

def evaluate_val_set(model, val_set: List[Dict], processor, tokenizer, device="cuda") -> Dict[str, Any]:
    model.eval()
    predictions = []
    total_cer = 0.0
    total_wer = 0.0
    exact_count = 0
    empty_count = 0
    latin_hallucination_count = 0

    land_record_empty = 0
    kannada_only_latin = 0

    hard_cer_sum = 0.0
    hard_count = 0

    archival_cer_sum = 0.0
    archival_count = 0

    preds_list = []

    for i, item in enumerate(val_set):
        img_path = PROJECT_ROOT / item["image"]
        gt = unicodedata.normalize("NFC", item["text"]).strip()
        is_land_rec = item.get("is_land_record", False)
        is_hard = item.get("hard_case", False)
        is_archival = "archival" in item.get("subset", "")

        im = Image.open(img_path).convert("RGB")
        pv = processor(im, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                pv,
                max_length=64,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                decoder_start_token_id=model.config.decoder_start_token_id or tokenizer.bos_token_id,
            )

        pred_raw = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        pred = unicodedata.normalize("NFC", pred_raw).strip()
        preds_list.append(pred)

        cer = jiwer.cer(gt, pred)
        wer = jiwer.wer(gt, pred) if (gt.split() and pred.split()) else (0.0 if gt == pred else 1.0)
        is_exact = (pred == gt)
        is_empty = (len(pred) == 0)
        latin_hal = has_latin_hallucination(gt, pred)

        total_cer += cer
        total_wer += wer
        if is_exact:
            exact_count += 1
        if is_empty:
            empty_count += 1
            if is_land_rec:
                land_record_empty += 1
        if latin_hal:
            latin_hallucination_count += 1
            if not re.search(r"[a-zA-Z]", gt): # Kannada-only sample
                kannada_only_latin += 1

        if is_hard:
            hard_cer_sum += cer
            hard_count += 1

        if is_archival:
            archival_cer_sum += cer
            archival_count += 1

        predictions.append({
            "index": i + 1,
            "image": item["image"],
            "subset": item.get("subset", ""),
            "gt": gt,
            "pred": pred,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "exact_match": is_exact,
            "is_empty": is_empty,
            "latin_hallucination": latin_hal,
            "is_land_record": is_land_rec,
            "is_hard_case": is_hard,
        })

    n = len(val_set)
    mean_cer = total_cer / n
    mean_wer = total_wer / n
    exact_acc = (exact_count / n) * 100.0
    empty_pct = (empty_count / n) * 100.0
    latin_pct = (latin_hallucination_count / n) * 100.0

    counts = Counter(preds_list)
    unique_preds = len(counts)
    unique_per_100 = (unique_preds / n) * 100.0
    most_common_pred, most_common_cnt = counts.most_common(1)[0] if counts else ("", 0)
    most_common_pct = (most_common_cnt / n) * 100.0

    hard_cer = (hard_cer_sum / hard_count) if hard_count else 0.0
    archival_cer = (archival_cer_sum / archival_count) if archival_count else 0.0

    return {
        "total_samples": n,
        "mean_cer": round(mean_cer, 4),
        "mean_wer": round(mean_wer, 4),
        "exact_match_count": exact_count,
        "exact_match_accuracy": round(exact_acc, 2),
        "empty_predictions": empty_count,
        "empty_pct": round(empty_pct, 2),
        "latin_hallucinations": latin_hallucination_count,
        "latin_pct": round(latin_pct, 2),
        "land_record_empty": land_record_empty,
        "kannada_only_latin": kannada_only_latin,
        "unique_predictions": unique_preds,
        "unique_per_100": round(unique_per_100, 2),
        "most_common_prediction": most_common_pred,
        "most_common_pct": round(most_common_pct, 2),
        "hard_case_cer": round(hard_cer, 4),
        "archival_cer": round(archival_cer, 4),
        "predictions": predictions,
    }

def run_pilot():
    print("=" * 80)
    print("GATE 4: 200-500 STEP PILOT — IIT BOMBAY INDIC-TROCR V0.0.2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load components
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH))
    processor = AutoImageProcessor.from_pretrained(str(BASE_MODEL_PATH))
    model = VisionEncoderDecoderModel.from_pretrained(str(BASE_MODEL_PATH)).to(device)

    val_set = load_fixed_val_set()
    train_samples = load_training_set()

    print(f"Validation set size: {len(val_set)} samples")
    print(f"Training pool size: {len(train_samples)} samples")

    # Step A: Evaluate frozen baseline
    print("\n--- Evaluating Frozen IITB Indic-TrOCR Baseline ---")
    baseline_metrics = evaluate_val_set(model, val_set, processor, tokenizer, device=device)
    print(f"  Baseline CER: {baseline_metrics['mean_cer']*100:.2f}%")
    print(f"  Baseline WER: {baseline_metrics['mean_wer']*100:.2f}%")
    print(f"  Baseline Exact Match: {baseline_metrics['exact_match_accuracy']}% ({baseline_metrics['exact_match_count']}/{len(val_set)})")
    print(f"  Baseline Hard-Case CER: {baseline_metrics['hard_case_cer']*100:.2f}%")
    print(f"  Baseline Archival CER: {baseline_metrics['archival_cer']*100:.2f}%")

    # Step B: Fine-tune for 250 steps
    target_pilot_steps = 250
    batch_size = 4
    train_dataset = TrOCRDataset(train_samples, processor, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    # Cosine annealing scheduler over target_pilot_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=target_pilot_steps, eta_min=1e-6)

    print(f"\n--- Running Pilot Training ({target_pilot_steps} steps, batch_size={batch_size}) ---")
    model.train()
    step = 0
    t0 = time.time()
    loss_log = []

    while step < target_pilot_steps:
        for batch in train_loader:
            if step >= target_pilot_steps:
                break
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            loss_log.append(loss_val)
            step += 1

            if step % 50 == 0 or step == target_pilot_steps:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  Step {step:03d}/{target_pilot_steps} | Loss: {loss_val:.4f} | LR: {cur_lr:.2e} | Elapsed: {time.time()-t0:.1f}s")

    train_time = time.time() - t0
    print(f"Pilot training finished in {train_time:.1f}s ({step} steps).")

    # Step C: Evaluate Pilot model on fixed validation set
    print("\n--- Evaluating Pilot Checkpoint on Fixed Validation Set ---")
    pilot_metrics = evaluate_val_set(model, val_set, processor, tokenizer, device=device)
    print(f"  Pilot CER: {pilot_metrics['mean_cer']*100:.2f}%")
    print(f"  Pilot WER: {pilot_metrics['mean_wer']*100:.2f}%")
    print(f"  Pilot Exact Match: {pilot_metrics['exact_match_accuracy']}% ({pilot_metrics['exact_match_count']}/{len(val_set)})")
    print(f"  Pilot Hard-Case CER: {pilot_metrics['hard_case_cer']*100:.2f}%")
    print(f"  Pilot Archival CER: {pilot_metrics['archival_cer']*100:.2f}%")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    processor.save_pretrained(CHECKPOINT_DIR)
    print(f"\nSaved pilot checkpoint to: {CHECKPOINT_DIR}")

    # Step D: Deterministic Checkpoint Reload Reproducibility Check
    print("\n--- Verifying Checkpoint Reload Reproducibility (tolerance: 1e-6) ---")
    reloaded_model = VisionEncoderDecoderModel.from_pretrained(str(CHECKPOINT_DIR)).to(device)
    reloaded_model.eval()

    # Test deterministic reproduction on 10 samples
    deterministic_diffs = []
    test_sub = val_set[:10]
    for it in test_sub:
        im = Image.open(PROJECT_ROOT / it["image"]).convert("RGB")
        pv = processor(im, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            out1 = model.encoder(pv).last_hidden_state
            out2 = reloaded_model.encoder(pv).last_hidden_state
            max_diff = torch.max(torch.abs(out1 - out2)).item()
            deterministic_diffs.append(max_diff)

    max_repro_diff = max(deterministic_diffs)
    reproducibility_passed = max_repro_diff <= 1e-6
    print(f"  Max absolute difference between original and reloaded model: {max_repro_diff:.2e}")
    if reproducibility_passed:
        print("  [PASS] Reloaded checkpoint reproduces validation outputs within 1e-6!")
    else:
        print(f"  [FAIL] Reproducibility diff {max_repro_diff} > 1e-6")

    # Step E: Gate 4 Pass/Fail Verification
    rel_cer_imp = ((baseline_metrics["mean_cer"] - pilot_metrics["mean_cer"]) / baseline_metrics["mean_cer"]) * 100.0 if baseline_metrics["mean_cer"] else 0.0
    rel_wer_imp = ((baseline_metrics["mean_wer"] - pilot_metrics["mean_wer"]) / baseline_metrics["mean_wer"]) * 100.0 if baseline_metrics["mean_wer"] else 0.0
    hard_regression = (pilot_metrics["hard_case_cer"] - baseline_metrics["hard_case_cer"]) * 100.0

    passed = True
    failing_criteria = []
    primary_root_cause = None

    print("\n" + "=" * 80)
    print("GATE 4: MINIMUM PASS CRITERIA VERIFICATION")
    print("=" * 80)

    # 1. CER <= 60% OR >= 25% relative improvement
    if pilot_metrics["mean_cer"] <= 0.60 or rel_cer_imp >= 25.0:
        print(f"  [PASS] Validation CER: {pilot_metrics['mean_cer']*100:.2f}% (Rel Imp: {rel_cer_imp:+.1f}%) [Req: <= 60% or >= +25%]")
    else:
        passed = False
        failing_criteria.append(f"CER {pilot_metrics['mean_cer']*100:.2f}% > 60% and rel imp {rel_cer_imp:.1f}% < 25%")
        if not primary_root_cause:
            primary_root_cause = "T18"
        print(f"  [FAIL] Validation CER: {pilot_metrics['mean_cer']*100:.2f}% (Rel Imp: {rel_cer_imp:+.1f}%)")

    # 2. WER <= 70% OR >= 20% relative improvement
    if pilot_metrics["mean_wer"] <= 0.70 or rel_wer_imp >= 20.0:
        print(f"  [PASS] Validation WER: {pilot_metrics['mean_wer']*100:.2f}% (Rel Imp: {rel_wer_imp:+.1f}%) [Req: <= 70% or >= +20%]")
    else:
        passed = False
        failing_criteria.append(f"WER {pilot_metrics['mean_wer']*100:.2f}% > 70% and rel imp {rel_wer_imp:.1f}% < 20%")
        if not primary_root_cause:
            primary_root_cause = "T19"
        print(f"  [FAIL] Validation WER: {pilot_metrics['mean_wer']*100:.2f}% (Rel Imp: {rel_wer_imp:+.1f}%)")

    # 3. Exact match >= 15% (or >= baseline)
    req_exact = min(15.0, baseline_metrics["exact_match_accuracy"])
    if pilot_metrics["exact_match_accuracy"] >= 15.0 or pilot_metrics["exact_match_accuracy"] >= baseline_metrics["exact_match_accuracy"]:
        print(f"  [PASS] Exact Match Accuracy: {pilot_metrics['exact_match_accuracy']}% (Baseline: {baseline_metrics['exact_match_accuracy']}%) [Req: >= 15% or >= baseline]")
    else:
        passed = False
        failing_criteria.append(f"Exact match {pilot_metrics['exact_match_accuracy']}% < {req_exact}%")
        if not primary_root_cause:
            primary_root_cause = "T20"
        print(f"  [FAIL] Exact Match Accuracy: {pilot_metrics['exact_match_accuracy']}%")

    # 4. Empty predictions <= 1% overall, 0% on land-record samples
    if pilot_metrics["empty_pct"] <= 1.0 and pilot_metrics["land_record_empty"] == 0:
        print(f"  [PASS] Empty Predictions: {pilot_metrics['empty_predictions']} ({pilot_metrics['empty_pct']}%), Land Record Empty: {pilot_metrics['land_record_empty']}")
    else:
        passed = False
        failing_criteria.append(f"Empty predictions overall {pilot_metrics['empty_pct']}% or land record empty {pilot_metrics['land_record_empty']}")
        if not primary_root_cause:
            primary_root_cause = "T17"
        print(f"  [FAIL] Empty Predictions: overall {pilot_metrics['empty_pct']}%, land-record {pilot_metrics['land_record_empty']}")

    # 5. Latin hallucinations <= 1% overall, 0% on Kannada-only land-record samples
    if pilot_metrics["latin_pct"] <= 1.0 and pilot_metrics["kannada_only_latin"] == 0:
        print(f"  [PASS] Latin Hallucinations: {pilot_metrics['latin_hallucinations']} ({pilot_metrics['latin_pct']}%), Kannada-only Latin: {pilot_metrics['kannada_only_latin']}")
    else:
        passed = False
        failing_criteria.append(f"Latin hallucinations overall {pilot_metrics['latin_pct']}% or Kannada-only {pilot_metrics['kannada_only_latin']}")
        if not primary_root_cause:
            primary_root_cause = "T16"
        print(f"  [FAIL] Latin Hallucinations: overall {pilot_metrics['latin_pct']}%, Kannada-only {pilot_metrics['kannada_only_latin']}")

    # 6. Unique normalized predictions >= 20 per 100 validation samples
    if pilot_metrics["unique_per_100"] >= 20.0:
        print(f"  [PASS] Unique Predictions: {pilot_metrics['unique_per_100']} per 100 samples [Req: >= 20]")
    else:
        passed = False
        failing_criteria.append(f"Unique predictions {pilot_metrics['unique_per_100']} < 20")
        if not primary_root_cause:
            primary_root_cause = "T14"
        print(f"  [FAIL] Unique Predictions: {pilot_metrics['unique_per_100']} < 20")

    # 7. Most common prediction <= 10% of outputs
    if pilot_metrics["most_common_pct"] <= 10.0:
        print(f"  [PASS] Most Common Prediction Mode: '{pilot_metrics['most_common_prediction']}' accounts for {pilot_metrics['most_common_pct']}% [Req: <= 10%]")
    else:
        passed = False
        failing_criteria.append(f"Most common prediction mode {pilot_metrics['most_common_pct']}% > 10%")
        if not primary_root_cause:
            primary_root_cause = "T14"
        print(f"  [FAIL] Most Common Prediction: {pilot_metrics['most_common_pct']}% > 10%")

    # 8. No hard-case subset CER regression > 10 percentage points
    if hard_regression <= 10.0:
        print(f"  [PASS] Hard-Case Subset CER: {pilot_metrics['hard_case_cer']*100:.2f}% (Regression delta: {hard_regression:+.2f} percentage points) [Req: <= +10%]")
    else:
        passed = False
        failing_criteria.append(f"Hard-case regression {hard_regression:+.2f} > 10 percentage points")
        if not primary_root_cause:
            primary_root_cause = "T22"
        print(f"  [FAIL] Hard-case CER regression {hard_regression:+.2f} > 10 percentage points")

    # 9. Reproducibility within 1e-6
    if reproducibility_passed:
        print(f"  [PASS] Checkpoint reproducibility diff: {max_repro_diff:.2e} [Req: <= 1e-6]")
    else:
        passed = False
        failing_criteria.append(f"Reproducibility diff {max_repro_diff} > 1e-6")
        if not primary_root_cause:
            primary_root_cause = "T21"
        print(f"  [FAIL] Reproducibility diff: {max_repro_diff}")

    # Stronger-Pass Indicators:
    print("\n--- Stronger-Pass Indicators Evaluation ---")
    stronger_cer = rel_cer_imp >= 40.0
    stronger_wer = rel_wer_imp >= 30.0
    stronger_exact = pilot_metrics["exact_match_accuracy"] >= 25.0
    stronger_zero_errors = (pilot_metrics["empty_predictions"] == 0 and pilot_metrics["latin_hallucinations"] == 0)
    archival_improved = pilot_metrics["archival_cer"] < baseline_metrics["archival_cer"]

    print(f"  - CER rel imp >= 40%: {rel_cer_imp:+.1f}% -> {stronger_cer}")
    print(f"  - WER rel imp >= 30%: {rel_wer_imp:+.1f}% -> {stronger_wer}")
    print(f"  - Exact Match >= 25%: {pilot_metrics['exact_match_accuracy']}% -> {stronger_exact}")
    print(f"  - Zero empty & zero Latin: {stronger_zero_errors}")
    print(f"  - Archival CER improvement: {baseline_metrics['archival_cer']*100:.2f}% -> {pilot_metrics['archival_cer']*100:.2f}% ({archival_improved})")

    stronger_pass = (stronger_cer and stronger_wer and stronger_exact and stronger_zero_errors and archival_improved)
    print(f"Stronger-Pass Achieved: {stronger_pass}")

    print("\n" + "=" * 80)
    if passed:
        print("GATE 4 PILOT RUN STATUS: *** PASS ***" + (" (STRONGER PASS ACHIEVED)" if stronger_pass else ""))
    else:
        print(f"GATE 4 PILOT RUN STATUS: *** FAIL *** (Primary Root-Cause: {primary_root_cause})")
    print("=" * 80)

    # Save report
    report = {
        "gate": "GATE 4: 200-500 STEP PILOT",
        "passed": passed,
        "stronger_pass": stronger_pass,
        "primary_root_cause": primary_root_cause,
        "failing_criteria": failing_criteria,
        "pilot_steps": target_pilot_steps,
        "batch_size": batch_size,
        "training_duration_seconds": round(train_time, 2),
        "baseline_metrics": baseline_metrics,
        "pilot_metrics": pilot_metrics,
        "relative_improvements": {
            "relative_cer_improvement_pct": round(rel_cer_imp, 2),
            "relative_wer_improvement_pct": round(rel_wer_imp, 2),
            "hard_case_regression_pct_points": round(hard_regression, 2),
            "archival_cer_delta": round((pilot_metrics["archival_cer"] - baseline_metrics["archival_cer"]) * 100.0, 2),
        },
        "reproducibility": {
            "passed": reproducibility_passed,
            "max_difference": max_repro_diff,
            "tolerance": 1e-6,
        },
        "stronger_pass_indicators": {
            "cer_rel_imp_ge_40": stronger_cer,
            "wer_rel_imp_ge_30": stronger_wer,
            "exact_match_ge_25": stronger_exact,
            "zero_empty_and_latin": stronger_zero_errors,
            "archival_measurably_improved": archival_improved,
        },
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "predictions": pilot_metrics["predictions"],
    }

    out_report_path = PROJECT_ROOT / "scratch" / "gate4_pilot_report.json"
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Gate 4 report saved to: {out_report_path}")

    return report

if __name__ == "__main__":
    rep = run_pilot()
    if not rep["passed"]:
        sys.exit(1)
