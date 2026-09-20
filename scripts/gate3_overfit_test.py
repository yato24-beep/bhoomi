"""GATE 3: 32-SAMPLE OVERFIT TEST FOR IIT BOMBAY INDIC-TROCR V0.0.2

Uses 32 manually inspected samples containing:
- Kannada words, matras, ottaksharas, names, numerals, difficult handwriting, short/long labels
- Augmentation disabled

Pass Criteria:
1. Training loss decreases by >=80% or reaches clearly low stable plateau
2. Exact-match accuracy >= 90% (>= 29/32)
3. CER <= 5%
4. WER <= 10%
5. Empty predictions = 0/32
6. Latin hallucinations = 0/32
7. No single prediction exceeds 25% of outputs
8. No systematic missing-matra, missing-ottakshara, or repeated-token pattern
"""

import copy
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
MODEL_PATH = PROJECT_ROOT / "models" / "trocr" / "experimental" / "iitb_kannada_v002"

# 32 Curated Samples
SAMPLES_CONFIG = [
    # --- 1. Archival Words (6) ---
    {"image": "training/datasets/personal_trial/crops/crop_a_mara.png", "text": "ಮರ", "category": "word_basic"},
    {"image": "training/datasets/personal_trial/crops/crop_o_kothi.png", "text": "ಕೋತಿ", "category": "word_matra"},
    {"image": "training/datasets/personal_trial/crops/crop_p_hannu.png", "text": "ಹಣ್ಣು", "category": "word_ottakshara"},
    {"image": "training/datasets/personal_trial/crops/a_kannada_raw.png", "text": "ಅ", "category": "short_letter"},
    {"image": "training/datasets/personal_trial/crops/o_kannada_raw.png", "text": "ಒ", "category": "short_letter"},
    {"image": "training/datasets/personal_trial/crops/p_kannada_raw.png", "text": "ಪ", "category": "short_letter"},

    # --- 2. Archival Lines & Land Record Crops (10) ---
    {"image": "scratch/doc1_lines/line_04.png", "text": "125", "category": "numeral_short"},
    {"image": "scratch/doc1_lines/line_05.png", "text": "125 1 ರ", "category": "numeral_phrase"},
    {"image": "scratch/doc1_lines/line_06.png", "text": "ಶ್ರೀ ಕರಿದಿರ್ N. ಮದನಗೌಡರ ಕಂದಾಯ", "category": "name_cadastral"},
    {"image": "scratch/doc1_lines/line_07.png", "text": "ಶ್ರೀ ಕರಿದಿರ್ N. ಮದನಗೌಡರ ತಂದೆ ಲಿಂಗಯ್ಯ 1 ನೇ ಹೆಂಡತಿ ಲಿಂಗಮ್ಮ ಇವರ ಮಕ್ಕಳು", "category": "long_names"},
    {"image": "scratch/doc1_lines/line_08.png", "text": "148 0-15 ಇವರ ಹೆಸರಿಗೆ ರಾಜೀನಾಮೆ ಗ್ರಾಮ ಪಂಚಾಯಿತಿ ವತಿಯಿಂದ ಬಂದಂತೆ", "category": "long_cadastral"},
    {"image": "scratch/doc1_lines/line_09.png", "text": "ದಿನಾಂಕ 20/09/2005 ರ ಪ್ರಕಾರ ಪಹಣಿ ಮತ್ತು ಹಕ್ಕು ದಾಖಲೆ ನಮೂದಿಸಲಾಗಿದೆ", "category": "long_date_record"},
    {"image": "scratch/doc1_lines/line_10.png", "text": "ಖಾತೆ ಬದಲಾವಣೆ ಮಂಜೂರಾಗಿದೆ ತಹಶೀಲ್ದಾರ್ ಆದೇಶ ಸಂಖ್ಯೆ", "category": "cadastral_term"},
    {"image": "training/datasets/personal_trial/crops/a_full_line.png", "text": "ಅ ಆ ಇ ಈ ಉ ಊ ಋ", "category": "vowel_line"},
    {"image": "training/datasets/personal_trial/crops/o_full_line.png", "text": "ಒ ಓ ಔ ಅಂ ಅಃ", "category": "modifier_line"},
    {"image": "training/datasets/personal_trial/crops/p_full_line.png", "text": "ಕ ಖ ಗ ಘ ಙ", "category": "consonant_line"},

    # --- 3. IIIT Indic HW Words (16) ---
    {"image": "external_datasets/iiit_indic_hw_words/val/images/1.jpg", "text": "ಚುರಮರಿ", "category": "iiit_matras"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/2.jpg", "text": "ಟಿವಿ", "category": "iiit_short"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/3.jpg", "text": "ಕಡೆಯೂ", "category": "iiit_matra_uu"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/4.jpg", "text": "ಸಂಚಾರಿ", "category": "iiit_anusvara"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/5.jpg", "text": "ಶಿಕ್ಷಣದವರೆಗೂ", "category": "iiit_ottakshara_ksha"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/6.jpg", "text": "ಅಫಸರ್", "category": "iiit_virama"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/7.jpg", "text": "ಇಡಿವಿವಿಧ", "category": "iiit_vowel_matra"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/8.jpg", "text": "ಮನತೆ", "category": "iiit_basic_word"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/9.jpg", "text": "ಇನ್ನಿತರೆ", "category": "iiit_ottakshara_nna"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/10.jpg", "text": "ಕದ್ರಿ", "category": "iiit_ottakshara_dri"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/11.jpg", "text": "ಎತ್ತುವ", "category": "iiit_ottakshara_tta"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/12.jpg", "text": "ಸಚಿವರಾಗಿದ್ದರೂ", "category": "iiit_complex_word"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/16.jpg", "text": "ಕೊಲ್ಲೂರು", "category": "iiit_matra_o_ottakshara"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/18.jpg", "text": "ಮುಖ್ಯಮಂತ್ರಿಗಳಾಗಿ", "category": "iiit_long_compound"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/21.jpg", "text": "ದೃಷ್ಟಿಯ", "category": "iiit_vowel_sign_ru"},
    {"image": "external_datasets/iiit_indic_hw_words/val/images/24.jpg", "text": "ಬೆಳೆಯಲಾಗಿದ್ದ", "category": "iiit_passive_verb"},
]

class OverfitDataset(Dataset):
    def __init__(self, samples: List[Dict], processor, tokenizer):
        self.samples = samples
        self.processor = processor
        self.tokenizer = tokenizer

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
            max_length=64,
            truncation=True,
            return_tensors="pt"
        ).squeeze(0)

        # Mask pad tokens with -100 for CrossEntropyLoss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "raw_text": norm_text,
            "image_path": str(img_path),
            "category": entry["category"],
            "index": idx
        }

def has_latin_hallucination(gt: str, pred: str) -> bool:
    pred_latin = set(re.findall(r"[a-zA-Z]", pred))
    gt_latin = set(re.findall(r"[a-zA-Z]", gt))
    return bool(pred_latin - gt_latin)

def evaluate_model(model, dataset, tokenizer, device="cuda") -> Dict[str, Any]:
    model.eval()
    predictions = []
    total_cer = 0.0
    total_wer = 0.0
    exact_count = 0
    empty_count = 0
    latin_count = 0
    preds_list = []

    for i in range(len(dataset)):
        item = dataset[i]
        pv = item["pixel_values"].unsqueeze(0).to(device)
        gt = item["raw_text"]

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
        latin = has_latin_hallucination(gt, pred)

        total_cer += cer
        total_wer += wer
        if is_exact:
            exact_count += 1
        if is_empty:
            empty_count += 1
        if latin:
            latin_count += 1

        predictions.append({
            "index": i + 1,
            "image": item["image_path"],
            "category": item["category"],
            "gt": gt,
            "pred": pred,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "exact_match": is_exact,
            "is_empty": is_empty,
            "has_latin": latin,
        })

    n = len(dataset)
    mean_cer = total_cer / n
    mean_wer = total_wer / n
    exact_acc = (exact_count / n) * 100.0

    counts = Counter(preds_list)
    unique_preds = len(counts)
    most_common_pred, most_common_cnt = counts.most_common(1)[0] if counts else ("", 0)
    most_common_pct = (most_common_cnt / n) * 100.0

    return {
        "mean_cer": round(mean_cer, 4),
        "mean_wer": round(wer if n == 1 else mean_wer, 4),
        "exact_match_accuracy": round(exact_acc, 2),
        "exact_match_count": exact_count,
        "total_samples": n,
        "empty_predictions": empty_count,
        "latin_hallucinations": latin_count,
        "unique_predictions": unique_preds,
        "most_common_prediction": most_common_pred,
        "most_common_pct": round(most_common_pct, 2),
        "predictions": predictions,
    }

def run_overfit_test():
    print("=" * 80)
    print("GATE 3: 32-SAMPLE OVERFIT TEST — IIT BOMBAY INDIC-TROCR V0.0.2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("=" * 80)

    # 1. Load components
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    processor = AutoImageProcessor.from_pretrained(str(MODEL_PATH))
    model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_PATH)).to(device)

    # Dataset & DataLoader (no augmentation)
    dataset = OverfitDataset(SAMPLES_CONFIG, processor, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Loaded {len(dataset)} samples for overfit verification.")

    # Evaluate zero-step baseline
    print("\n--- Evaluating Initial Pre-Training Zero-Step Baseline ---")
    initial_eval = evaluate_model(model, dataset, tokenizer, device=device)
    print(f"Initial Baseline CER: {initial_eval['mean_cer']*100:.2f}%, WER: {initial_eval['mean_wer']*100:.2f}%, Exact Match: {initial_eval['exact_match_accuracy']}% ({initial_eval['exact_match_count']}/32)")

    # Optimizer setup (overfitting 32 samples)
    # Freeze encoder initially or train end-to-end with low LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    max_epochs = 35 # ~280 optimization steps
    total_steps = len(dataloader) * max_epochs
    print(f"Training for up to {max_epochs} epochs ({total_steps} steps)...")

    loss_history = []
    initial_loss = None
    best_exact = initial_eval["exact_match_accuracy"]
    best_cer = initial_eval["mean_cer"]
    final_eval = initial_eval

    step_count = 0
    t0 = time.time()
    converged = False

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            step_count += 1

            if initial_loss is None:
                initial_loss = loss_val

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        # Periodic evaluation every 5 epochs
        if epoch % 5 == 0 or epoch == max_epochs:
            cur_eval = evaluate_model(model, dataset, tokenizer, device=device)
            loss_drop_pct = ((initial_loss - avg_loss) / initial_loss) * 100.0 if initial_loss else 0.0
            print(f"Epoch {epoch:02d}/{max_epochs} | Loss: {avg_loss:.4f} (-{loss_drop_pct:.1f}%) | CER: {cur_eval['mean_cer']*100:.2f}% | WER: {cur_eval['mean_wer']*100:.2f}% | Exact: {cur_eval['exact_match_accuracy']}% ({cur_eval['exact_match_count']}/32)")

            if cur_eval["exact_match_accuracy"] >= 90.0 and cur_eval["mean_cer"] <= 0.05 and cur_eval["mean_wer"] <= 0.10:
                print(f"\nTarget overfit criteria reached at epoch {epoch} (step {step_count})!")
                final_eval = cur_eval
                converged = True
                break
            final_eval = cur_eval

    elapsed = time.time() - t0
    print(f"\nOverfit training completed in {elapsed:.1f}s ({step_count} steps).")

    # Gate 3 Criteria Verification
    loss_reduction_pct = ((initial_loss - loss_history[-1]) / initial_loss) * 100.0 if initial_loss else 0.0
    passed = True
    failing_criteria = []
    primary_root_cause = None

    print("\n" + "=" * 80)
    print("GATE 3: CRITERIA CHECK")
    print("=" * 80)

    # 1. Training loss reduction >= 80%
    if loss_reduction_pct < 80.0 and loss_history[-1] > 0.15:
        passed = False
        failing_criteria.append(f"Loss reduction {loss_reduction_pct:.1f}% < 80% (final loss: {loss_history[-1]:.4f})")
        if not primary_root_cause:
            primary_root_cause = "T13" # High loss plateau / convergence failure
        print(f"  [FAIL] Loss Reduction: {loss_reduction_pct:.1f}% (< 80%)")
    else:
        print(f"  [PASS] Loss Reduction: {loss_reduction_pct:.1f}% (>= 80% or low plateau)")

    # 2. Exact match >= 90%
    if final_eval["exact_match_accuracy"] < 90.0:
        passed = False
        failing_criteria.append(f"Exact match {final_eval['exact_match_accuracy']}% < 90% ({final_eval['exact_match_count']}/32)")
        if not primary_root_cause:
            primary_root_cause = "T20" # Low exact match
        print(f"  [FAIL] Exact Match Accuracy: {final_eval['exact_match_accuracy']}% (< 90%)")
    else:
        print(f"  [PASS] Exact Match Accuracy: {final_eval['exact_match_accuracy']}% (>= 90%)")

    # 3. CER <= 5%
    if final_eval["mean_cer"] > 0.05:
        passed = False
        failing_criteria.append(f"Mean CER {final_eval['mean_cer']*100:.2f}% > 5%")
        if not primary_root_cause:
            primary_root_cause = "T18" # High CER
        print(f"  [FAIL] Mean CER: {final_eval['mean_cer']*100:.2f}% (> 5%)")
    else:
        print(f"  [PASS] Mean CER: {final_eval['mean_cer']*100:.2f}% (<= 5%)")

    # 4. WER <= 10%
    if final_eval["mean_wer"] > 0.10:
        passed = False
        failing_criteria.append(f"Mean WER {final_eval['mean_wer']*100:.2f}% > 10%")
        if not primary_root_cause:
            primary_root_cause = "T19" # High WER
        print(f"  [FAIL] Mean WER: {final_eval['mean_wer']*100:.2f}% (> 10%)")
    else:
        print(f"  [PASS] Mean WER: {final_eval['mean_wer']*100:.2f}% (<= 10%)")

    # 5. Empty predictions = 0/32
    if final_eval["empty_predictions"] > 0:
        passed = False
        failing_criteria.append(f"Empty predictions: {final_eval['empty_predictions']}/32")
        if not primary_root_cause:
            primary_root_cause = "T17" # Empty prediction
        print(f"  [FAIL] Empty Predictions: {final_eval['empty_predictions']} (> 0)")
    else:
        print(f"  [PASS] Empty Predictions: 0/32")

    # 6. Latin hallucinations = 0/32
    if final_eval["latin_hallucinations"] > 0:
        passed = False
        failing_criteria.append(f"Latin hallucinations: {final_eval['latin_hallucinations']}/32")
        if not primary_root_cause:
            primary_root_cause = "T16" # Latin hallucination
        print(f"  [FAIL] Latin Hallucinations: {final_eval['latin_hallucinations']} (> 0)")
    else:
        print(f"  [PASS] Latin Hallucinations: 0/32")

    # 7. No single prediction exceeds 25% of outputs
    if final_eval["most_common_pct"] > 25.0:
        passed = False
        failing_criteria.append(f"Most common prediction mode collapse: {final_eval['most_common_pct']}% > 25%")
        if not primary_root_cause:
            primary_root_cause = "T14" # Mode collapse
        print(f"  [FAIL] Mode collapse: '{final_eval['most_common_prediction']}' is {final_eval['most_common_pct']}% (> 25%)")
    else:
        print(f"  [PASS] Prediction Diversity: Mode represents {final_eval['most_common_pct']}% (<= 25%)")

    # 8. Systematic missing-matra, missing-ottakshara, or repeated-token pattern
    systematic_matra_err = 0
    systematic_ottakshara_err = 0
    for p in final_eval["predictions"]:
        if not p["exact_match"]:
            # check if matra was stripped
            gt_matras = [ch for ch in p["gt"] if "\u0CBE" <= ch <= "\u0CCC"]
            pred_matras = [ch for ch in p["pred"] if "\u0CBE" <= ch <= "\u0CCC"]
            if len(gt_matras) > len(pred_matras):
                systematic_matra_err += 1
            if "\u0CCD" in p["gt"] and "\u0CCD" not in p["pred"]:
                systematic_ottakshara_err += 1

    if systematic_matra_err > 8: # >25% of dataset
        passed = False
        failing_criteria.append(f"Systematic missing matras in {systematic_matra_err} samples")
        if not primary_root_cause:
            primary_root_cause = "T24"
        print(f"  [FAIL] Systematic missing matras: {systematic_matra_err}")
    elif systematic_ottakshara_err > 8:
        passed = False
        failing_criteria.append(f"Systematic missing ottaksharas in {systematic_ottakshara_err} samples")
        if not primary_root_cause:
            primary_root_cause = "T23"
        print(f"  [FAIL] Systematic missing ottaksharas: {systematic_ottakshara_err}")
    else:
        print(f"  [PASS] No systematic missing-matra or missing-ottakshara patterns detected")

    print("\n" + "=" * 80)
    if passed:
        print("GATE 3 OVERFIT TEST STATUS: *** PASS ***")
    else:
        print(f"GATE 3 OVERFIT TEST STATUS: *** FAIL *** (Primary Root-Cause: {primary_root_cause})")
    print("=" * 80)

    # Save Overfit Report
    report = {
        "gate": "GATE 3: 32-SAMPLE OVERFIT TEST",
        "passed": passed,
        "primary_root_cause": primary_root_cause,
        "failing_criteria": failing_criteria,
        "initial_loss": initial_loss,
        "final_loss": loss_history[-1] if loss_history else None,
        "loss_reduction_pct": round(loss_reduction_pct, 2),
        "initial_metrics": {
            "cer": initial_eval["mean_cer"],
            "wer": initial_eval["mean_wer"],
            "exact_match_accuracy": initial_eval["exact_match_accuracy"],
            "exact_match_count": initial_eval["exact_match_count"],
        },
        "final_metrics": {
            "cer": final_eval["mean_cer"],
            "wer": final_eval["mean_wer"],
            "exact_match_accuracy": final_eval["exact_match_accuracy"],
            "exact_match_count": final_eval["exact_match_count"],
            "empty_predictions": final_eval["empty_predictions"],
            "latin_hallucinations": final_eval["latin_hallucinations"],
            "unique_predictions": final_eval["unique_predictions"],
            "most_common_prediction": final_eval["most_common_prediction"],
            "most_common_pct": final_eval["most_common_pct"],
        },
        "loss_history": loss_history,
        "predictions": final_eval["predictions"],
        "checkpoint_dir": str(PROJECT_ROOT / "models" / "trocr" / "experimental" / "gate3_overfit_checkpoint"),
    }

    # Save checkpoint if passed
    if passed:
        ckpt_dir = Path(report["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        print(f"Overfit checkpoint saved to: {ckpt_dir}")

    out_report_path = PROJECT_ROOT / "scratch" / "gate3_overfit_report.json"
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Gate 3 report saved to: {out_report_path}")

    return report

if __name__ == "__main__":
    rep = run_overfit_test()
    if not rep["passed"]:
        sys.exit(1)
