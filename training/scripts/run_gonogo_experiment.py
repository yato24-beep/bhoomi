"""GO/NO-GO Controlled Experiments for Native Kannada TrOCR.

Compares:
- Experiment A: Original microsoft/trocr-small-handwritten DeiT encoder + KanBERTo decoder
- Experiment B: Checkpoint-trained DeiT encoder + KanBERTo decoder

Data:
- 500 authentic IIIT Kannada training samples
- 100 authentic IIIT Kannada validation samples (fixed deterministic set)

Evaluates:
- Loss, True CER, True WER (preserving <unk>), Exact Match, <unk> rate, Latin hallucinations
- Evaluates at steps 0, 100, 200, 300, 400, 500
"""

import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import jiwer
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    GenerationConfig,
    RobertaForCausalLM,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gonogo_experiment")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_MANIFEST = PROJECT_ROOT / "training" / "datasets" / "iiit_kannada_train.jsonl"
VAL_MANIFEST = PROJECT_ROOT / "training" / "datasets" / "iiit_kannada_val.jsonl"
CHECKPOINT_B_PATH = Path(
    r"C:\Users\akars\Downloads\kannada_trocr_best_checkpoint (1)\content\drive\MyDrive\trocr_kannada_checkpoints\best_checkpoint"
)
OUTPUT_DIR = PROJECT_ROOT / "training" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUTPUT_DIR / "gonogo_experiment_results.json"

ENCODER_A_BASE = "microsoft/trocr-small-handwritten"
DECODER_BASE = "Naveen-k/KanBERTo"

SEED = 42
NUM_TRAIN = 500
NUM_VAL = 100
MAX_STEPS = 500
EVAL_INTERVAL = 100
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
MAX_SEQ_LEN = 32


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IIITWordDataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]], tokenizer, image_processor, max_len: int = 32):
        self.samples = samples
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = PROJECT_ROOT / item["image"]
        if not img_path.exists():
            img_path = Path(item["image"])
        
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.image_processor(img, return_tensors="pt").pixel_values.squeeze(0)

        text = item["text"].strip()
        labels = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 so loss ignores padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "text": text,
            "image_name": Path(item["image"]).name,
        }


def robust_decode(token_ids: List[int], tokenizer) -> str:
    """Decodes token IDs while PRESERVING <unk> without stripping it."""
    clean_ids = []
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.sep_token_id
    mask_id = getattr(tokenizer, "mask_token_id", -1)

    for tid in token_ids:
        if tid in (pad_id, bos_id, eos_id, mask_id):
            continue
        clean_ids.append(tid)

    if not clean_ids:
        return ""

    # Decode without skipping special tokens so <unk> is retained as '<unk>'
    return tokenizer.decode(clean_ids, skip_special_tokens=False).strip()


def build_model(encoder_source: str, device: torch.device):
    """Assembles DeiT vision encoder with KanBERTo causal decoder."""
    logger.info(f"Loading KanBERTo tokenizer from {DECODER_BASE}")
    tokenizer = AutoTokenizer.from_pretrained(DECODER_BASE)

    logger.info(f"Loading vision encoder from: {encoder_source}")
    encoder_model = VisionEncoderDecoderModel.from_pretrained(str(encoder_source))
    encoder = encoder_model.encoder

    logger.info(f"Loading KanBERTo decoder from {DECODER_BASE}")
    decoder = RobertaForCausalLM.from_pretrained(
        DECODER_BASE,
        is_decoder=True,
        add_cross_attention=True,
    )

    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
    config.pad_token_id = tokenizer.pad_token_id
    config.decoder_start_token_id = tokenizer.cls_token_id
    config.eos_token_id = tokenizer.sep_token_id
    config.bos_token_id = tokenizer.cls_token_id
    config.vocab_size = decoder.config.vocab_size

    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

    gen_config = GenerationConfig(
        decoder_start_token_id=tokenizer.cls_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        max_length=MAX_SEQ_LEN,
        no_repeat_ngram_size=3,
        num_beams=1,
        early_stopping=True,
    )
    model.generation_config = gen_config
    model.to(device)
    return model, tokenizer


def evaluate_model(model, val_loader, tokenizer, device: torch.device) -> Dict[str, Any]:
    """Evaluates validation loss, CER, WER, exact match, unk rate, and Latin hallucinations."""
    model.eval()
    total_loss = 0.0
    num_loss_batches = 0

    all_preds: List[str] = []
    all_refs: List[str] = []
    sample_details = []

    unk_count = 0
    total_token_count = 0

    with torch.no_grad():
        for batch in val_loader:
            pv = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Compute eval loss
            outputs = model(pixel_values=pv, labels=labels)
            total_loss += float(outputs.loss.item())
            num_loss_batches += 1

            # Autoregressive generation
            gen_ids = model.generate(pv, max_length=MAX_SEQ_LEN)
            for i, single_ids in enumerate(gen_ids):
                id_list = single_ids.tolist()
                unk_id = tokenizer.unk_token_id
                unk_count += sum(1 for t in id_list if t == unk_id)
                total_token_count += len(id_list)

                pred_str = robust_decode(id_list, tokenizer)
                ref_str = batch["text"][i].strip()

                all_preds.append(pred_str)
                all_refs.append(ref_str)
                sample_details.append({
                    "image": batch["image_name"][i],
                    "gt": ref_str,
                    "pred": pred_str,
                })

    avg_loss = total_loss / max(1, num_loss_batches)

    # Compute CER & WER without stripping <unk> from references or predictions
    # Guard against completely empty predictions
    safe_preds = [p if p else "<empty>" for p in all_preds]
    cer = float(jiwer.cer(all_refs, safe_preds))
    wer = float(jiwer.wer(all_refs, safe_preds))

    exact_matches = sum(1 for r, p in zip(all_refs, all_preds) if r == p)
    em_rate = exact_matches / max(1, len(all_refs))

    unk_rate = unk_count / max(1, total_token_count)

    latin_count = sum(
        1 for p in all_preds if any("a" <= c.lower() <= "z" for c in p)
    )

    return {
        "val_loss": round(avg_loss, 4),
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "exact_match": round(em_rate, 4),
        "unk_rate": round(unk_rate, 4),
        "latin_hallucinations": latin_count,
        "sample_details": sample_details,
    }


def train_experiment(
    name: str,
    encoder_source: str,
    train_samples: List[Dict[str, str]],
    val_samples: List[Dict[str, str]],
    device: torch.device,
) -> Dict[str, Any]:
    """Runs a single controlled 500-step experiment with checkpoints every 100 steps."""
    logger.info("=" * 80)
    logger.info(f"STARTING {name}")
    logger.info(f"Encoder: {encoder_source}")
    logger.info("=" * 80)

    set_seed(SEED)
    model, tokenizer = build_model(encoder_source, device)
    image_processor = AutoImageProcessor.from_pretrained(ENCODER_A_BASE)

    train_dataset = IIITWordDataset(train_samples, tokenizer, image_processor, MAX_SEQ_LEN)
    val_dataset = IIITWordDataset(val_samples, tokenizer, image_processor, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    eval_history: List[Dict[str, Any]] = []

    # Initial zero-shot evaluation at step 0
    logger.info(f"[{name}] Running initial evaluation at step 0...")
    init_metrics = evaluate_model(model, val_loader, tokenizer, device)
    init_metrics["step"] = 0
    init_metrics["train_loss"] = None
    eval_history.append(init_metrics)
    logger.info(
        f"[{name} - Step 0] Val Loss: {init_metrics['val_loss']} | "
        f"CER: {init_metrics['cer']*100:.2f}% | WER: {init_metrics['wer']*100:.2f}% | "
        f"EM: {init_metrics['exact_match']*100:.1f}% | unk: {init_metrics['unk_rate']*100:.1f}%"
    )

    model.train()
    step = 0
    running_loss = 0.0
    loss_count = 0

    epoch = 0
    start_time = time.time()

    while step < MAX_STEPS:
        epoch += 1
        for batch in train_loader:
            step += 1
            pv = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pv, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            loss_count += 1

            if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                train_loss = running_loss / max(1, loss_count)
                running_loss = 0.0
                loss_count = 0

                val_metrics = evaluate_model(model, val_loader, tokenizer, device)
                val_metrics["step"] = step
                val_metrics["train_loss"] = round(train_loss, 4)
                eval_history.append(val_metrics)

                logger.info(
                    f"[{name} - Step {step}/{MAX_STEPS}] Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']} | "
                    f"CER: {val_metrics['cer']*100:.2f}% | WER: {val_metrics['wer']*100:.2f}% | "
                    f"EM: {val_metrics['exact_match']*100:.1f}% | unk: {val_metrics['unk_rate']*100:.1f}%"
                )
                model.train()

            if step >= MAX_STEPS:
                break

    total_time = round(time.time() - start_time, 2)
    logger.info(f"[{name}] Finished in {total_time}s")

    return {
        "name": name,
        "encoder_source": str(encoder_source),
        "total_training_time_seconds": total_time,
        "eval_history": eval_history,
        "final_samples": eval_history[-1]["sample_details"] if eval_history else [],
    }


def verify_target_strings(tokenizer) -> Dict[str, Any]:
    """Verifies round-trip and <unk> rate on the 5 mandatory land-record strings."""
    test_strings = ["ಕನ್ನಡ", "ಶಿವಣ್ಣ", "ಬೆಂಗಳೂರು", "ವಿಸ್ತೀರ್ಣ", "ಸರ್ವೆ ನಂ. 124/2"]
    out = {}
    for s in test_strings:
        ids = tokenizer.encode(s)
        has_unk = tokenizer.unk_token_id in ids
        dec = robust_decode(ids, tokenizer)
        out[s] = {
            "token_ids": ids,
            "has_unk": has_unk,
            "round_trip_match": (dec == s),
            "decoded": dec,
        }
    return out


def main():
    print("=" * 80)
    print("GO/NO-GO CONTROLLED EXPERIMENT FOR NATIVE KANNADA TrOCR")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Load Data
    print(f"\nLoading authentic IIIT samples...")
    with open(TRAIN_MANIFEST, "r", encoding="utf-8") as f:
        train_all = [json.loads(line) for line in f if line.strip()]
    with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
        val_all = [json.loads(line) for line in f if line.strip()]

    train_samples = train_all[:NUM_TRAIN]
    val_samples = val_all[:NUM_VAL]

    print(f"Selected train samples: {len(train_samples)}")
    print(f"Selected val samples:   {len(val_samples)}")

    # 2. Verify Target Strings on KanBERTo Tokenizer
    kanberto_tokenizer = AutoTokenizer.from_pretrained(DECODER_BASE)
    target_results = verify_target_strings(kanberto_tokenizer)
    print("\nTarget Strings Tokenizer Check:")
    all_targets_pass = True
    for s, res in target_results.items():
        print(f"  {s:20s}: Round-Trip={res['round_trip_match']} | Contains <unk>={res['has_unk']}")
        if res["has_unk"] or not res["round_trip_match"]:
            all_targets_pass = False

    # 3. Experiment A
    exp_a_results = train_experiment(
        name="EXPERIMENT_A_DEIT_BASE",
        encoder_source=ENCODER_A_BASE,
        train_samples=train_samples,
        val_samples=val_samples,
        device=device,
    )

    # 4. Experiment B
    exp_b_results = train_experiment(
        name="EXPERIMENT_B_DEIT_CHECKPOINT",
        encoder_source=str(CHECKPOINT_B_PATH),
        train_samples=train_samples,
        val_samples=val_samples,
        device=device,
    )

    # 5. Qualitative Comparison on 20 Samples
    print("\n" + "=" * 80)
    print("QUALITATIVE COMPARISON: 20 VALIDATION SAMPLES (GROUND TRUTH vs A vs B)")
    print("=" * 80)
    samples_a = exp_a_results["final_samples"]
    samples_b = exp_b_results["final_samples"]

    qualitative_data = []
    num_to_print = min(20, len(samples_a), len(samples_b))
    for i in range(num_to_print):
        gt = samples_a[i]["gt"]
        p_a = samples_a[i]["pred"]
        p_b = samples_b[i]["pred"]
        img = samples_a[i]["image"]
        qualitative_data.append({
            "image": img,
            "ground_truth": gt,
            "prediction_a": p_a,
            "prediction_b": p_b,
        })
        print(f"\n--- Sample {i+1} ({img}) ---")
        print(f"GROUND TRUTH : {gt}")
        print(f"PREDICTION A : {p_a}")
        print(f"PREDICTION B : {p_b}")

    # 6. Save Full Results
    full_output = {
        "device": str(device),
        "target_strings_verification": target_results,
        "all_targets_pass": all_targets_pass,
        "experiment_a": exp_a_results,
        "experiment_b": exp_b_results,
        "qualitative_20_samples": qualitative_data,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)

    print(f"\nFull results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
