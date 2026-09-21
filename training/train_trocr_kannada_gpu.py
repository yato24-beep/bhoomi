"""Standalone, Self-Contained TrOCR Line-Level Fine-Tuning Script for Cloud GPUs (Google Colab / Kaggle / Linux GPU).

Usage on Google Colab or Kaggle:
    python training/train_trocr_kannada_gpu.py \
        --train-manifest training/datasets/synthetic_lines/train.jsonl \
        --val-manifest training/datasets/synthetic_lines/val.jsonl \
        --base-model microsoft/trocr-small-handwritten \
        --output-dir models/trocr/kannada_retrained_v3 \
        --epochs 5 \
        --batch-size 8 \
        --lr 3e-5 \
        --fp16
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)
import jiwer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("trocr_trainer_gpu")


class KannadaLineDataset(Dataset):
    """Dataset for line-level handwriting image crops and Unicode labels."""

    def __init__(self, manifest_path: str, processor: TrOCRProcessor, max_target_length: int = 64):
        self.processor = processor
        self.max_target_length = max_target_length
        self.samples = []

        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        base_dir = manifest_file.parent
        with open(manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                img_p = Path(item["image"])
                if not img_p.is_absolute():
                    # Check relative to cwd or manifest directory
                    if not img_p.exists() and (base_dir / img_p).exists():
                        img_p = base_dir / img_p
                self.samples.append({
                    "image_path": str(img_p),
                    "text": item["text"].strip(),
                })

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            logger.warning(f"Error opening image {item['image_path']}: {e}. Using fallback white canvas.")
            image = Image.new("RGB", (384, 64), color=(255, 255, 255))

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            item["text"],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id's with -100 so CrossEntropy ignores them
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def compute_cer_metrics(pred, processor):
    """Computes CER using jiwer across decoded predictions and references."""
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cers = []
    wers = []
    for ref, hyp in zip(label_str, pred_str):
        ref_clean = ref.strip()
        hyp_clean = hyp.strip()
        if not ref_clean:
            continue
        cers.append(jiwer.cer(ref_clean, hyp_clean))
        wers.append(jiwer.wer(ref_clean, hyp_clean))

    mean_cer = float(sum(cers) / max(len(cers), 1))
    mean_wer = float(sum(wers) / max(len(wers), 1))
    return {"cer": mean_cer, "wer": mean_wer}


def run_training(
    train_manifest: str,
    val_manifest: str,
    base_model_path: str = "microsoft/trocr-small-handwritten",
    output_dir: str = "models/trocr/kannada_retrained_v3",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 3e-5,
    fp16: bool = True,
    max_length: int = 64,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    save_total_limit: int = 3,
    max_eval_samples: Optional[int] = 500,
    resume_from_checkpoint: Optional[str] = None,
    early_stopping_patience: int = 3,
):
    """Runs TrOCR line-level fine-tuning with validation CER early stopping and Drive checkpointing."""
    import re

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target execution device: {device.upper()}")
    if device == "cuda":
        logger.info(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        logger.warning("CUDA is not available! Training on CPU will be extremely slow. Run on Google Colab / Kaggle with GPU accelerator enabled.")

    logger.info(f"Loading processor and model from: {base_model_path}")
    processor = TrOCRProcessor.from_pretrained(base_model_path)
    model = VisionEncoderDecoderModel.from_pretrained(base_model_path)

    # Latin token suppression to eliminate English decoder hallucinations
    suppress_tokens = []
    if hasattr(processor, "tokenizer"):
        vocab = processor.tokenizer.get_vocab()
        latin_regex = re.compile(r"[a-zA-Z]")
        for token_str, token_id in vocab.items():
            decoded = processor.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            if latin_regex.search(decoded):
                suppress_tokens.append(token_id)
        logger.info(f"Configured Latin token suppression: blocking {len(suppress_tokens)} ASCII/Latin tokens.")

    # Generation configuration
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id or processor.tokenizer.eos_token_id
    model.config.max_length = max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 1.0
    model.config.num_beams = 4
    if suppress_tokens:
        model.config.suppress_tokens = suppress_tokens
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.suppress_tokens = suppress_tokens

    train_dataset = KannadaLineDataset(train_manifest, processor, max_target_length=max_length)
    val_dataset = KannadaLineDataset(val_manifest, processor, max_target_length=max_length)

    # Subsample validation set if specified for fast evaluation cycles
    if max_eval_samples is not None and len(val_dataset) > max_eval_samples:
        import random
        logger.info(f"Subsampling validation set from {len(val_dataset)} to {max_eval_samples} samples for fast eval.")
        indices = list(range(len(val_dataset)))
        random.seed(42)
        random.shuffle(indices)
        val_dataset.samples = [val_dataset.samples[i] for i in indices[:max_eval_samples]]

    training_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "fp16": (fp16 and device == "cuda"),
        "predict_with_generate": True,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "logging_steps": max(1, min(50, eval_steps // 4)),
        "learning_rate": lr,
        "num_train_epochs": epochs,
        "warmup_steps": 100,
        "save_total_limit": save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "cer",
        "greater_is_better": False,
        "report_to": "none",
        "dataloader_num_workers": 2 if device == "cuda" else 0,
    }

    try:
        training_args = Seq2SeqTrainingArguments(
            eval_strategy="steps",
            **training_kwargs,
        )
    except TypeError:
        training_args = Seq2SeqTrainingArguments(
            evaluation_strategy="steps",
            **training_kwargs,
        )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=lambda pred: compute_cer_metrics(pred, processor),
        callbacks=callbacks,
    )

    logger.info("Starting TrOCR fine-tuning...")
    if resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed / 60:.2f} minutes!")

    # Save final model and processor
    final_dir = Path(output_dir) / "best_checkpoint"
    logger.info(f"Saving retrained checkpoint to: {final_dir}")
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    # Save training metadata
    meta = {
        "base_model": base_model_path,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "early_stopping_patience": early_stopping_patience,
        "device": device,
        "elapsed_seconds": round(elapsed, 2),
        "train_loss": train_result.training_loss,
        "global_step": train_result.global_step,
        "resumed_from": resume_from_checkpoint,
        "latin_suppressed_token_count": len(suppress_tokens),
    }
    with open(final_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Model, processor, and metadata successfully saved. Ready for drop-in deployment.")


def main():
    parser = argparse.ArgumentParser(description="TrOCR Kannada Line-Level GPU Retraining")
    parser.add_argument("--train-manifest", type=str, default="training/datasets/combined_handwriting/train.jsonl")
    parser.add_argument("--val-manifest", type=str, default="training/datasets/combined_handwriting/val.jsonl")
    parser.add_argument("--base-model", type=str, default="microsoft/trocr-small-handwritten")
    parser.add_argument("--output-dir", type=str, default="models/trocr/kannada_retrained_combined")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    args = parser.parse_args()

    run_training(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        fp16=args.fp16,
        max_length=args.max_length,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_eval_samples=args.max_eval_samples,
        resume_from_checkpoint=args.resume_from_checkpoint,
        early_stopping_patience=args.early_stopping_patience,
    )


if __name__ == "__main__":
    main()


