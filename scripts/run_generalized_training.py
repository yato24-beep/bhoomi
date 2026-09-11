import os
import sys
import json
import time
import torch

sys.path.insert(0, r"c:\Land Record")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.training.config import TrainingConfig
from src.training.dataset import HandwritingDataset
from src.training.augmentation import HandwritingAugmentor
from src.training.trainer import HandwritingTrainer
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, XLMRobertaTokenizer

CONFIG_PATH = r"c:\Land Record\training\configs\kannada_generalized.yaml"
OUTPUT_DIR = r"c:\Land Record\models\trocr\kannada_generalized_checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80, flush=True)
print("  GENERALIZED KANNADA OCR TRAINING PIPELINE (FULL SCRIPT + BALANCED)", flush=True)
print("=" * 80, flush=True)

config = TrainingConfig.from_yaml(CONFIG_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)
print(f"Base Model: {config.model.model_name_or_path}", flush=True)
print(f"Output Dir: {OUTPUT_DIR}", flush=True)

print("\n[1] Loading Base Checkpoint and Processor...", flush=True)
tokenizer = XLMRobertaTokenizer.from_pretrained(config.model.model_name_or_path)
processor = TrOCRProcessor.from_pretrained(config.model.model_name_or_path)
model = VisionEncoderDecoderModel.from_pretrained(config.model.model_name_or_path).to(device)

print("\n[2] Preparing Datasets with Augmentation...", flush=True)
augmentor = HandwritingAugmentor(config.augmentation) if config.augmentation.enabled else None

train_dataset = HandwritingDataset(
    manifest_path=config.dataset.train_manifest,
    processor=processor,
    transform=augmentor,
    max_target_length=config.model.max_sequence_length,
    root_dir=r"c:\Land Record",
)
val_dataset = HandwritingDataset(
    manifest_path=config.dataset.val_manifest,
    processor=processor,
    max_target_length=config.model.max_sequence_length,
    root_dir=r"c:\Land Record",
)
print(f"Train Dataset size: {len(train_dataset):,}", flush=True)
print(f"Val Dataset size:   {len(val_dataset):,}", flush=True)

print("\n[3] Starting Generalized Training...", flush=True)
trainer = HandwritingTrainer(
    config=config,
    model=model,
    processor=processor,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

t0 = time.time()
train_summary = trainer.train(num_epochs=config.num_epochs)
elapsed = time.time() - t0

print("\n" + "=" * 80, flush=True)
print(f"Generalized Training Complete in {elapsed:.1f}s ({elapsed/60:.2f} mins)!", flush=True)
print(f"Total Steps: {train_summary['total_steps']}", flush=True)
print(f"Best CER: {train_summary['best_cer']:.4f}", flush=True)
print("=" * 80, flush=True)

# Ensure best_checkpoint directory exists and has model files
best_ckpt_dir = os.path.join(OUTPUT_DIR, "best_checkpoint")
if not os.path.exists(best_ckpt_dir):
    model.save_pretrained(best_ckpt_dir)
    tokenizer.save_pretrained(best_ckpt_dir)
    processor.save_pretrained(best_ckpt_dir)

print(f"[✓] Candidate checkpoint ready at: {best_ckpt_dir}", flush=True)
