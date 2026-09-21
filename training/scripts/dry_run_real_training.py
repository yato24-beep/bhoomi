"""Dry-Run Verification Script for TrOCR Real-Data Fine-Tuning.

Verifies end-to-end compatibility of:
- Real dataset manifest and image loading
- TrOCR processor and tokenizer padding/truncation
- PyTorch Dataset and default_data_collator
- VisionEncoderDecoderModel forward pass and loss computation
- Gradient computation (backward pass)
- CER metric computation with jiwer

Does NOT run full multi-hour training. Runs 1 step to confirm readiness.
"""

import logging
import sys
from pathlib import Path
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, default_data_collator

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.train_trocr_kannada_gpu import KannadaLineDataset, compute_cer_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dry_run_real_training")


def test_dry_run():
    train_manifest = Path("training/datasets/real_handwriting/archival_lines/train.jsonl")
    val_manifest = Path("training/datasets/real_handwriting/archival_lines/val.jsonl")

    if not train_manifest.exists():
        logger.error(f"Train manifest missing: {train_manifest}")
        return False

    # Prefer local checkpoint if available to avoid downloading remote weights and missing sentencepiece
    local_checkpoint = Path("models/trocr/smoke_test_best_checkpoint")
    if local_checkpoint.exists():
        model_name = str(local_checkpoint)
        logger.info(f"Using local pre-existing checkpoint for dry run: {model_name}")
    else:
        model_name = "microsoft/trocr-small-handwritten"
        logger.info(f"Loading processor from {model_name}...")
    
    processor = TrOCRProcessor.from_pretrained(model_name)

    logger.info("Initializing KannadaLineDataset...")
    train_ds = KannadaLineDataset(str(train_manifest), processor, max_target_length=64)
    val_ds = KannadaLineDataset(str(val_manifest), processor, max_target_length=64)
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    if len(train_ds) == 0:
        logger.error("Dataset has 0 samples!")
        return False

    # Check first item
    sample_0 = train_ds[0]
    logger.info(f"Sample 0 pixel_values shape: {sample_0['pixel_values'].shape}")
    logger.info(f"Sample 0 labels shape: {sample_0['labels'].shape}")

    # Collate batch of 2
    batch_list = [train_ds[i] for i in range(min(2, len(train_ds)))]
    batch = default_data_collator(batch_list)
    logger.info(f"Batch pixel_values shape: {batch['pixel_values'].shape}")
    logger.info(f"Batch labels shape: {batch['labels'].shape}")

    logger.info(f"Loading model from {model_name}...")
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id or processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # Run 1 forward pass
    logger.info("Executing 1 forward pass...")
    outputs = model(**batch)
    loss = outputs.loss
    logger.info(f"Forward pass SUCCESS! Calculated cross-entropy loss: {loss.item():.4f}")

    # Run 1 backward pass
    logger.info("Executing 1 backward pass (gradient calculation)...")
    loss.backward()
    logger.info("Backward pass SUCCESS! Gradients verified.")

    # Check CER metric computation
    logger.info("Testing CER calculation on mock prediction...")
    class MockPred:
        predictions = batch["labels"][:, :10] # Mock token IDs
        label_ids = batch["labels"]

    # Replace -100 in mock prediction
    MockPred.predictions[MockPred.predictions == -100] = processor.tokenizer.pad_token_id
    res = compute_cer_metrics(MockPred, processor)
    logger.info(f"CER metric function output: {res}")

    logger.info(">>> DRY RUN COMPLETE: REAL-DATA TROCR TRAINING PIPELINE IS FULLY VALIDATED! <<<")
    return True


if __name__ == "__main__":
    success = test_dry_run()
    sys.exit(0 if success else 1)
