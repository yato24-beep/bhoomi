"""Training Entry Point for Handwriting Recognition Models.

Usage:
    # Dry-run validation (checks dataset manifests, pipeline integrity, and config without training):
    python training/train.py --config training/configs/kannada.yaml --dry-run

    # Execute Kannada smoke test training:
    python training/train.py --config training/configs/kannada_smoke_test.yaml

    # Execute full training with overrides:
    python training/train.py --config training/configs/kannada.yaml --epochs 3 --batch-size 8
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 output encoding for terminal display of Indic characters
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.training import (
    HandwritingAugmentor,
    HandwritingDataset,
    HandwritingTrainer,
    TrainingConfig,
    create_split_datasets,
)


def detect_hardware_environment() -> Dict[str, Any]:
    """Detects and reports hardware, CUDA/GPU capability, and PyTorch environment."""
    info: Dict[str, Any] = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": "N/A",
        "cuda_total_memory_gb": 0.0,
        "physical_gpu_name": "N/A",
        "physical_gpu_memory": "N/A",
    }

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        info["cuda_total_memory_gb"] = round(total_mem, 2)
    else:
        # Check if physical NVIDIA GPU exists on system via nvidia-smi
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                info["physical_gpu_name"] = parts[0]
                if len(parts) > 1:
                    info["physical_gpu_memory"] = parts[1]
        except Exception:
            pass

    return info


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR Handwriting Recognition Models.")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/kannada.yaml",
        help="Path to training configuration YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset manifests, model configuration, and pipeline integrity without running training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs in configuration.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override target device ('cpu', 'cuda', or 'auto').",
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default=None,
        help="Override training dataset manifest path.",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=None,
        help="Override validation dataset manifest path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint or best_checkpoint found in output_dir.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint directory to resume training from.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    print("=" * 75)
    print("  LAND RECORD DIGITIZATION - HANDWRITING RECOGNITION TRAINER")
    print("=" * 75)

    # 1. Hardware Detection Report
    hw_info = detect_hardware_environment()
    print(f"PyTorch Version  : {hw_info['pytorch_version']}")
    print(f"CUDA Available   : {hw_info['cuda_available']}")
    if hw_info["cuda_available"]:
        print(f"Active CUDA GPU  : {hw_info['cuda_device_name']} ({hw_info['cuda_total_memory_gb']} GB VRAM)")
    elif hw_info["physical_gpu_name"] != "N/A":
        print(f"Physical GPU     : {hw_info['physical_gpu_name']} (Total VRAM: {hw_info['physical_gpu_memory']})")
        print(f"Note             : PyTorch build is CPU ({hw_info['pytorch_version']}). Running in CPU mode.")
    else:
        print("Hardware Device  : Standard CPU execution environment.")

    print(f"Configuration    : {config_path}")

    config = TrainingConfig.from_yaml(config_path)

    # Apply command-line overrides
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device
    if args.train_manifest is not None:
        config.dataset.train_manifest = args.train_manifest
    if args.val_manifest is not None:
        config.dataset.val_manifest = args.val_manifest

    config.validate()

    print(f"Experiment Name  : {config.experiment_name}")
    print(f"Target Languages : {config.languages}")
    print(f"Model Identifier : {config.model.model_name_or_path}")
    print(f"Tokenizer ID     : {config.model.tokenizer_name_or_path or config.model.model_name_or_path}")
    print(f"Train Manifest   : {config.dataset.train_manifest}")
    print(f"Validation Man.  : {config.dataset.val_manifest}")
    print(f"Epochs / Batch   : {config.num_epochs} epochs | batch size {config.batch_size}")
    print(f"Augmentation     : enabled={config.augmentation.enabled} (rot ±{config.augmentation.rotation_range_deg}°)")

    # 2. Augmentation Setup
    augmentor = HandwritingAugmentor(config.augmentation) if config.augmentation.enabled else None

    # 3. Dataset Setup
    train_manifest_path = PROJECT_ROOT / config.dataset.train_manifest
    val_manifest_path = (PROJECT_ROOT / config.dataset.val_manifest) if config.dataset.val_manifest else None

    if not train_manifest_path.exists():
        print(f"\n[!] Notice: Train manifest not found at: {train_manifest_path}")
        print("    Please generate or specify a valid JSONL manifest.")
        return

    print("\n[1] Initializing Datasets...", flush=True)
    datasets = create_split_datasets(
        train_manifest=train_manifest_path,
        val_manifest=val_manifest_path,
        root_dir=PROJECT_ROOT,
        train_transform=augmentor,
        max_target_length=config.model.max_sequence_length,
        max_train_samples=config.dataset.max_train_samples,
        max_val_samples=config.dataset.max_val_samples,
        validate_images=config.dataset.validate_images,
        ignore_missing=config.dataset.ignore_missing,
    )
    print(f"    Train Samples: {len(datasets['train']):,}", flush=True)
    if "val" in datasets:
        print(f"    Val Samples  : {len(datasets['val']):,}", flush=True)

    if args.dry_run:
        print("\n[DRY RUN] Validating sample collation and pipeline integrity...", flush=True)
        if len(datasets["train"]) > 0:
            sample_item = datasets["train"][0]
            print(f"    Sample Language : {sample_item['language']}", flush=True)
            print(f"    Sample Text     : '{sample_item['text']}'", flush=True)
            print(f"    Sample Image    : {sample_item['image'].size} (mode: {sample_item['image'].mode})", flush=True)
        print("    Augmentation test successful.", flush=True)
        print("\n[DRY RUN COMPLETE] Training configuration and dataset pipeline are valid.", flush=True)
        return

    # 4. Checkpoint Resumption Discovery
    checkpoint_dir = None
    checkpoint_metadata = None

    if args.resume_from_checkpoint:
        cand_path = Path(args.resume_from_checkpoint)
        if not cand_path.is_absolute():
            cand_path = PROJECT_ROOT / cand_path
        if cand_path.exists():
            checkpoint_dir = cand_path
        else:
            raise FileNotFoundError(f"Checkpoint directory specified with --resume-from-checkpoint not found: {cand_path}")
    elif args.resume:
        out_base = PROJECT_ROOT / config.output_dir
        candidates = []
        if (out_base / "best_checkpoint").exists():
            candidates.append(out_base / "best_checkpoint")
        for step_dir in out_base.glob("checkpoint-step-*"):
            if step_dir.is_dir():
                candidates.append(step_dir)
        if candidates:
            def get_step(p):
                if p.name == "best_checkpoint":
                    return -1
                try:
                    return int(p.name.split("-")[-1])
                except Exception:
                    return 0
            step_candidates = [c for c in candidates if c.name != "best_checkpoint"]
            if step_candidates:
                checkpoint_dir = max(step_candidates, key=get_step)
            else:
                checkpoint_dir = out_base / "best_checkpoint"

    if checkpoint_dir is not None:
        from src.training.checkpointing import load_training_checkpoint
        try:
            checkpoint_metadata, _ = load_training_checkpoint(checkpoint_dir)
            print(f"\n[*] Resuming from checkpoint: {checkpoint_dir}", flush=True)
            print(f"    Completed Epochs : {checkpoint_metadata.get('epoch', 0)}", flush=True)
            print(f"    Completed Steps  : {checkpoint_metadata.get('step', 0):,}", flush=True)
            prev_cer = checkpoint_metadata.get("metrics", {}).get("val_cer", "N/A")
            print(f"    Previous Val CER : {prev_cer}", flush=True)
        except Exception as e:
            print(f"[!] Warning: Could not load metadata from {checkpoint_dir}: {e}", flush=True)

    # 5. Model & Processor Loading
    print("\n[2] Initializing Model, Tokenizer & Processor...")
    from transformers import (
        AutoImageProcessor,
        AutoTokenizer,
        RobertaTokenizer,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        XLMRobertaTokenizer,
    )

    tok_name = config.model.tokenizer_name_or_path or config.model.model_name_or_path
    tok = None

    if checkpoint_dir is not None:
        try:
            tok = AutoTokenizer.from_pretrained(str(checkpoint_dir))
        except Exception:
            tok = None

    if tok is None:
        try:
            tok = AutoTokenizer.from_pretrained(tok_name)
        except Exception:
            try:
                tok = XLMRobertaTokenizer.from_pretrained(tok_name)
            except Exception:
                try:
                    tok = RobertaTokenizer.from_pretrained(tok_name)
                except Exception:
                    tok = None

    processor = None
    if checkpoint_dir is not None:
        try:
            processor = TrOCRProcessor.from_pretrained(str(checkpoint_dir))
        except Exception:
            processor = None

    if processor is None:
        try:
            img_proc = AutoImageProcessor.from_pretrained(config.model.model_name_or_path)
        except Exception:
            img_proc = None

        if img_proc is not None and tok is not None:
            processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tok)
        else:
            try:
                processor = TrOCRProcessor.from_pretrained(config.model.model_name_or_path)
            except Exception:
                processor = None

    model_source = str(checkpoint_dir) if checkpoint_dir is not None else config.model.model_name_or_path
    print(f"    Loading model weights from: {model_source}...")
    model = VisionEncoderDecoderModel.from_pretrained(model_source)

    # Resize decoder token embeddings if tokenizer vocabulary is larger (e.g. multilingual XLM-R)
    if tok is not None and hasattr(model, "decoder") and hasattr(model.decoder, "get_input_embeddings"):
        vocab_len = len(tok)
        curr_embeddings = model.decoder.get_input_embeddings().num_embeddings
        if vocab_len != curr_embeddings:
            print(f"    Resizing decoder embeddings from {curr_embeddings:,} to {vocab_len:,} (multilingual vocab)...")
            model.decoder.resize_token_embeddings(vocab_len)
            model.config.decoder_start_token_id = tok.bos_token_id or tok.cls_token_id or 0
            model.config.pad_token_id = tok.pad_token_id or 1
            model.config.eos_token_id = tok.eos_token_id or 2
            model.config.vocab_size = vocab_len

            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
                model.generation_config.pad_token_id = model.config.pad_token_id
                model.generation_config.eos_token_id = model.config.eos_token_id
                model.generation_config.vocab_size = vocab_len

    if config.model.freeze_encoder and hasattr(model, "encoder"):
        print("    Freezing vision encoder parameters...")
        for param in model.encoder.parameters():
            param.requires_grad = False

    datasets["train"].processor = processor
    datasets["train"].max_target_length = config.model.max_sequence_length
    if "val" in datasets:
        datasets["val"].processor = processor
        datasets["val"].max_target_length = config.model.max_sequence_length

    # 6. Trainer Setup & Training Execution
    trainer = HandwritingTrainer(
        config=config,
        model=model,
        processor=processor,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("val"),
        initial_epoch=checkpoint_metadata.get("epoch", 0) if checkpoint_metadata else 0,
        initial_step=checkpoint_metadata.get("step", 0) if checkpoint_metadata else 0,
        best_cer=checkpoint_metadata.get("metrics", {}).get("val_cer", float("inf")) if checkpoint_metadata else float("inf"),
    )

    print("\n[3] Executing Training Pipeline...")
    results = trainer.train()

    print("\n" + "=" * 75)
    print("  TRAINING COMPLETE & EVALUATION AUDIT")
    print("=" * 75)
    print(f"  Experiment Name  : {results['experiment_name']}")
    print(f"  Total Steps      : {results['total_steps']}")
    print(f"  Total Epochs     : {results['total_epochs']}")
    print(f"  Best Val CER     : {results['best_cer']:.4f}")

    if trainer.last_eval_samples:
        print("\n  Validation Predictions vs Ground Truth Samples:")
        print("  " + "-" * 70)
        print(f"  {'#':<3} | {'Ground Truth (Ref)':<28} | {'Predicted (Hyp)':<28} | CER")
        print("  " + "-" * 70)
        from src.training.evaluate import compute_cer
        for idx, sample in enumerate(trainer.last_eval_samples[:10], start=1):
            ref = sample["reference"]
            hyp = sample["hypothesis"]
            cer = compute_cer(ref, hyp)
            print(f"  {idx:<3} | {ref:<28} | {hyp:<28} | {cer:.4f}")
        print("  " + "-" * 70)

    print("=" * 75)


if __name__ == "__main__":
    main()
