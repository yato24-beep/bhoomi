"""Modular Hugging Face / PyTorch Training Wrapper for Handwriting OCR.

Provides structured fine-tuning for VisionEncoderDecoder / TrOCR models with:
- Dependency injection for models, processors, optimizers, and datasets
- Batch collation with label masking
- Periodic evaluation with CER/WER calculation
- Checkpointing with full audit metadata
- Fully testable without downloading large models
"""

import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    DataLoader = None
    HAS_TORCH = False

from src.training.checkpointing import CheckpointMetadata, save_training_checkpoint
from src.training.config import TrainingConfig
from src.training.dataset import HandwritingDataset
from src.training.evaluate import EvaluationReport, evaluate_predictions


def default_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates a list of dataset item dictionaries into batched tensors."""
    if not batch:
        return {}

    has_tensors = "pixel_values" in batch[0] and "labels" in batch[0] and HAS_TORCH and torch is not None

    if has_tensors:
        try:
            pixel_values = torch.stack([item["pixel_values"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            return {
                "pixel_values": pixel_values,
                "labels": labels,
                "texts": [item.get("text", "") for item in batch],
                "languages": [item.get("language", "unknown") for item in batch],
                "scripts": [item.get("script", "unknown") for item in batch],
            }
        except Exception:
            pass

    # Fallback dictionary for non-tensor or mock batches
    return {
        "images": [item.get("image") for item in batch],
        "texts": [item.get("text", "") for item in batch],
        "languages": [item.get("language", "unknown") for item in batch],
        "scripts": [item.get("script", "unknown") for item in batch],
        "raw_items": batch,
    }


class HandwritingTrainer:
    """Modular trainer for fine-tuning TrOCR / VisionEncoderDecoder architectures."""

    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[Any] = None,
        processor: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
        collate_fn: Optional[Callable] = None,
        initial_epoch: int = 0,
        initial_step: int = 0,
        best_cer: float = float("inf"),
        training_history: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initializes trainer with config and optional injected dependencies."""
        self.config = config
        self.config.validate()

        self._model = model
        self._processor = processor
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._collate_fn = collate_fn or default_collate_fn

        # Device determination
        if self.config.device == "auto":
            self.device_str = "cuda" if (HAS_TORCH and torch is not None and torch.cuda.is_available()) else "cpu"
        else:
            self.device_str = self.config.device

        # Setup model on target device if real PyTorch module
        if self._model is not None and HAS_TORCH and torch is not None and isinstance(self._model, torch.nn.Module):
            try:
                self._model.to(self.device_str)
            except Exception:
                pass

        self.global_step: int = initial_step
        self.current_epoch: int = initial_epoch
        self.best_cer: float = best_cer
        self.training_history: List[Dict[str, Any]] = list(training_history) if training_history else []
        self.last_eval_samples: List[Dict[str, str]] = []

        # Setup Mixed Precision (AMP)
        self.use_amp = (
            self.config.mixed_precision in ["fp16", "mixed"]
            and self.device_str == "cuda"
            and HAS_TORCH
            and torch is not None
            and torch.cuda.is_available()
        )
        if self.use_amp and hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda")
        elif self.use_amp and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    @property
    def model(self) -> Any:
        return self._model

    @property
    def processor(self) -> Any:
        return self._processor

    def _init_optimizer_if_needed(self) -> None:
        """Instantiates AdamW optimizer if not injected for real PyTorch modules."""
        if (
            self._optimizer is None
            and self._model is not None
            and HAS_TORCH
            and torch is not None
            and isinstance(self._model, torch.nn.Module)
        ):
            params = list(self._model.parameters())
            if params:
                self._optimizer = torch.optim.AdamW(
                    params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )

    def train_step(self, batch: Dict[str, Any]) -> float:
        """Executes a single forward and backward optimization step.

        Args:
            batch: Dictionary containing 'pixel_values' and 'labels' tensors.

        Returns:
            float: Training loss scalar.
        """
        self._init_optimizer_if_needed()

        if self._model is None:
            return 0.0

        # Handle real PyTorch training step
        if HAS_TORCH and torch is not None and isinstance(self._model, torch.nn.Module):
            self._model.train()
            if self._optimizer:
                self._optimizer.zero_grad()

            pixel_values = batch.get("pixel_values")
            labels = batch.get("labels")

            if pixel_values is None or labels is None:
                return 0.0

            pixel_values = pixel_values.to(self.device_str)
            labels = labels.to(self.device_str)

            if self.use_amp and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                with torch.amp.autocast("cuda"):
                    outputs = self._model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            elif self.use_amp and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                with torch.cuda.amp.autocast():
                    outputs = self._model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            else:
                outputs = self._model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self._optimizer:
                    self.scaler.step(self._optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                if self._optimizer:
                    self._optimizer.step()

            if self._lr_scheduler:
                self._lr_scheduler.step()

            loss_val = float(loss.item()) if hasattr(loss, "item") else float(loss)
            self.global_step += 1
            return loss_val

        # If mocked model, simulate forward step
        if hasattr(self._model, "forward") or hasattr(self._model, "__call__"):
            outputs = self._model(batch)
            loss_val = getattr(outputs, "loss", 0.0)
            if hasattr(loss_val, "item"):
                loss_val = float(loss_val.item())
            self.global_step += 1
            return float(loss_val)

        self.global_step += 1
        return 0.0

    def evaluate(self, eval_dataloader: Optional[Any] = None) -> EvaluationReport:
        """Runs evaluation over the validation/test dataset and computes CER/WER.

        Args:
            eval_dataloader: Optional DataLoader to evaluate. If None, builds from eval_dataset.

        Returns:
            EvaluationReport: Aggregated error rates and per-language metrics.
        """
        if self._eval_dataset is None and eval_dataloader is None:
            return EvaluationReport()

        loader = eval_dataloader
        if loader is None and self._eval_dataset is not None:
            if HAS_TORCH and DataLoader is not None:
                loader = DataLoader(
                    self._eval_dataset,
                    batch_size=self.config.eval_batch_size,
                    shuffle=False,
                    collate_fn=self._collate_fn,
                )
            else:
                loader = [[self._eval_dataset[i] for i in range(len(self._eval_dataset))]]

        if self._model is not None and HAS_TORCH and torch is not None and isinstance(self._model, torch.nn.Module):
            self._model.eval()

        all_references: List[str] = []
        all_hypotheses: List[str] = []
        all_languages: List[str] = []

        for batch in loader:
            texts = batch.get("texts", [])
            languages = batch.get("languages", [])

            # Generate predictions using model and processor
            hypotheses: List[str] = []
            if self._model is not None and self._processor is not None and HAS_TORCH and torch is not None:
                pixel_values = batch.get("pixel_values")
                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device_str)
                    with torch.no_grad():
                        gen_kwargs = {
                            "max_new_tokens": self.config.model.max_sequence_length,
                        }
                        if hasattr(self._model, "config"):
                            if getattr(self._model.config, "decoder_start_token_id", None) is not None:
                                gen_kwargs["decoder_start_token_id"] = self._model.config.decoder_start_token_id
                            if getattr(self._model.config, "pad_token_id", None) is not None:
                                gen_kwargs["pad_token_id"] = self._model.config.pad_token_id
                            if getattr(self._model.config, "eos_token_id", None) is not None:
                                gen_kwargs["eos_token_id"] = self._model.config.eos_token_id
                        generated_ids = self._model.generate(
                            pixel_values,
                            **gen_kwargs,
                        )
                    if hasattr(self._processor, "batch_decode"):
                        decoded = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
                        hypotheses.extend(decoded)
                    else:
                        hypotheses.extend([""] * len(texts))
                else:
                    hypotheses.extend([""] * len(texts))
            else:
                # Fallback / mock hypothesis generation
                if hasattr(self._model, "generate_mock"):
                    hypotheses.extend(self._model.generate_mock(batch))
                else:
                    hypotheses.extend([""] * len(texts))

            all_references.extend(texts)
            all_hypotheses.extend(hypotheses[:len(texts)])
            all_languages.extend(languages[:len(texts)])

        self.last_eval_samples = [
            {"reference": r, "hypothesis": h, "language": l}
            for r, h, l in zip(all_references, all_hypotheses, all_languages)
        ]

        report = evaluate_predictions(
            references=all_references,
            hypotheses=all_hypotheses,
            languages=all_languages,
        )
        return report

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> Path:
        """Saves current training state and checkpoint metadata."""
        ckpt_id = f"checkpoint-step-{step}"
        meta = CheckpointMetadata(
            checkpoint_id=ckpt_id,
            model_name=self.config.model.model_name_or_path,
            model_version=self.config.model.model_version,
            languages=self.config.languages,
            epoch=epoch,
            step=step,
            training_config=self.config.to_dict(),
            metrics=metrics or {},
            device=self.device_str,
        )

        return save_training_checkpoint(
            output_dir=self.config.output_dir,
            model=self._model,
            processor=self._processor,
            metadata=meta,
            is_best=is_best,
        )

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """Executes full training loop over configured epochs.

        Args:
            num_epochs: Optional override for config.num_epochs.

        Returns:
            Dict[str, Any]: Summary dictionary of training history and final metrics.
        """
        start_epoch = self.current_epoch + 1
        epochs = num_epochs or self.config.num_epochs
        if self._train_dataset is None:
            raise ValueError("Cannot train: train_dataset is None.")

        # Build DataLoader
        if HAS_TORCH and DataLoader is not None:
            pin_mem = bool(self.device_str == "cuda" and torch is not None and torch.cuda.is_available())
            train_loader = DataLoader(
                self._train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=self._collate_fn,
                pin_memory=pin_mem,
            )
        else:
            train_loader = [[self._train_dataset[i] for i in range(len(self._train_dataset))]]

        if start_epoch > epochs:
            print(f"Training already completed up to Epoch {self.current_epoch} (target epochs: {epochs}). Nothing to run.", flush=True)
            return {
                "experiment_name": self.config.experiment_name,
                "total_steps": self.global_step,
                "total_epochs": self.current_epoch,
                "best_cer": self.best_cer,
                "history": self.training_history,
            }

        print(f"Starting training: Epoch {start_epoch} to {epochs} (total configured: {epochs}), device='{self.device_str}', dataset_size={len(self._train_dataset)}", flush=True)

        for epoch in range(start_epoch, epochs + 1):
            self.current_epoch = epoch
            epoch_loss = 0.0
            steps_in_epoch = 0
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(train_loader, start=1):
                step_start = time.time()
                loss = self.train_step(batch)
                step_duration_ms = (time.time() - step_start) * 1000
                epoch_loss += loss
                steps_in_epoch += 1

                if self.global_step % self.config.logging_steps == 0:
                    vram_str = ""
                    if HAS_TORCH and torch is not None and torch.cuda.is_available() and self.device_str.startswith("cuda"):
                        curr_vram = torch.cuda.memory_allocated() / (1024 ** 2)
                        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
                        vram_str = f" | VRAM: {curr_vram:.1f}/{peak_vram:.1f}MB"
                    print(
                        f"Epoch {epoch}/{epochs} | Step {self.global_step:4d} | Batch {batch_idx:4d}/{len(train_loader)} | Loss: {loss:.4f} | {step_duration_ms:.1f}ms/step{vram_str}",
                        flush=True,
                    )

            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / max(1, steps_in_epoch)
            throughput = len(self._train_dataset) / max(0.001, epoch_time)

            # Evaluate at epoch end
            eval_start = time.time()
            eval_report = self.evaluate()
            eval_duration = time.time() - eval_start

            print(
                f"Epoch {epoch} Complete in {epoch_time:.1f}s ({throughput:.1f} samp/s) | Avg Loss: {avg_loss:.4f} | Val CER: {eval_report.overall_cer:.4f} | Val WER: {eval_report.overall_wer:.4f} | Eval Time: {eval_duration:.1f}s",
                flush=True,
            )

            is_best = eval_report.overall_cer < self.best_cer
            if is_best:
                self.best_cer = eval_report.overall_cer
                print(f"  --> [*] New Best Validation CER: {self.best_cer:.4f}", flush=True)

            # Save epoch checkpoint
            saved_path = self.save_checkpoint(
                step=self.global_step,
                epoch=epoch,
                metrics={
                    "avg_loss": round(avg_loss, 4),
                    "val_cer": eval_report.overall_cer,
                    "val_wer": eval_report.overall_wer,
                    "per_language": eval_report.per_language,
                    "epoch_time_sec": round(epoch_time, 2),
                    "throughput_samples_sec": round(throughput, 2),
                },
                is_best=is_best,
            )

            self.training_history.append({
                "epoch": epoch,
                "step": self.global_step,
                "avg_loss": round(avg_loss, 4),
                "val_cer": eval_report.overall_cer,
                "val_wer": eval_report.overall_wer,
                "epoch_time_sec": round(epoch_time, 2),
                "throughput_samples_sec": round(throughput, 2),
                "checkpoint_path": str(saved_path),
            })

        return {
            "experiment_name": self.config.experiment_name,
            "total_steps": self.global_step,
            "total_epochs": epochs,
            "best_cer": self.best_cer,
            "history": self.training_history,
        }
