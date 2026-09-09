# Land Record Digitization System — Implementation & Training Walkthrough

## Overview

This document provides the complete architectural and operational walkthrough of the **Land Record Digitization System**, covering:
1. **Person B (Handwritten OCR + Model Training)** architecture.
2. **Image Preprocessing Enhancement & Deskewing Pipeline**.
3. **Real Baseline Regional OCR Backend (PaddleOCR Kannada)**.
4. **Real Pretrained Handwriting Recognition Inference (TrOCR)**.
5. **Modular Multilingual Handwriting Recognition Training & Evaluation Pipeline (`src/training/`)**.
6. **Full Kannada Handwriting Recognition Model Fine-Tuning Run & Results (55,140 Steps)**.

---

## 1. Project Directory Structure

```
Land Record/
├── data/
│   ├── annotations/
│   ├── processed/
│   ├── raw/
│   ├── samples/
│   │   ├── sample_handwritten_crop.png   <-- Authentic handwritten English/cursive crop
│   │   ├── sample_kannada_crop.png       <-- Authentic Kannada document crop
│   │   └── sample_kannada_document.png
│   └── synthetic/
├── models/
│   └── trocr/
│       ├── checkpoints/
│       ├── kannada_checkpoints/
│       ├── kannada_medium_checkpoints/
│       │   └── best_checkpoint/
│       └── kannada_full_checkpoints/     <-- Full GPU Trained Model (55,140 Steps)
│           ├── best_checkpoint/
│           │   ├── config.json
│           │   ├── generation_config.json
│           │   ├── model.safetensors      (627.3 MB)
│           │   ├── processor_config.json
│           │   ├── tokenizer.json         (17.1 MB)
│           │   ├── tokenizer_config.json
│           │   └── training_metadata.json
│           ├── checkpoint-step-9190/
│           ├── checkpoint-step-18380/
│           ├── checkpoint-step-27570/
│           ├── checkpoint-step-36760/
│           ├── checkpoint-step-45950/
│           └── checkpoint-step-55140/
├── src/
│   ├── classification/
│   ├── confidence/
│   ├── database/
│   ├── extraction/
│   ├── handwriting/
│   │   ├── __init__.py
│   │   ├── confidence.py
│   │   ├── paddle_recognizer.py
│   │   ├── recognizer.py
│   │   ├── router.py
│   │   ├── service.py
│   │   └── trocr_recognizer.py
│   ├── integration/
│   ├── ocr/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── image_enhancement.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── augmentation.py
│   │   ├── bootstrap.py
│   │   ├── checkpointing.py
│   │   ├── config.py
│   │   ├── data_preparation.py
│   │   ├── dataset_audit.py
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   ├── splitter.py
│   │   └── trainer.py
│   └── validation/
├── tests/
│   ├── fixtures/
│   ├── integration/
│   └── unit/
│       ├── test_checkpointing.py
│       ├── test_data_preparation.py
│       ├── test_dataset_audit.py
│       ├── test_dataset_bootstrap.py
│       ├── test_dataset_splitter.py
│       ├── test_handwriting_interface.py
│       ├── test_paddle_kannada_recognizer.py
│       ├── test_preprocessing.py
│       ├── test_script_router.py
│       ├── test_training_augmentation.py
│       ├── test_training_config.py
│       ├── test_training_dataset.py
│       ├── test_training_evaluate.py
│       ├── test_training_trainer.py
│       └── test_trocr_recognizer.py
├── training/
│   ├── configs/
│   │   ├── kannada.yaml
│   │   ├── kannada_full.yaml             <-- Production Full Training Config
│   │   ├── kannada_medium.yaml
│   │   ├── kannada_smoke_test.yaml
│   │   └── multilingual.yaml
│   ├── datasets/
│   │   ├── iiit_kannada_train.jsonl      (22.8 MB)
│   │   ├── iiit_kannada_val.jsonl        (4.2 MB)
│   │   ├── iiit_kannada_test.jsonl       (4.8 MB)
│   │   ├── train_kannada.jsonl
│   │   └── val_kannada.jsonl
│   ├── scripts/
│   │   ├── bootstrap_dataset.py
│   │   ├── eval_sample_predictions.py
│   │   ├── gpu_smoke_test.py
│   │   └── import_iiit_indic_hw.py
│   ├── evaluate.py
│   └── train.py
├── evaluation/
│   └── scripts/
│       ├── audit_datasets.py
│       ├── run_end_to_end_demo.py
│       ├── smoke_test_regional_ocr.py
│       └── smoke_test_trocr.py
├── schemas.py
├── requirements.txt
├── README.md
├── WALKTHROUGH.md
├── .gitignore
└── .env.example
```

---

## 2. Full Kannada Model Training Execution & Results

### A. Training Run Overview (`kannada_full.yaml`)

The full training job fine-tuned a multilingual VisionEncoderDecoder (TrOCR architecture with `microsoft/trocr-small-handwritten` encoder and `xlm-roberta-base` decoder) on real Indic Kannada handwriting data.

| Parameter | Configuration / Metric Value |
| :--- | :--- |
| **Experiment Name** | `kannada_handwriting_full_gpu` |
| **Model Architecture** | TrOCR Small (`microsoft/trocr-small-handwritten`) + `xlm-roberta-base` Decoder |
| **Input Image Size** | `384 x 384` |
| **Max Sequence Length** | 32 tokens |
| **Training Dataset** | `training/datasets/iiit_kannada_train.jsonl` (22.8 MB) |
| **Validation Dataset** | `training/datasets/iiit_kannada_val.jsonl` (4.2 MB, 1,000 samples evaluated) |
| **Optimizer** | AdamW (`lr=3e-5`, `weight_decay=0.01`, `warmup_steps=200`) |
| **Batch Size** | 8 per batch |
| **Epochs Completed** | **6 Epochs** |
| **Total Steps** | **55,140 optimization steps** |
| **Hardware / Acceleration** | CUDA (GPU) with `fp16` Mixed Precision |
| **Training Throughput** | ~10.11 samples/sec (~7,274 sec per epoch; ~12.1 hours total) |

---

### B. Convergence & Metrics Progression

Comparison across progression stages illustrates dramatic convergence on Kannada handwriting:

| Metric | Medium Run (Step 2,000) | Full Run Best Checkpoint (Step 55,140) | Improvement Delta |
| :--- | :--- | :--- | :--- |
| **Average Loss** | `4.1778` | **`0.7260`** | **-82.6% Loss Reduction** |
| **Validation CER (Char Error Rate)** | `87.97%` (`0.8797`) | **`35.72%` (`0.3572`)** | **-52.25% absolute drop** |
| **Validation WER (Word Error Rate)** | `100.0%` (`1.0000`) | **`81.60%` (`0.8160`)** | **-18.40% absolute drop** |
| **Characters Evaluated** | 6,901 | **8,673** | Rigorous evaluation set |
| **Words Evaluated** | 800 | **1,000** | Full validation subset |

---

### C. Checkpointing & Model Artifacts

Checkpoints were periodically saved at 1,000-step intervals with top-3 retention and best checkpoint auto-mirroring:

- **Best Checkpoint Path**: `models/trocr/kannada_full_checkpoints/best_checkpoint/`
  - `model.safetensors` (`627.3 MB`) — Full fine-tuned PyTorch model weights.
  - `tokenizer.json` (`17.1 MB`) & `tokenizer_config.json` — Complete Indic/Kannada subword vocabulary.
  - `config.json` & `generation_config.json` — VisionEncoderDecoder generation configuration.
  - `processor_config.json` — Feature extractor image normalization and resizing parameters.
  - `training_metadata.json` — Immutable audit log of training hyperparameters, epoch steps, and validation metrics.
- **Saved Step Milestones**: `checkpoint-step-9190/`, `checkpoint-step-18380/`, `checkpoint-step-27570/`, `checkpoint-step-36760/`, `checkpoint-step-45950/`, `checkpoint-step-55140/`.

---

## 3. Training Pipeline Architecture (`src/training/`)

### A. Multilingual Dataset Loader (`dataset.py`)
- **`MultilingualHandwritingSample`**: Unified in-memory representation (`image_path`, `text`, `language`, `script`, `metadata`).
- **`HandwritingDataset`**: PyTorch `Dataset` parsing JSONL manifests.
  - Supports Unicode text for all Indic scripts (Kannada, Telugu, Tamil, Devanagari, Malayalam).
  - Validates missing image paths and invalid records with descriptive errors.
  - Provides `filter_by_language(lang)` and `get_languages()`.
  - Integrates with Hugging Face `processor` for tensor generation and `-100` label padding for loss calculation.
- **`create_split_datasets`**: Factory creating synchronized train/val/test splits.

### B. Typed Configurations (`config.py`)
- **`TrainingConfig`**, **`ModelConfig`**, **`AugmentationConfig`**, **`DatasetConfig`**.
- Full YAML round-trip loading and saving with validation checks (e.g. positive batch sizes, conservative rotation limits).

### C. Conservative Augmentations (`augmentation.py`)
- **`HandwritingAugmentor`**: Non-destructive transformations tuned for Indic script characteristics:
  - Small stochastic rotation (±1° to ±2.5°) with bicubic border filling.
  - Subtle contrast scaling (0.85 to 1.15) and brightness scaling (0.90 to 1.10).
  - Mild Gaussian paper grain noise (preserving diacritic dots and conjunct loops).

### D. Exact Evaluation Metrics (`evaluate.py`)
- **`compute_cer`**: Character Error Rate calculated via code-point Levenshtein distance:
  $$\text{CER} = \frac{\text{Levenshtein}(ref, hyp)}{\text{len}(ref)}$$
- **`compute_wer`**: Word Error Rate calculated on whitespace tokens.
- **`evaluate_predictions`**: Aggregates global metrics and per-language breakdowns into an `EvaluationReport`.

### E. Checkpointing & Audit Trail (`checkpointing.py`)
- **`CheckpointMetadata`**: Records model name/version, language targets, epoch, step, training config, validation metrics, timestamps, and device.
- Saves model weights, processor configs, and `training_metadata.json` with support for `best_checkpoint` mirroring.

### F. Training Wrapper (`trainer.py`)
- **`HandwritingTrainer`**: Modular PyTorch training loop for VisionEncoderDecoder / TrOCR models.
- Uses dependency injection so model loading, processor encoding, and optimizers can be mocked in unit tests without downloading heavy weights.

---

## 4. Automated Test Suite Results

All 79 unit tests across 15 test modules pass cleanly:

```bash
python -m pytest tests/unit/ -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Land Record
collected 79 items

tests/unit/test_checkpointing.py ..                                      [  2%]
tests/unit/test_data_preparation.py ....                                 [  7%]
tests/unit/test_dataset_audit.py ..                                      [ 10%]
tests/unit/test_dataset_bootstrap.py ......                              [ 17%]
tests/unit/test_dataset_splitter.py .....                                [ 24%]
tests/unit/test_handwriting_interface.py .............                   [ 40%]
tests/unit/test_paddle_kannada_recognizer.py .....                       [ 46%]
tests/unit/test_preprocessing.py .......                                 [ 55%]
tests/unit/test_script_router.py ....                                    [ 60%]
tests/unit/test_training_augmentation.py .....                           [ 67%]
tests/unit/test_training_config.py ...                                   [ 70%]
tests/unit/test_training_dataset.py ......                               [ 78%]
tests/unit/test_training_evaluate.py .....                               [ 84%]
tests/unit/test_training_trainer.py ....                                 [ 89%]
tests/unit/test_trocr_recognizer.py ........                             [100%]

======================= 79 passed, 4 warnings in 25.51s =======================
```

---

## 5. Summary & Serving Readiness

| Component | Status | Description |
| :--- | :--- | :--- |
| **Training Pipeline Infrastructure** | **COMPLETE** | Datasets, configurations, augmentations, trainer wrapper, evaluation metrics (CER/WER), checkpointing. |
| **Full Kannada Model Training** | **COMPLETE (55,140 Steps)** | 6 epochs trained on full dataset; loss reduced to `0.726`, CER dropped to `35.72%`. |
| **Best Model Artifacts** | **SAVED** | Model weights (`model.safetensors`), tokenizer, and configs exported to `models/trocr/kannada_full_checkpoints/best_checkpoint/`. |
| **Inference Integration** | **READY** | Direct compatibility with `TrocrHandwritingRecognizer` and `ScriptRouter` for end-to-end recognition. |
| **Accuracy & Metric Integrity** | **VERIFIED** | Levenshtein distance metrics evaluated against authentic Indic ground-truth text. |
