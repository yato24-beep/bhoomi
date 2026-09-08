# Land Record Digitization System

A modular pipeline for digitizing, recognizing, extracting, and indexing archival land records and cadastral documents.

---

## Project Overview

The Land Record Digitization System transforms scanned and photographed land record documents into structured, queryable data. The system is designed with clear modular interfaces to enable independent development and integration across three primary ownership boundaries.

---

## Architecture & Ownership Boundaries

```
[ Scanned Documents / Images ]
              │
              ▼
   ┌─────────────────────────────────────────┐
   │ Person A: Preprocessing & Layout        │
   │ - Document Cleaning & Deskewing         │
   │ - Document Classification               │
   │ - Region & Layout Detection             │
   └─────────────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────────┐
   │ Person B: Handwritten OCR & Training    │
   │ - Printed & Handwritten Text OCR (TrOCR)│
   │ - Handwriting vs. Printed Routing       │
   │ - Model Fine-Tuning & Evaluation        │
   └─────────────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────────┐
   │ Person C: Extraction & Database         │
   │ - Key-Value & Entity Extraction         │
   │ - Field Validation & Confidence Scoring │
   │ - Database Storage & Search Integration │
   └─────────────────────────────────────────┘
              │
              ▼
    [ Structured Land Records ]
```

### 1. Person A — Preprocessing, Classification & Document Layout
- **Scope:** Document ingestion, enhancement (deskew, denoising, binarization), document type classification, line/word segmentation, and layout analysis.
- **Key Modules:** `src/preprocessing/`, `src/classification/`.

### 2. Person B — Handwritten OCR & Training
- **Scope:** Optical character recognition for printed and handwritten scripts (including fine-tuning Vision-Encoder-Decoder architectures like TrOCR), handwriting classification, dataset curation, and model training pipelines.
- **Key Modules:** `src/ocr/`, `src/handwriting/`, `models/trocr/`, `training/`.

### 3. Person C — Information Extraction, Validation & Database Integration
- **Scope:** Domain-specific entity and key-value extraction, confidence aggregation, rule-based and checksum validations, database schema storage, and API/integration layers.
- **Key Modules:** `src/extraction/`, `src/validation/`, `src/confidence/`, `src/database/`, `src/integration/`.

---

## Repository Structure

```
.
├── data/                    # Dataset storage (raw, processed, synthetic, annotations, samples)
├── models/                  # Model weight storage and checkpoints (e.g., TrOCR)
├── src/                     # Source modules
│   ├── preprocessing/       # Document cleaning and segmentation
│   ├── ocr/                 # OCR engine interfaces and inference
│   ├── classification/      # Document and script classification
│   ├── handwriting/         # Handwriting detection and routing
│   ├── extraction/          # Entity and field extraction
│   ├── validation/          # Field validation and business rules
│   ├── confidence/          # Confidence scoring and metrics
│   ├── database/            # Database models and access layer
│   └── integration/         # Integration pipelines and API adapters
├── training/                # Training pipelines, datasets, scripts, configs, and evaluation
├── tests/                   # Unit, integration, and fixture tests
├── evaluation/              # Standalone evaluation scripts and benchmarks
├── schemas.py               # Shared data schemas and models
├── requirements.txt         # Project dependencies
├── .env.example             # Environment configuration template
└── .gitignore               # Version control ignore rules
```

---

## Getting Started

1. **Clone the repository and set up environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   # source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```

3. **Run Unit Test Suite:**
   ```bash
   python -m pytest tests/unit/ -v
   ```

---

## Multilingual Handwriting Training Pipeline (`src/training/`)

The system includes an extensible training and evaluation pipeline for fine-tuning Vision-Encoder-Decoder / TrOCR architectures on Indic handwriting (Kannada, Telugu, Tamil, Hindi, Malayalam, and others).

### 1. Dataset Structure & JSONL Manifest Format

Training, validation, and test sets are organized as line-delimited JSON (`.jsonl`) manifests:

```
training/datasets/
├── train_kannada.jsonl
├── val_kannada.jsonl
└── test_kannada.jsonl
```

Each record in a manifest file specifies:

```json
{"image": "data/raw/kannada/sample_001.png", "text": "ಕರ್ನಾಟಕ ಕಂದಾಯ ಇಲಾಖೆ ಸರ್ವೆ ನಂ ೧೪೨", "language": "kannada", "script": "Kannada"}
```

**Fields:**
- `image` (or `image_path`): Relative or absolute path to the text line/word crop image.
- `text` (or `ground_truth`): Unicode ground truth transcription.
- `language`: Standardized language identifier (`"kannada"`, `"telugu"`, `"tamil"`, `"hindi"`, `"malayalam"`, etc.).
- `script`: Canonical script name (`"Kannada"`, `"Telugu"`, `"Tamil"`, `"Devanagari"`, `"Malayalam"`, `"Latin"`).
- `metadata` (optional): Arbitrary audit dictionary (e.g. source document ID, annotator ID).

---

### 2. How to Prepare Kannada & Indic Training Data

1. **Place cropped line/word images** into `data/raw/<language>/` (e.g. `data/raw/kannada/`).
2. **Generate manifest files** mapping image paths to their ground truth transcriptions.
3. **Verify dataset integrity with dry-run:**
   ```bash
   python training/train.py --config training/configs/kannada.yaml --dry-run
   ```

---

### 3. Extensibility to Additional Regional Languages

The training pipeline does not hardcode language-specific assumptions. To add support for another regional language (e.g. Telugu, Tamil, Hindi):

1. Create a configuration file in `training/configs/<language>.yaml` (or use `training/configs/multilingual.yaml` for joint multi-script training).
2. Specify the target language tags in `languages: [...]`.
3. Provide corresponding train/val manifests.
4. The loader (`HandwritingDataset`), augmentor (`HandwritingAugmentor`), trainer (`HandwritingTrainer`), and evaluator (`evaluate_predictions`) automatically partition and track metrics on a per-language basis.

---

### 4. Dataset Bootstrap & Annotation Workflow

The system provides an automated discovery and template generation workflow to bridge raw scans to training manifests without fabricating labels:

```
[ Raw Scans / Crops in data/raw/<language>/ ]
                      │
                      ▼
       [ python training/scripts/bootstrap_dataset.py --all ]
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
[ Verified Annotated Pairs ]   [ Unannotated Crops ]
        │                           │
        ▼                           ▼
[ <language>_all.jsonl ]     [ <language>_template.jsonl ]
        │                           │
        │                           ▼  (Human Annotation)
        │                    [ Fill "text" ground truth ]
        └─────────────┬─────────────┘
                      │
                      ▼
[ src.training.data_preparation.clean_and_save_manifest ]
                      │
                      ▼
[ src.training.splitter.split_and_save_manifests ]
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
  [ train.jsonl ] [ val.jsonl ] [ test.jsonl ]
```

#### Commands:
- **Scan and Bootstrap All Indic Languages:**
  ```bash
  python training/scripts/bootstrap_dataset.py --all
  ```
- **Scan Specific Language Directory (e.g. Kannada):**
  ```bash
  python training/scripts/bootstrap_dataset.py --language kannada
  ```
- **Read-Only Scan Mode:**
  ```bash
  python training/scripts/bootstrap_dataset.py --scan
  ```

---

### 5. Data Type Classification & Authenticity Rules

To prevent model corruption, all samples track a `source_type` metadata field:
- **`real_handwriting`**: Genuine pen-drawn or cursive text from archival land records.
- **`printed`**: Typeset/typography characters (e.g. standard government forms).
- **`synthetic`**: Programmatically generated or font-rendered handwriting simulations.

---

### 6. Training, Preparation & Audit Commands

- **Run Dataset Readiness Audit:**
  ```bash
  python evaluation/scripts/audit_datasets.py
  ```

- **Clean and Validate Raw Manifest:**
  ```python
  from src.training.data_preparation import clean_and_save_manifest
  clean_and_save_manifest("data/annotations/kannada_template.jsonl", "training/datasets/train_kannada.jsonl")
  ```

- **Stratified Train / Val / Test Partitioning:**
  ```python
  from src.training.splitter import split_and_save_manifests
  split_and_save_manifests("cleaned_data.jsonl", output_dir="training/datasets", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
  ```

- **Dry-run pipeline check (validates manifests and models without heavy compute):**
  ```bash
  python training/train.py --config training/configs/kannada.yaml --dry-run
  ```

- **Run Kannada handwriting fine-tuning:**
  ```bash
  python training/train.py --config training/configs/kannada.yaml --epochs 5 --batch-size 4
  ```

- **Run Multilingual Indic handwriting fine-tuning:**
  ```bash
  python training/train.py --config training/configs/multilingual.yaml --epochs 5
  ```

- **Evaluate a model checkpoint on a test manifest:**
  ```bash
  python training/evaluate.py --config training/configs/kannada.yaml --manifest training/datasets/test_kannada.jsonl
  ```

- **Evaluate a specific language subset and export JSON metrics:**
  ```bash
  python training/evaluate.py --config training/configs/multilingual.yaml --manifest training/datasets/test_multilingual.jsonl --language kannada --output-report evaluation/reports/kannada_results.json
  ```


