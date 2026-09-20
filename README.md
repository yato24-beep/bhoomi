# Land Record Digitization & Multimodal AI Platform

An end-to-end, production-grade multimodal document intelligence platform for archival land records, RTC/Pahani forms, cadastral maps, and government deeds. Built with **FastAPI**, **Next.js 14**, **PaddleOCR**, **Fine-tuned TrOCR**, and **Gemini Multimodal Semantics**.

---

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [OCR Routing & Modality Detection](#2-ocr-routing--modality-detection)
3. [Semantic Pipeline Architecture](#3-semantic-pipeline-architecture)
4. [Karnataka Canonical Schema](#4-karnataka-canonical-schema)
5. [Cadastral Gating & Document Classification](#5-cadastral-gating--document-classification)
6. [Provenance & Spatial Grounding](#6-provenance--spatial-grounding)
7. [Confidence Semantics & Calibration Requirements](#7-confidence-semantics--calibration-requirements)
8. [Human Review Workflow & Audit Immutability](#8-human-review-workflow--audit-immutability)
9. [Environment Variables](#9-environment-variables)
10. [Quick Start: Running Backend & Frontend](#10-quick-start-running-backend--frontend)
11. [Running the Test Suites](#11-running-the-test-suites)
12. [Running Real-Document Semantic Evaluation](#12-running-real-document-semantic-evaluation)
13. [TrOCR Integration Guide (Post-Training)](#13-trocr-integration-guide-post-training)
14. [Known Limitations & Current Verification Baseline](#14-known-limitations--current-verification-baseline)

---

## 1. System Architecture

The digitization platform coordinates image preprocessing, vision layout gating, script-aware OCR routing, semantic extraction, deterministic validation, confidence scoring, and an immutable human review state machine:

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Uploaded Document (PDF / TIFF / Image)            │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 1. Ingestion & Robustness Gate                                         │
 │    - Multi-page rendering, file-size enforcement (<=50MB)             │
 │    - Image enhancement (CLAHE, deskew, noise removal)                  │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 2. Layout Gating & Document Classification                             │
 │    - Detects: karnataka_rtc, mutation_extract, sale_deed, non_cadastral│
 │    - Pre-semantic cadastral gate: suppresses extraction if non-cadastral│
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 3. OCR Routing Engine                                                  │
 │    - Printed Kannada/English     ──► PaddleOCR                         │
 │    - Handwritten Kannada/English ──► Vision Transformer (TrOCR)        │
 │    - Outputs: reading-order text + bboxes [ymin, xmin, ymax, xmax]     │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 4. Semantic Extraction Layer                                           │
 │    - Primary Engine: Gemini (gemini-3.1-flash-lite / structured JSON) │
 │    - Deterministic Fallback: RuleSemanticEngine (regex + NER)          │
 │    - Regional Aliasing: MH (7/12) & TN (Patta) mapped to KA schema     │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 5. Normalization, Grounding & Provenance                               │
 │    - Kannada numerals (೦-೯ ──► 0-9), land units to Hectares/Acres-Guntas│
 │    - Exact bbox & source region linking (immutable raw OCR text)       │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 6. Validation & Confidence Scoring                                     │
 │    - Survey number format checks, owner name completeness               │
 │    - Heuristic Confidence vs. Calibrated Confidence (requires N >= 50) │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 7. Human Review Decision & Audit Trail                                 │
 │    - Flags low-confidence, invalid, or conflicting candidates          │
 │    - Actions: ACCEPT, CORRECT, REJECT (preserves original raw OCR text)│
 │    - Web UI: interactive review queue with document viewer & diff      │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 2. OCR Routing & Modality Detection

OCR execution is script and modality aware:
- **`LanguageScriptRouter` (`src/handwriting/router.py`)**:
  - Classifies text regions into `printed` vs `handwritten` and scripts (`kannada`, `english`, `telugu`, `tamil`, `hindi`).
  - Routes printed Kannada text to **EasyOCR** (primary, 3.06% CER) or **PaddleOCR** (`ppocr_v4_kannada` / `ppocr_v4_en`).
  - Routes handwritten Kannada text to **TrOCR Checkpoint-12000** (`TrOCR12000KannadaRecognizer`).
- **Reading Order Sorting**: Regions are spatially sorted top-to-bottom, left-to-right with line grouping.
- **Bounding Box Normalization**: Bounding boxes are stored as `[ymin, xmin, ymax, xmax]` normalized coordinates (0 to 1000 or absolute pixels), with `region_id` tags for unambiguous provenance linkage.

---

## 3. Semantic Pipeline Architecture

The semantic layer translates noisy, unsegmented OCR regions into canonical structured land records.

- **Primary Engine (`GeminiSemanticEngine`)**:
  - Calls Google GenAI (`gemini-3.1-flash-lite`) using strict Pydantic structured output schemas (`LandRecordExtractionResponse`).
  - Transmits OCR regions with reading-order identifiers and spatial coordinates.
  - Automatically recovers from rate limits, timeouts, and JSON decode errors.
- **Deterministic Engine (`RuleSemanticEngine`)**:
  - Standalone regex, keyword proximity, and heuristic parser for offline environments or when semantic AI calls time out or fail.
- **Unified Interface (`SemanticPipeline`)**:
  - Encapsulates engine fallback logic: if Gemini fails or times out, the pipeline falls back to `RuleSemanticEngine`, recording a fallback warning in document warnings without crashing the pipeline.

---

## 4. Karnataka Canonical Schema

Karnataka RTC (Record of Rights, Tenancy, and Crops / Pahani) is the **primary canonical schema**:

| Field Name | Description | Example Canonical Output |
| :--- | :--- | :--- |
| `survey_number` | Core survey identifier (Survey/Hissa) | `"124/2A"`, `"45/1"` |
| `hissa_number` | Sub-division / Hissa number | `"2A"`, `"1"` |
| `owner_name` | Primary khatedar / landowner name | `"ರಾಮಪ್ಪ (Ramappa)"` |
| `owner_father_name` | Father or husband of khatedar | `"ಭೀಮಪ್ಪ (Bheemappa)"` |
| `total_extent` | Standardized total parcel area | `"4 Acres 12 Guntas"` |
| `cultivable_area` | Usable agricultural area | `"4 Acres 00 Guntas"` |
| `pot_kharab` | Uncultivable land area | `"0 Acres 12 Guntas"` |
| `village` | Revenue village name | `"ಯಲಹಂಕ (Yelahanka)"` |
| `hobli` | Sub-tehsil administrative cluster | `"ಯಲಹಂಕ ಹೋಬಳಿ"` |
| `taluk` | Tehsil / Taluk | `"ಬೆಂಗಳೂರು ಉತ್ತರ"` |
| `district` | District | `"ಬೆಂಗಳೂರು ನಗರ"` |
| `soil_type` | Soil classification (Red, Black, Sandy) | `"ಕಪ್ಪು ಮಣ್ಣು (Black soil)"` |
| `land_revenue` | Assessment tax payable (Jodi/Kandaya) | `"Rs. 18.50"` |

### Regional Aliasing (Maharashtra & Tamil Nadu)
Records from other states are mapped cleanly into this canonical model via explicit alias registries:
- **Maharashtra (7/12 Extract)**:
  - `Gat No / Khasra No` ──► mapped to canonical `survey_number`.
  - `Pot Kharaba` ──► `pot_kharab`.
  - `Bhogvatdar` ──► `owner_name`.
  - `Gaon / Taluka` ──► `village` / `taluk`.
- **Tamil Nadu (Patta/Chitta)**:
  - `Patta No / Survey No` ──► `survey_number`.
  - `Urimaiyalargal` ──► `owner_name`.
  - `Nanjai / Punjai Area` ──► `total_extent`.

---

## 5. Cadastral Gating & Document Classification

Before extracting land-record properties, every document passes through `DocumentLayoutGate`:
1. **Classification**: Evaluates whether the document is a valid cadastral instrument (`karnataka_rtc`, `mutation_extract`, `sale_deed`, `patta_passbook`) or a non-cadastral document (`historical_narrative`, `identity_card`, `unrelated_invoice`, `general_letter`).
2. **Gating Logic**:
   - If `is_cadastral == False`: Land record semantic extraction is **suppressed**. The pipeline does not fabricate or guess cadastral values.
   - The returned `LandRecordDocument` has `classification_label = "not_land_record"`, `extracted_fields = {}`, and an explicit warning explaining the suppression.

---

## 6. Provenance & Spatial Grounding

Every extracted field is grounded in physical document regions:
- **`source_region_id`**: Integer identifier linking the field back to the source OCR region.
- **`source_page`**: Index of the document page (1-indexed).
- **`raw_ocr_text`**: The verbatim, unedited text read by the OCR model.
- **`normalized_value`**: The normalized value (e.g. Kannada numerals `೦-೯` converted to Western digits `0-9`, area unified).
- **`bbox`**: Coordinate bounding box `[ymin, xmin, ymax, xmax]` representing the exact spatial bounding zone on the source image.
- **Audit Immutability**: Even if a human reviewer corrects a field, the `raw_ocr_text` and `source_region_id` remain preserved in the audit log for complete forensic traceability.

---

## 7. Confidence Semantics & Calibration Requirements

Field confidence represents the system's certainty in its predictions:
- **Heuristic Confidence (`confidence`)**: Computed from token OCR probabilities, character perplexity, and format match heuristics ($0.0 \le c \le 1.0$).
- **Calibrated Confidence (`calibrated_confidence`)**:
  - Strict rule: **Never fabricate calibration curves or fit calibrations on unverified data.**
  - If held-out labeled verification samples $N < 50$: Calibrated confidence is set to `null` / `None`, and `confidence_state = "UNCALIBRATED"`.
  - When $N \ge 50$ authenticated labeled samples are supplied: Isotonic regression or Platt sigmoid scaling is fitted, computing Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

---

## 8. Human Review Workflow & Audit Immutability

When a field triggers human review (e.g., confidence below threshold, failed format validation, conflicting candidate values), it is flagged for review:
1. **Review Item Ingestion**: `ReviewState.NEEDS_REVIEW` added with reason code (`LOW_CONFIDENCE`, `VALIDATION_FAILED`, `CONFLICTING_CANDIDATE`).
2. **Review Actions**:
   - **`ACCEPT`**: Reviewer approves extracted value. Status becomes `ACCEPTED`.
   - **`CORRECT`**: Reviewer supplies correct value. The active field's `normalized_value` updates to the correction, `raw_ocr_text` remains untouched, status becomes `CORRECTED`, and full reviewer ID, notes, and timestamp are written to the audit log.
   - **`REJECT`**: Reviewer rejects invalid/hallucinated field. Field is marked `REJECTED`.
3. **Audit Immutability**: All decisions are recorded in `audit_trail` records preserving forensic document history.

---

## 9. Environment Variables

Create `.env` from `.env.example`:

```bash
# ==============================================================================
# Semantic Layer & GenAI Configuration
# ==============================================================================
# Engine to use: 'gemini' for multimodal AI, or 'rule' for deterministic fallback
SEMANTIC_ENGINE=gemini

# Google Gemini API Key (Required when SEMANTIC_ENGINE=gemini)
GEMINI_API_KEY=your_gemini_api_key_here

# Model name for semantic extraction
SEMANTIC_MODEL_NAME=gemini-3.1-flash-lite

# HTTP timeout in seconds for GenAI calls
SEMANTIC_TIMEOUT_SECONDS=15.0

# Minimum labeled samples required to activate statistical calibration
CALIBRATION_MIN_SAMPLES=50

# ==============================================================================
# Core Platform & Storage
# ==============================================================================
DATABASE_URL=sqlite:///./doc_platform.db
STORAGE_TYPE=local
STORAGE_LOCAL_DIR=./storage/uploads
MAX_FILE_SIZE_MB=50
```

---

## 10. Quick Start: Running Backend & Frontend

### Backend Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and supply your GEMINI_API_KEY

# 3. Start FastAPI application
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
- Web Application: `http://localhost:3000`

---

## 11. Running the Test Suites

Execute the comprehensive test suites:

```bash
# 1. Semantic Layer Unit Tests (38/38 passing)
pytest tests/unit/test_semantic_layer.py tests/unit/test_semantic_ai_layer.py tests/unit/test_semantic_improvements.py -v

# 2. Confidence Calibration Tests (5/5 passing)
pytest tests/unit/test_confidence_calibration.py -v

# 3. End-to-End Pipeline Integration Tests (6/6 passing)
pytest tests/integration/test_pipeline_e2e_semantic.py -v

# 4. Human Review & Audit Immutability Tests (3/3 passing)
pytest tests/integration/test_review_workflow.py -v

# 5. API Data Contract Tests (4/4 passing)
pytest tests/integration/test_api_contracts.py -v

# 6. Person C Validation & Normalization Tests (25/25 passing)
pytest tests/unit/test_extraction.py tests/unit/test_normalization.py tests/unit/test_rules.py tests/unit/test_gis.py tests/unit/test_duplicates.py tests/unit/test_confidence.py tests/unit/test_person_c_adapter.py -v

# 7. Run Complete Core Regression Suite
pytest tests/unit/test_semantic_layer.py tests/unit/test_semantic_ai_layer.py tests/unit/test_semantic_improvements.py tests/unit/test_confidence_calibration.py tests/integration/test_pipeline_e2e_semantic.py tests/integration/test_review_workflow.py tests/integration/test_api_contracts.py -v
```

---

## 12. Running Real-Document Semantic Evaluation

To evaluate semantic extraction accuracy against ground-truth Karnataka records:

```bash
python -m src.evaluation.eval_semantic_real
```
Metrics produced:
- **Field Accuracy**: Exact/normalized semantic match against ground truth.
- **Missing-Field Rate**: Target ground truth fields not extracted.
- **Wrong-Field Rate**: Extracted fields with incorrect values.
- **False-Positive Rate**: Extracted fields not present in source document.
- **Provenance Accuracy**: Spatial bounding box correctness.
- **Human-Review Trigger Rate**: Percentage of fields flagged for human review.

---

## 13. Handwritten TrOCR Checkpoint-12000 Integration & Verification

Checkpoint-12000 is wired into production via `TrOCR12000KannadaRecognizer` and `LanguageScriptRouter`:

- **Location**: `models/trocr/checkpoint-12000/` (`config.json`, `generation_config.json`, `model.safetensors`, `trainer_state.json`)
- **Base Image Processor**: `models/trocr/experimental/iitb_kannada_v002/` (ViTImageProcessor, 224x224)
- **Tokenizer**: `Chakita/KannadaBERT` (100k vocabulary, genuine Unicode coverage)
- **Decode Logic**: Strips only `pad_token_id`, `bos_token_id`, `eos_token_id` before decoding. Special tokens like `<unk>` are visibly surfaced rather than silently dropped (`skip_special_tokens=False`).

### Verified Empirical Performance (Settled Benchmark Numbers)
1. **In-Distribution Performance** (Pilot set: 400 IIIT isolated Kannada dictionary words):
   - **CER**: **4.86%**
   - **WER**: **16.50%**
   - **Exact Match**: **83.50%**
2. **Out-of-Distribution Performance** (Locked benchmark: 13 real archival land-record crops):
   - **CER**: **93.73%** (matches training baseline ~93.06%)
   - **WER**: **96.08%** (matches training baseline ~96.92%)
   - **Exact Match**: **15.38%** (2 / 13 crops — isolated word crops match; multi-word lines fail)

### Known Failure Mode (Language Model Priors)
- **Trained on isolated words only**: Checkpoint-12000 was trained exclusively on IIIT-INDIC-HW-WORDS isolated dictionary entries. It was **never exposed to multi-word cursive archival lines** during training.
- **Language prior hallucination**: On real archival lines, the RoBERTa decoder outputs plausible-sounding but wrong Kannada words (e.g. `"ಸ್ಪರ್ಧಿಸಿಕೊಂಡಿದ್ದು"`, `"ಘಟ್ನಿಸಿಕೊಳ್ಳುವುದಕ್ಕೂ"`, `"ತಪ್ಪಿಸಿಕೊಳ್ಳುವುದಕ್ಕೂ"`) driven by language-model priors, rather than reading actual strokes.

---

## 14. Known Limitations & Production Guidance

- **Handwritten Line Recognition**: Checkpoint-12000 is **reliable for short, isolated field values** (e.g. a cleanly segmented isolated name or numeral crop). It is **NOT reliable for full unsegmented handwritten lines or cursive archival paragraphs as-is**.
- **Next Step for Handwriting**: Retrain on real line-level handwriting data (ICDAR 2025 IHDR Task B page/line recognition dataset is the identified target).
- **Printed Text OCR**: EasyOCR remains the primary printed text engine (**3.06% CER** on real Bhoomi/Satbara documents).
- **Confidence Calibration Status**: Uncalibrated (`null` calibrated score) by- **Audit Logging**: Low-confidence inferences are systematically logged in the audit trail without fabricating confidence scores or suppressing errors.
- **Gemini Fallback**: If `GEMINI_API_KEY` is omitted or quota is exceeded, the pipeline gracefully defaults to `RuleSemanticEngine` without interrupting document processing.

---

## 14. Deployment Architecture

The application is structured for production deployment across separated frontend and backend services:

### Frontend (Next.js / Vercel)
- **Deployment Platform**: Vercel
- **Framework**: Next.js 14+ (App Router)
- **Environment Variables**:
  ```bash
  NEXT_PUBLIC_API_BASE_URL=https://api.your-domain.com
  ```
  *(In local development, defaults to `http://localhost:8000`)*
- **Security**: The frontend communicates strictly via HTTP API calls. No AI API keys or secrets are packaged in client-side code.

### Backend (FastAPI Persistent Service)
- **Deployment Platform**: Persistent Linux container (Render, Railway, Fly.io, AWS ECS, GCP Cloud Run)
- **Runtime**: Python 3.10 - 3.13
- **Entrypoint**: `uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`
- **Environment Variables**:
  ```bash
  GEMINI_API_KEY=your_gemini_api_key_here
  SEMANTIC_ENGINE=gemini
  SEMANTIC_MODEL_NAME=gemini-3.1-flash-lite
  SEMANTIC_TIMEOUT_SECONDS=30.0
  DATABASE_URL=sqlite:///./doc_platform.db
  STORAGE_TYPE=local
  STORAGE_LOCAL_DIR=./storage/uploads
  MAX_FILE_SIZE_MB=50
  ```

---

## 15. Single-Command Demo Execution

To verify the end-to-end production pipeline with live API upload, status polling, structured extraction, bilingual translation, and human review verification:

```bash
# Ingest synthetic Karnataka RTC PNG
python -X utf8 scripts/run_demo_e2e.py demo_artifacts/synthetic_karnataka_rtc.png

# Ingest synthetic Karnataka RTC PDF
python -X utf8 scripts/run_demo_e2e.py demo_artifacts/synthetic_karnataka_rtc.pdf
```
