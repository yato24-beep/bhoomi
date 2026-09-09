# Land Record Digitization & Document Processing Platform

An end-to-end, production-grade multimodal document intelligence platform for archival land records, RTC/Pahani forms, cadastral records, and government invoices. Built with **FastAPI**, **Next.js 14**, **SQLAlchemy**, **PaddleOCR**, and **Fine-tuned TrOCR**.

---

## 🏛️ System Architecture

```text
                                  ┌─────────────────────────────┐
                                  │      Next.js Frontend       │
                                  │ (Dashboard / Upload / View) │
                                  └──────────────┬──────────────┘
                                                 │ HTTP / REST
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                FastAPI Main Application Backend                              │
│                                                                                             │
│  ┌───────────────────────┐   ┌────────────────────────┐   ┌──────────────────────────────┐  │
│  │   Auth & RBAC API     │   │   Documents API        │   │    Multimodal OCR API        │  │
│  │ (/api/v1/auth)        │   │ (/api/v1/documents)    │   │  (/api/ocr/process, health)  │  │
│  └───────────────────────┘   └───────────┬────────────┘   └──────────────┬───────────────┘  │
└──────────────────────────────────────────┼───────────────────────────────┼──────────────────┘
                                           │                               │
                                           ▼                               ▼
                      ┌───────────────────────────────────────────────────────────┐
                      │              Document Processing Pipeline                 │
                      │                                                           │
                      │  1. Non-destructive Image Preprocessing & Enhancement     │
                      │  2. Layout Region Detection & Spatial Reading-Order Sort  │
                      │  3. Multimodal Language & Script Routing                  │
                      │                                                           │
                      │     ├── Printed Kannada / English   ──► PaddleOCR         │
                      │     └── Handwritten Kannada / EN    ──► TrOCR Transformer │
                      │                                                           │
                      │  4. Conservative Kannada Normalization (Unicode NFC)      │
                      │  5. Confidence Calibration & Review Flagging              │
                      └────────────────────────────┬──────────────────────────────┘
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
         ┌───────────────────────────────┐                   ┌───────────────────────────────┐
         │ Relational Database (Postgres │                   │ Storage Service (MinIO / S3   │
         │ with auto SQLite fallback)    │                   │ with local disk fallback)     │
         └───────────────────────────────┘                   └───────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ (Node 20+ / 24+ recommended)
- Optional: CUDA-compatible GPU (automatic CPU fallback supported)

### 2. Backend Setup & Startup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Configure environment (default fallback works out of the box)
cp .env.example .env

# 3. Start the unified FastAPI backend
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend API Swagger Docs will be live at: **http://localhost:8000/docs**  
Health Check: **http://localhost:8000/health**  
OCR Health Check: **http://localhost:8000/api/ocr/health**

### 3. Frontend Setup & Startup

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Next.js dev server
npm run dev
```

Frontend Dashboard will be live at: **http://localhost:3000**

---

## 🧩 Core Components & Ownership

### 1. Platform Infrastructure & API (`backend/`, `frontend/`)
- **FastAPI Core**: Asynchronous API with Pydantic v2 validation, RBAC authentication (JWT), SHA-256 duplicate detection, and granular field queries.
- **Resilient Database Layer**: Automatic PostgreSQL connection check with zero-config local **SQLite fallback** (`doc_platform.db`).
- **Resilient Storage Layer**: MinIO object storage with automatic **local filesystem fallback** (`storage/uploads/`).
- **Next.js 14 Dashboard**: Drag-and-drop document upload, live processing status tracking, and structured field search.

### 2. Multimodal OCR & Handwriting Engine (`src/`)
- **Printed Text Engine**: High-accuracy regional OCR powered by **PaddleOCR** (Kannada `ppocr_v4_kannada` & English `ppocr_v4_en`).
- **Handwriting Engine**: Vision-Encoder-Decoder (**TrOCR**) fine-tuned on native Kannada land-record cursive script.
- **Multimodal Script Router (`LanguageScriptRouter`)**: Intelligently routes image regions to the optimal model based on language (`kannada`, `english`, `telugu`, `tamil`, `hindi`) and text modality (`printed` vs `handwritten`).
- **Document Pipeline (`process_document`)**: Orchestrates deskewing, enhancement, reading-order spatial sorting, text normalization, and length-weighted confidence scoring.
- **Active Learning & Correction (`CorrectionService`)**: Collects human reviewer corrections and exports audit manifests for active learning.

### 3. Handwriting Training & Evaluation Suite (`training/`, `src/training/`)
- Line/word dataset bootstrap and JSONL manifest preparation tools.
- Stratified dataset partitioning (train/val/test).
- Training loop with mixed-precision, checkpointing, and Character/Word Error Rate (CER/WER) evaluation.

---

## 📡 API Endpoints Reference

### Health & Status
- `GET /health` — General service health and storage/database connectivity.
- `GET /api/ocr/health` — Multimodal OCR pipeline status and supported models.

### Multimodal OCR Processing
- `POST /api/ocr/process` — Direct image OCR processing with language routing and structured JSON output.
  - **Parameters**: `file` (Multipart image), `language` (default: `kannada`), `is_handwritten` (optional bool), `apply_preprocessing` (bool), `apply_normalization` (bool).

### Document Management & Extraction
- `POST /api/v1/documents/upload` — Upload document (PDF/Image), hash verification, storage, and asynchronous/synchronous extraction pipeline execution.
- `GET /api/v1/documents/` — List all uploaded documents with pagination.
- `GET /api/v1/documents/{id}` — Retrieve document metadata and processing status.
- `GET /api/v1/documents/{id}/results` — Retrieve structured key-value extracted data.
- `GET /api/v1/documents/{id}/fields` — Retrieve granular extracted fields with normalized bounding boxes.
- `GET /api/v1/documents/search?q={query}` — Search across filenames and extracted text fields.
- `GET /api/v1/documents/{id}/download` — Stream raw file from storage.
- `DELETE /api/v1/documents/{id}` — Delete document record and associated storage.

### Authentication & Users
- `POST /api/v1/auth/login` — OAuth2 password flow returning JWT bearer token.
- `GET /api/v1/auth/me` — Current authenticated user profile and RBAC role.

---

## 🧪 Testing & Validation

Run unit and integration test suites:

```bash
# Run Person B OCR & pipeline unit tests
python -m pytest tests/unit/ -v

# Run Person A backend test suite
python -m pytest backend/tests/test_health.py backend/tests/test_hashing.py -v

# Run OCR modality verification script
python evaluation/scripts/verify_four_modalities.py
```

---

## 🛠️ Docker & Container Deployment

To run the complete platform stack (PostgreSQL, MinIO, Redis, Celery Worker, FastAPI Backend, Next.js Frontend) using Docker Compose:

```bash
docker-compose up --build
```
