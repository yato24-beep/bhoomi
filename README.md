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

### 3. Structured Extraction & Land Intelligence (`src/extraction/`, `src/validation/`, `src/database/`)
- **Structured Field Extraction**: State-specific extraction rules (`configs/states/` - KA, UP, MP, MH, BR, etc.) for Survey/Khasra numbers, Khatauni, Owner/Father names, Land Area, Village, Tehsil, and District.
- **Normalization & Business Rules**: Standardizes regional area units (acres, guntas, bighas, cents) to hectares, converts Vikram Samvat / Fasli years, and validates ownership shares and survey formats.
- **Cadastral GIS Verification**: Validates extracted land parcels against spatial GIS boundaries (PostGIS enabled; Shapely GeoJSON fallback when offline).
- **Duplicate Detection**: Fast SHA-256 exact match combined with semantic vector cosine similarity.
- **Multi-Factor Confidence Scoring**: Aggregates OCR token probabilities, dictionary validity, and business rule conformance to determine whether human review is required.

---

## 📡 API Endpoints Reference

### Health & Status
- `GET /health` — General platform service health and storage/database connectivity.
- `GET /api/ocr/health` — Multimodal OCR pipeline status and supported models.
- `GET /api/v1/person-c/health` — Person C active extraction, GIS, and state configs.

### Multimodal OCR & Land Record Processing
- `POST /api/ocr/process` — Unified image processing endpoint returning both full multimodal OCR and Person C structured land-record extractions, normalization, GIS validation, and confidence.
  - **Parameters**: `file` (Multipart image), `language` (default: `kannada`), `is_handwritten` (optional bool), `selected_state` (e.g. `KA`, `UP`), `apply_preprocessing` (bool), `apply_normalization` (bool).

### Document Management & Extraction
- `POST /api/v1/documents/upload` — Upload document (PDF/Image), hash verification, storage, and asynchronous/synchronous extraction pipeline execution.
- `GET /api/v1/documents/` — List all uploaded documents with pagination.
- `GET /api/v1/documents/{id}` — Retrieve document metadata and processing status.
- `GET /api/v1/documents/{id}/results` — Retrieve structured key-value extracted data and GIS validation.
- `GET /api/v1/documents/{id}/fields` — Retrieve granular extracted fields with normalized bounding boxes.
- `GET /api/v1/documents/search?q={query}` — Search across filenames and extracted text fields.
- `GET /api/v1/documents/{id}/download` — Stream raw file from storage.
- `DELETE /api/v1/documents/{id}` — Delete document record and associated storage.

### Authentication & Users
- `POST /api/v1/auth/login` — OAuth2 password flow returning JWT bearer token.
- `GET /api/v1/auth/me` — Current authenticated user profile and RBAC role.

---

## 🧪 Testing & Validation

Run unit, integration, and full pipeline test suites:

```bash
# Run Complete 10-scenario A+B+C Integration Test Suite
pytest tests/integration/test_full_pipeline.py -v

# Run Person C Extraction, Normalization, Rules, GIS, and Duplicates Tests
pytest tests/unit/test_extraction.py tests/unit/test_normalization.py tests/unit/test_rules.py tests/unit/test_gis.py tests/unit/test_duplicates.py tests/unit/test_confidence.py tests/unit/test_person_c_adapter.py -v

# Run Person A backend test suite (29 tests)
pytest backend/tests/ -v

# Run Frontend Production Build
cd frontend && npm run build
```

---

## 🛠️ Docker & Container Deployment

To run the complete platform stack (PostgreSQL, MinIO, Redis, Celery Worker, FastAPI Backend, Next.js Frontend) using Docker Compose:

```bash
docker-compose up --build
```
