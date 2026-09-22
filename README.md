# Bhoomi: Land Record Digitization & Multimodal AI Platform

> **Status Notice:** This repository contains a functional prototype. Karnataka is the current implementation and validation environment, while the target platform is designed for expansion across India.

---

## 1. Project Overview

**Bhoomi** is an intelligent, privacy-preserving, and cloud-hybrid land record digitization platform designed to process complex, multi-lingual, and historical Indian cadastral records (RTC/Pahani, Mutation extracts, Title deeds, and Revenue survey records).

The platform bridges edge client processing with cloud intelligence:
- **Client-Side Edge OCR:** High-efficiency, zero-cost-per-query client inference via ONNX Runtime Web (WebGPU Vision Encoder + WASM Autoregressive Text Decoder).
- **Backend Services:** Robust cadastral field extraction, script translation (Kannada ↔ English), spatial verification, granular auditing, and role-based access control.

---

## 2. National Objective & Vision

India's agricultural and rural real-estate ecosystem comprises over 150 million operational land holdings across 28 states and 8 union territories. Each state maintains localized terminology, colonial-era archival formats, and distinctive regional languages (e.g., Karnataka's *Bhoomi RTC*, Maharashtra's *7/12 Satbara*, Tamil Nadu's *Patta/Chitta*, Uttar Pradesh's *Khatauni*).

**Bhoomi's National Vision:**
- **Standardized Pan-India Data Contract:** Uniform representation for land parcel identifiers, ownership tenures, liabilities/encumbrances, soil classification, and mutation history across all states.
- **Hierarchical Governance Hierarchy:** `India -> State -> District -> Taluk/Tehsil -> Hobli/Circle -> Village -> Survey Number / Hissa`.
- **Zero-Trust Human-in-the-Loop:** Automated confidence scoring and validation rules flagging discrepancies for authorized land revenue officers.

---

## 3. Current Karnataka Scope

The current validation baseline is centered on Karnataka's Department of Revenue (Bhoomi RTC / Form No. 16 / Mutation Extracts):
- **Primary Script:** Kannada (ಕನ್ನಡ) script with mixed English revenue terminology.
- **Form Formats:** Standard RTC (Record of Rights, Tenancy and Crops) featuring owner schedules, survey & hissa breakdowns, water-rate tenures, and soil classifications.
- **Validation Rules:** Rigorous Karnataka survey number formats (`\d+(?:/[0-9A-Za-z]+)?`), Khata serial sequences, and Taluk/Village gazetteer lookups.

---

## 4. System Architecture

The platform is divided into a decoupled, layered microservices topology:

```text
               ┌──────────────────────────────────────────────┐
               │           Client Browser (Next.js 14)        │
               │  - WebGPU/WASM BrowserTrOCR Pipeline         │
               │  - Client-side Image Hashing (SHA-256)       │
               │  - Review Workbench & Spatial Grounding      │
               └──────────────────────┬───────────────────────┘
                                      │ Ingest (JSON + Base64 Image)
                                      ▼
               ┌──────────────────────────────────────────────┐
               │              FastAPI Gateway                 │
               │  - JWT RBAC (Admin, Officer, Reviewer, Viewer│
               │  - CORS & Rate Limiting                      │
               └───────┬──────────────┬──────────────┬────────┘
                       │              │              │
         ┌─────────────▼────┐  ┌──────▼──────┐  ┌────▼─────────────┐
         │ Document Service │  │ Translation │  │ Extraction       │
         │ - Metadata CRUD  │  │   Service   │  │   Service        │
         │ - RBAC Deletion  │  │ - Kannada ↔ │  │ - Multi-state    │
         │ - Verification   │  │   English   │  │   Cadastral Maps │
         └─────────────┬────┘  └─────────────┘  └────┬─────────────┘
                       │                             │
         ┌─────────────▼─────────────────────────────▼─────────────┐
         │ Storage & Persistence Layer                             │
         │ - PostgreSQL 15+ (Relational Documents & Fields)        │
         │ - MinIO / S3 Object Storage (Preserved Raw Images)      │
         └─────────────────────────────────────────────────────────┘
```

### Request Flow
1. **Document Upload:** User drops RTC/Pahani scan on the frontend.
2. **Document Storage:** File is hashed (SHA-256 / pHash) and persisted securely to MinIO / S3.
3. **OCR / Vision Processing:** Browser runs local `BrowserTrOCR` (Vision Encoder on WebGPU, Token Decoder on WASM).
4. **Backend Ingestion:** Extracted Kannada text and raw image are ingested via `POST /api/v1/documents/browser-result`.
5. **Cadastral Extraction:** `ExtractionService` parses Kannada text into canonical fields (`owner_name`, `survey_number`, `khata_number`, `village`, `taluk`, `district`, `extent`).
6. **Translation:** Dedicated translation service creates faithful English equivalents without altering raw Kannada text.
7. **Validation & Review:** Deterministic validation rules assess confidence; low-confidence fields are queued for officer review.

---

## 5. Technology Stack

- **Frontend:**
  - Framework: Next.js 14 (App Router, TypeScript)
  - Styling: Vanilla Tailwind CSS (Modern Slate & Emerald government aesthetic)
  - Vision Inference: ONNX Runtime Web (`onnxruntime-web` v1.21.0)
  - Icons & UI: Lucide React
- **Backend:**
  - API Framework: FastAPI (Python 3.11 / 3.13)
  - Relational Database: PostgreSQL with SQLAlchemy 2.0 ORM (SQLite local fallback)
  - Object Storage: MinIO Python SDK / S3-compatible storage (Local filesystem fallback)
  - Background Execution: Celery with Redis broker (eager mode supported in dev)
  - Language Transliteration & Processing: Aksharamukha, Indic-NLP, Pillow
- **Testing:**
  - Pytest with Starlette TestClient (30 automated backend tests)
  - Automated Next.js production builds

---

## 6. OCR Architecture: Hybrid BrowserTrOCR

Bhoomi deploys an asynchronous, client-side hybrid OCR architecture that maximizes throughput and device compatibility while eliminating cloud GPU costs:

- **Vision Encoder:**
  - **Primary Backend:** WebGPU (via ONNX Runtime Web `webgpu` execution provider)
  - **Fallback Backend:** WebAssembly (`wasm`) with multi-threading SIMD
- **Autoregressive Decoder:**
  - **Dedicated Backend:** WebAssembly (`wasm`)
  - **Rationale:** Autoregressive token-by-token generation with dynamic shapes has proven strictly more reliable and deterministic on WASM than WebGPU across diverse browser shader implementations.
- **Model Isolation:**
  - Low-level tensor management and ONNX sessions are encapsulated in `frontend/src/lib/browserTrOCR.ts`.
  - Application UI interacts solely through standard promise-based interfaces: `processImage(file) -> OCRResult`.

---

## 7. Model Hosting & Assets

Large ONNX model weights and vocabulary assets are excluded from Git to keep the repository light and portable:

- **Hugging Face Hub Repository:** `yatookami/iitb-kannada-trocr-v002-browser-fp32`
- **Frontend Model Manifest:** `frontend/public/model_manifest.json`
- **Hosted Artifacts:**
  - `encoder_model.onnx` (~220 MB)
  - `decoder_model_merged.onnx` (~310 MB)
  - `vocab.json` & `tokenizer_config.json`
  - WASM binaries dynamically served via `frontend/public/ort/`

---

## 8. Canonical Data Contract

All document processing stages adhere to typed, immutable data contracts (`backend/app/schemas/canonical.py`):

```python
CanonicalDocumentPayload
├── Document Identifier & Metadata (id, filename, file_hash, created_at)
├── Storage Information (storage_path, content_type, file_size)
├── OCR Result (raw_text, confidence, execution_provider, latency_ms)
├── Translation (kannada_text, english_text, service_version)
├── Extracted Fields (owner_name, survey_number, khata_number, extent, etc.)
├── Validation Result (is_valid, checks_passed, checks_failed)
└── Review Status (status, reviewed_by, reviewed_at, reviewer_notes)
```

---

## 9. Database Architecture

The relational schema uses PostgreSQL with explicit foreign keys and cascade rules:
- **`users`:** RBAC accounts (`ADMIN`, `OFFICER`, `REVIEWER`, `VIEWER`) with hashed passwords.
- **`documents`:** Core document records with file hash, MIME type, storage path, and processing status.
- **`extracted_fields`:** Granular key-value fields with spatial bounding boxes (`x_min, y_min, x_max, y_max`), confidence score, and validation status.
- **`review_items`:** Immutable audit records of human review decisions, corrections, and reviewer notes.

---

## 10. Local Setup & Installation

### Prerequisites
- Python 3.11 or 3.13
- Node.js 18+ and npm
- (Optional) Docker & Docker Compose for PostgreSQL & MinIO

### 1. Backend Setup
```bash
# Clone repository
git clone https://github.com/yato24-beep/bhoomi.git
cd land-record-digitization/backend

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp ../.env.example .env

# Run FastAPI server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Frontend Setup
```bash
cd ../frontend

# Install dependencies
npm install

# Copy ONNX WASM binaries to public directory
node scripts/copy-wasm.mjs

# Start Next.js development server
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 11. Environment Variables

Reference `.env.example` for all configurable keys:

| Variable | Description | Default |
|---|---|---|
| `PROJECT_NAME` | Service display name | `Land Record Digitization Platform` |
| `ENVIRONMENT` | Environment (`development`, `production`) | `development` |
| `SECRET_KEY` | JWT signing secret | Secure random string |
| `POSTGRES_SERVER` | PostgreSQL host | `localhost` |
| `DATABASE_URL` | Complete DB connection string | `sqlite:///./doc_platform.db` (fallback) |
| `MINIO_ENDPOINT` | MinIO storage endpoint | `localhost:9000` (falls back to local FS) |
| `MINIO_BUCKET_NAME`| Target bucket name | `documents` |
| `NEXT_PUBLIC_API_URL`| Frontend backend URL | `http://localhost:8000` |

---

## 12. Deployment

- **Frontend (Vercel):**
  - Optimized for static and edge deployment on Vercel.
  - Build Command: `node scripts/copy-wasm.mjs && next build`
  - Output Directory: `.next`
  - Environment Variable: `NEXT_PUBLIC_API_URL=https://<your-render-backend>.onrender.com`
- **Backend (Render / Railway / AWS):**
  - Dockerfile and `render.yaml` are pre-configured.
  - Health Check Path: `/health`
  - Build Command: `pip install -r backend/requirements.txt`
  - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

---

## 13. Testing & Verification

Run the full automated test suite:
```bash
# Run all 30 backend tests
python -m pytest backend/tests/ -v

# Run frontend production build & type check
cd frontend && npm run build
```

---

## 14. Known Limitations

1. **Hardware Acceleration Variability:** WebGPU execution is dependent on user GPU driver capabilities; browsers without WebGPU automatically fallback to WASM with negligible throughput reduction.
2. **Complex Historical Scripts:** 19th-century Modi or archaic Grantha Kannada handwritten deeds require human review verification.
3. **Current Scope:** Cadastral validation rules are currently tailored to Karnataka land revenue nomenclature.

---

## 15. Production Roadmap

- [ ] **Phase 1 (Complete):** Hybrid WebGPU/WASM client OCR, backend ingestion, and Karnataka cadastral extraction.
- [ ] **Phase 2 (Current):** Modularization into layered services, canonical data contracts, and RBAC audit trail.
- [ ] **Phase 3:** Pan-India expansion with regional extractors for Maharashtra (7/12) and Tamil Nadu (Patta).
- [ ] **Phase 4:** PostGIS integration for vector cadastral boundary overlay and satellite land verification.
- [ ] **Phase 5:** Decentralized verifiable land certificates with cryptographic signatures.

---

## License & Attribution

Designed and maintained for government and public digitization initiatives.
Open source under the [MIT License](LICENSE).
