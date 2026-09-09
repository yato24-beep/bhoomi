# Integration Status & Production Architecture Report

**Project:** Land Record Digitization & Document Processing Platform  
**Integration Status:** Unified & Validated  
**Version:** 1.0.0

---

## 1. Final Architecture

The repository integrates Person A (Backend application, PostgreSQL metadata, MinIO storage, Next.js frontend, layout segmentation) and Person B (Multimodal OCR, PaddleOCR for printed text, TrOCR for Kannada/English handwriting, LanguageScriptRouter, active learning correction service) into a unified codebase.

```text
                                  ┌─────────────────────────────┐
                                  │      Next.js Frontend       │
                                  │   (Dashboard, Upload, View) │
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

## 2. Backend Entry Point

- **File**: `backend/app/main.py`
- **ASGI Instance**: `backend.app.main:app` or `app.main:app` (when executing within `backend/`)
- **Startup Command**:
  ```bash
  uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
  ```
- **Integrated Routes**:
  - `GET /health`: Platform health check (Database, Storage, Environment)
  - `GET /api/ocr/health`: Multimodal OCR pipeline status & registered script routes
  - `POST /api/ocr/process`: Direct multimodal OCR processing endpoint
  - `POST /api/v1/documents/upload`: File upload, SHA-256 duplicate detection & extraction
  - `GET /api/v1/documents/`: Paginated document list
  - `GET /api/v1/documents/{id}/results`: Structured key-value extraction payload
  - `GET /api/v1/documents/{id}/fields`: Granular extracted fields with spatial bounding boxes
  - `GET /api/v1/documents/search`: Multi-field substring search across documents
  - `POST /api/v1/auth/login`: JWT login
  - `GET /docs`: Interactive OpenAPI / Swagger UI

---

## 3. Frontend Entry Point

- **Framework**: Next.js 14 (App Router) with TypeScript & Tailwind CSS
- **Root Directory**: `frontend/`
- **Main Pages**:
  - `frontend/src/app/page.tsx`: Main dashboard with live statistics, search & document table
  - `frontend/src/app/upload/page.tsx`: Drag-and-drop document uploader with type validation
  - `frontend/src/app/documents/[id]/page.tsx`: Structured results viewer, bounding box table, JSON inspector
- **API Client**: `frontend/src/lib/api.ts` & `frontend/src/lib/types.ts`
- **Startup Command**:
  ```bash
  cd frontend
  npm run dev
  ```

---

## 4. OCR Integration Flow

```
Document Upload (Image / PDF)
       │
       ▼
Image Preprocessing (src/preprocessing/image_enhancement.py)
 - Deskewing (Radon / Hough transform)
 - Grayscale / Contrast Enhancement (CLAHE)
 - Denoising (Bilateral / Gaussian filter)
       │
       ▼
Layout & Reading-Order Sorting (src/integration/person_a_adapter.py)
 - 24px vertical quantized line sorting (top-to-bottom, left-to-right)
       │
       ▼
Language & Script Router (src/handwriting/router.py)
 ├── Printed Kannada (kannada, is_handwritten=False)    ──► PaddleOCR (kannada)
 ├── Printed English (english, is_handwritten=False)    ──► PaddleOCR (en)
 ├── Handwritten Kannada (kannada, is_handwritten=True) ──► Fine-tuned TrOCR
 └── Handwritten English (english, is_handwritten=True) ──► TrOCR Small Handwritten
       │
       ▼
Postprocessing & Normalization (src/postprocessing/kannada_normalizer.py)
 - Unicode NFC normalization
 - Whitespace and zero-width joiner cleanup
 - Revenue domain lexicon suggestion matching
       │
       ▼
Confidence Calibration & Review Flagging (src/handwriting/confidence.py)
 - Token posterior log-probability scoring
 - Threshold evaluation (< 0.60 flagged for human review)
       │
       ▼
Structured Document Output (src/integration/schemas.py)
 - Merged text, granular field coordinates, review flags, processing duration
```

---

## 5. Database Configuration & Fallback

- **Production Target**: PostgreSQL 16 (`postgresql+psycopg://postgres:postgres_password@localhost:5432/doc_platform`)
- **Local Zero-Config Fallback**: Automatic detection in `backend/app/db/session.py`. If PostgreSQL is not running on `localhost:5432`, the backend automatically falls back to local SQLite (`doc_platform.db`).
- **Data Persistence**: Both PostgreSQL and SQLite store full relational schemas (`users`, `documents`, `extraction_results`, `extracted_fields`).

---

## 6. Required Environment Variables

All variables are specified with working defaults in `.env.example`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PROJECT_NAME` | `"Land Record Digitization Platform"` | Application title |
| `API_V1_STR` | `"/api/v1"` | API version prefix |
| `BACKEND_CORS_ORIGINS` | `"http://localhost:3000,http://127.0.0.1:3000"` | Allowed CORS frontend origins |
| `SECRET_KEY` | `"super_secret_jwt_key_..."` | JWT signing secret key |
| `POSTGRES_SERVER` | `"localhost"` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `"doc_platform"` | Database name |
| `MINIO_ENDPOINT` | `"localhost:9000"` | MinIO S3 endpoint |
| `MINIO_BUCKET_NAME` | `"documents"` | Document storage bucket |
| `KANNADA_HANDWRITING_MODEL_PATH`| `"models/trocr/kannada_full_checkpoints/best_checkpoint"` | Fine-tuned Kannada TrOCR weights |
| `DEFAULT_DEVICE` | `"auto"` | Hardware device (`auto`, `cuda`, `cpu`) |
| `DEFAULT_OCR_LANGUAGE` | `"kannada"` | Default OCR target language |
| `NEXT_PUBLIC_API_URL` | `"http://localhost:8000"` | Frontend backend API URL |

---

## 7. Model Weights Configuration & Lazy Loading

- **Location**: `models/trocr/`
- **Kannada Handwriting Checkpoint**: `models/trocr/kannada_full_checkpoints/best_checkpoint/`
- **Behavior**:
  - Model weights are loaded **lazily upon the first OCR inference request**.
  - Server startup is **instantaneous** and never blocked by downloading multi-gigabyte models.
  - Automatic **FP16 CUDA acceleration** on NVIDIA GPUs; transparent fallback to **FP32 CPU** execution.

---

## 8. Feature Capability Matrix

| Feature | Works Without Model Weights? | Notes |
| :--- | :---: | :--- |
| **Backend API & Swagger Docs** | ✅ Yes | Full OpenAPI schema loads immediately at `/docs` |
| **User Authentication & RBAC** | ✅ Yes | Admin, Officer, Reviewer, Viewer accounts seeded |
| **Document Upload & Hashing** | ✅ Yes | SHA-256 deduplication and storage functional |
| **Database Queries & Search** | ✅ Yes | Full-text and field search on SQLite / Postgres |
| **Next.js Frontend Dashboard** | ✅ Yes | Production build and live dev server verified |
| **Printed Kannada & English OCR** | ✅ Yes | PaddleOCR automatically initializes lightweight weights |
| **English Handwriting OCR** | ⚠️ Needs HuggingFace cache | Loads `microsoft/trocr-small-handwritten` on first call |
| **Kannada Fine-Tuned Handwriting** | ⚠️ Checkpoint dependent | Uses local checkpoint if present; otherwise returns clear audit warning |
| **Correction & Active Learning** | ✅ Yes | Manifest export and correction logging functional |

---

## 9. Full System Startup Commands

### Terminal 1: Backend
```bash
# From repository root
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2: Frontend
```bash
cd frontend
npm install
npm run dev
```

### Terminal 3 (Optional): Full Stack via Docker Compose
```bash
docker-compose up --build
```

---

## 10. Known Limitations

1. **Indic Handwriting Checkpoint Availability**: Non-Kannada Indic handwriting scripts (Telugu, Tamil, Malayalam, Hindi) are architected with fallback review flags (`requires_human_review=True`) until dedicated fine-tuned checkpoints are placed in `models/trocr/`.
2. **Severely Degraded Archives**: Archival parchment with severe ink bleed-through or tears benefit from manual verification via the review queue.
3. **Local Development Storage**: When MinIO or PostgreSQL are offline, data is safely saved in local SQLite (`doc_platform.db`) and local filesystem (`storage/uploads/`).
