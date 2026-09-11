# Unified Land Record Digitization Platform: A + B + C Architecture

## 1. System Overview & Division of Responsibilities

This repository integrates three independently developed, specialized engineering subsystems into a production-grade Land Record Digitization system:

```
+-----------------------------------------------------------------------------------+
|                                  FRONTEND LAYER                                   |
|                Next.js 14 App Router, TypeScript, Tailwind CSS                    |
|   (Document Upload, Live Status Polling, Extracted Fields Viewer, Validation UI)  |
+-----------------------------------------------------------------------------------+
                                         |
                                         | REST HTTP / FormData
                                         v
+-----------------------------------------------------------------------------------+
|                        PERSON A: PLATFORM & INGESTION CORE                        |
| - Unified FastAPI Gateway & Routing (`backend/app/main.py`)                       |
| - Authentication & RBAC (Admin, Officer, Reviewer, Viewer)                        |
| - MinIO Object Storage (with zero-config local filesystem fallback)               |
| - Relational Persistence (PostgreSQL with zero-config SQLite fallback)            |
| - Asynchronous Job Queue (Celery/Redis with synchronous local worker fallback)    |
| - SHA-256 Checksum Ingestion Deduplication                                        |
| - Image Enhancement Preprocessing (Deskew, Denoise, Contrast Adjustment)         |
+-----------------------------------------------------------------------------------+
                                         |
                                         | NumPy / PIL Image + Region Specifications
                                         v
+-----------------------------------------------------------------------------------+
|                  PERSON B: MULTIMODAL OCR & HANDWRITING ROUTING                   |
| - DocumentProcessingPipeline (`src/integration/document_pipeline.py`)             |
| - LanguageScriptRouter (`src/handwriting/router.py`)                              |
| - Printed OCR Engines:                                                            |
|     * Printed Kannada: PaddleOCR (`ppocr_v4_kannada`)                             |
|     * Printed English: PaddleOCR (`ppocr_v4_en`)                                  |
| - Handwritten Recognition Engines:                                                |
|     * Handwritten Kannada: Fine-tuned TrOCR checkpoint                            |
|     * Handwritten English: TrOCR (`microsoft/trocr-small-handwritten`)            |
| - Coordinate-Preserving Spatial Layout Ingestion & Text Slicing                   |
| - Bounding-Box Math & Confidence Calibration                                      |
| - Output Contract: `DocumentProcessingResponse` & `RecognizedRegionResult`        |
+-----------------------------------------------------------------------------------+
                                         |
                                         | PersonCAdapter Bridge
                                         | (`src/integration/person_c_adapter.py`)
                                         v
+-----------------------------------------------------------------------------------+
|            PERSON C: EXTRACTION, NORMALIZATION, VALIDATION & GIS VERIFICATION     |
| - State Configuration Engine (`configs/states/*.json` - KA, UP, MP, MH, BR, etc.) |
| - Structured Field Extraction (`src/extraction/extractor.py`)                     |
| - Rule-based Normalization (`src/validation/normalization.py` - Hectares, Dates)  |
| - Deterministic Business Rules (`src/validation/rules.py`)                        |
| - Cross-Record Reconciliation (`src/validation/cross_record.py`)                 |
| - Cadastral GIS Spatial Validation (`src/database/gis.py`):                       |
|     * Real PostGIS queries when PostgreSQL/PostGIS is active                      |
|     * Shapely GeoJSON polygon fallback for offline local runs                     |
| - Duplicate Detection (`src/database/duplicates.py`):                             |
|     * Exact SHA-256 hash match                                                    |
|     * Vector cosine similarity (pgvector on DB; scikit-learn TF-IDF offline)      |
| - Multi-Factor Confidence Scoring (`src/confidence/scorer.py`)                    |
| - Output Contract: `FinalDocumentResult`                                          |
+-----------------------------------------------------------------------------------+
                                         |
                                         | Unified API Payload
                                         v
+-----------------------------------------------------------------------------------+
|                                  USER INTERFACE                                   |
|   Extracted Fields Table, Normalized Values, GIS Cadastral Flags, Confidence,     |
|              Human-Review Alerts, and Structured JSON Inspection                  |
+-----------------------------------------------------------------------------------+
```

---

## 2. End-to-End Data Flow (A -> B -> C)

1. **Ingestion (Person A)**:
   - A user uploads a land record (PNG, JPG, TIFF, or PDF) via the Next.js UI or `POST /api/ocr/process` / `POST /api/v1/documents/upload`.
   - The file is hashed with SHA-256 for instant duplicate detection and stored in MinIO (or local `backend/storage/uploads/`).
   - Image enhancement cleans the image (deskewing angle correction, contrast adjustment, and light median filtering).

2. **OCR & Recognition (Person B)**:
   - The preprocessed document is processed through `DocumentProcessingPipeline`.
   - `LanguageScriptRouter` routes each layout region or full page based on language (`kannada`, `english`) and modality (`printed`, `handwritten`).
   - Text is extracted with token probabilities, bounding boxes, and engine metadata, compiled into `DocumentProcessingResponse`.

3. **Adapter Boundary (`PersonCAdapter`)**:
   - `PersonCAdapter.to_person_c_input()` converts Person B's `DocumentProcessingResponse` into Person C's `DocumentOCRResult` and `HandwritingResult`.
   - All OCR evidence (region bounding boxes, raw text, normalized text, model identifiers, confidences) is preserved without loss.

4. **Structured Intelligence (Person C)**:
   - `extract_and_validate()` loads the state configuration (e.g. `configs/states/karnataka.json`).
   - Extracts revenue fields: Survey/Khasra Number, Khatauni, Owner Name, Father/Husband Name, Land Extent/Area, Village, Taluk, District.
   - Normalizes land units (acres, guntas, bighas, cents) into standard hectares.
   - Executes single-record validation (regex patterns, required fields, area boundaries).
   - Validates parcel against Cadastral GIS spatial layers (PostGIS or Shapely GeoJSON fallback).
   - Evaluates potential duplicate records via vector cosine similarity.
   - Calculates calibrated overall confidence score and determines human-review necessity (`requires_human_review`).

5. **Response & Presentation**:
   - The final combined output is serialized into `DocumentProcessingResponse` (and stored in database `ExtractionResult`).
   - The Next.js frontend displays the status badge, extracted fields table, confidence progress bars, cadastral GIS audit status, and full JSON payload.

---

## 3. Adapter Layer & Schema Specifications

The adapter layer resides in `src/integration/person_c_adapter.py`:

- **Input Conversion**:
  `PersonCAdapter.to_person_c_input(b_response, selected_state)`
  Maps `DocumentProcessingResponse.ordered_regions` to Person C `TextLineOCRResult` lines, with discrete bounding box coordinates and page numbers.

- **Unified Output**:
  `PersonCAdapter.to_unified_response(b_response, c_result)`
  Augments Person B's OCR response with:
  - `extracted_fields`: Key-value map of `ExtractedField` items with raw value, normalized value, bounding box, confidence, and validation status. Also provides `survey_number` alias for Karnataka/South Indian revenue terminology.
  - `validation`: Validation status (`valid`, `warning`, `invalid`), validation errors list, and cross-record consistency checks.
  - `gis_validation`: Cadastral verification status, spatial parcel match flag, area deviation percentage, and mismatch warnings.
  - `duplicate_analysis`: Duplicate status, similarity score, match type, and candidate duplicate IDs.
  - `overall_confidence`: Multi-factor confidence score between `0.0` and `1.0`.
  - `requires_human_review`: Boolean flag computed from OCR confidence and validation rule results.

---

## 4. API Endpoints

### 1. Unified OCR & Extraction Endpoint
- **URL**: `POST /api/ocr/process`
- **Consumes**: `multipart/form-data`
  - `file`: Image file (PNG, JPG, TIFF, BMP, WEBP)
  - `language`: `kannada` or `english` (default: `kannada`)
  - `is_handwritten`: Optional boolean (`true`, `false`, or omitted for auto-inference)
  - `selected_state`: State code (`KA`, `UP`, `MP`, `MH`, `BR`, `TN`, `DEFAULT`)
  - `regions`: Optional JSON array of sub-region bounding boxes
- **Response**: `DocumentProcessingResponse` with full OCR and Person C extraction payloads.

### 2. Platform Document Management
- **URL**: `POST /api/v1/documents/upload` - Upload file to MinIO + trigger asynchronous or sync pipeline.
- **URL**: `GET /api/v1/documents/` - Paginated list of ingested documents.
- **URL**: `GET /api/v1/documents/{id}` - Retrieve document metadata.
- **URL**: `GET /api/v1/documents/{id}/status` - Check ingestion/processing status.
- **URL**: `GET /api/v1/documents/{id}/results` - Retrieve structured fields and validation details.
- **URL**: `GET /api/v1/documents/{id}/fields` - Filter extracted fields by confidence threshold.
- **URL**: `GET /api/v1/documents/search` - Full-text search across documents and field values.

### 3. Health & Diagnostic Endpoints
- **URL**: `GET /health` - Platform overall health check.
- **URL**: `GET /api/ocr/health` - Person B multimodal engine availability.
- **URL**: `GET /api/v1/person-c/health` - Person C active services and supported states.

---

## 5. Offline & Fallback Strategies

The platform is designed for zero-dependency local operation:
- **MinIO Storage**: If MinIO is unreachable on `localhost:9000`, the system seamlessly falls back to storing files in `backend/storage/uploads/`.
- **PostgreSQL / PostGIS**: If PostgreSQL is not configured, the database initializes SQLite automatically (`backend/land_records.db`), and `GISValidator` validates parcels against local cadastral GeoJSON files (`data/gis/raw/*.geojson`) using Shapely.
- **Vector Duplication**: If `pgvector` is not available, `DuplicateDetector` uses `scikit-learn`'s `HashingVectorizer` and `TfidfVectorizer` for offline cosine similarity.
- **Celery / Redis**: If Redis is offline, Celery background tasks gracefully execute synchronously in-process (`local-sync-worker`).
- **TrOCR Models**: If the fine-tuned local Kannada checkpoint is not present, the pipeline falls back to printed OCR and clearly logs that handwritten Kannada requires the local checkpoint.

---

## 6. Model Weight Requirements

The repository does NOT commit large model weights to Git:
- **Local Checkpoint Path**:
  `models/trocr/kannada_full_checkpoints/best_checkpoint/`
- **HuggingFace Fallback**:
  `microsoft/trocr-small-handwritten` for handwritten English (cached automatically in HuggingFace cache).
- **PaddleOCR**:
  `ppocr_v4_kannada` and `ppocr_v4_en` (cached automatically in `~/.paddlex/official_models/`).

---

## 7. Commands to Run & Test

### Backend Startup
```bash
# Activate virtual environment
.venv\Scripts\activate

# Start Unified FastAPI backend server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Swagger UI is accessible at: `http://localhost:8000/docs`

### Frontend Startup
```bash
cd frontend
npm install
npm run dev
```
Next.js web application is accessible at: `http://localhost:3000`

### Test Commands
```bash
# 1. Run Complete A+B+C Integration Test Suite (10 Scenarios)
pytest tests/integration/test_full_pipeline.py -v

# 2. Run Person C Unit & Integration Tests
pytest tests/unit/test_extraction.py tests/unit/test_normalization.py tests/unit/test_rules.py tests/unit/test_gis.py tests/unit/test_duplicates.py tests/unit/test_confidence.py tests/unit/test_person_c_adapter.py -v

# 3. Run Backend Platform Tests (29 tests)
pytest backend/tests/ -v

# 4. Build Frontend Production Bundle
cd frontend && npm run build
```
