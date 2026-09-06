# Integrated AI/ML Document Processing Pipeline for Land Records

A production-grade, state-configuration driven AI/ML document processing pipeline designed to transform raw land-record scans (Khatauni, Khasra, 7/12 Satbara, Jamabandi, Khatiyan) into validated structured records with field-level confidence and complete provenance tracking.

---

## 1. High-Level Architecture & 3-Person Team Ownership

```
                             [ RAW DOCUMENT ]
                                    │ (SHA-256 Hashing)
                                    ▼
[ PERSON A ] ────────► PREPROCESSING (OpenCV / Real-ESRGAN)
                                    │
                                    ▼
                       PRINTED OCR (PaddleOCR / PP-OCRv5)
                                    │
                                    ▼
                       LAYOUT & TABLE DETECTION (PP-StructureV3)
                                    │
                                    ▼
                       DOCUMENT CLASSIFICATION
                                    │
                                    ▼ (Shared Schema: DocumentOCRResult)
[ PERSON B ] ────────► HANDWRITING RECOGNITION (TrOCR / HuggingFace)
                                    │
                                    ▼ (Shared Schema: HandwritingResult)
══════════════════════════════════════════════════════════════════════════════════
[ PERSON C ]
══════════════════════════════════════════════════════════════════════════════════
                                    │
                                    ▼
                       STRUCTURED FIELD EXTRACTION (Regex + State Config Engine)
                                    │
                                    ▼
                       FIELD NORMALIZATION (Names, Dates, Land Units -> Hectares)
                                    │
                       OCR CONSENSUS & DISAGREEMENT RESOLUTION
                                    │
                                    ▼
                       RULE & CROSS-RECORD VALIDATION
                                    │
                                    ▼
                       GIS VALIDATION (PostGIS / Cadastral Mismatch Checks)
                                    │
                                    ▼
                       DUPLICATE DETECTION (SHA-256 Exact + Vector Cosine)
                                    │
                                    ▼
                       FIELD-LEVEL CONFIDENCE SCORING (Multi-Factor Weighted)
                                    │
                                    ▼
                       EVIDENCE PROVENANCE & AUDIT LOGGING
                                    │
                                    ▼
                       FINAL STRUCTURED OUTPUT (FinalDocumentResult)
```

### Team Responsibilities:
- **Person A (Preprocessing + Printed OCR + Layout + Classification)**: Owns `src/preprocessing/`, `src/ocr/`, `src/classification/`.
- **Person B (Handwriting OCR + Training Pipeline)**: Owns `src/handwriting/`, `training/`, `models/`, fine-tuning TrOCR, MLflow tracking, DVC dataset versioning, Active Learning loops.
- **Person C (Extraction + Validation + Confidence + GIS + Duplicates + DB)**: Owns `src/extraction/`, `src/validation/`, `src/confidence/`, `src/database/`, `src/integration/person_c_service.py`.
- **Shared Foundation**: `schemas.py`, `configs/states/*.json`, `configs/pipeline.json`, hashing, logger, tests, evaluation suite.

---

## 2. Directory Structure

```
project-root/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
├── schemas.py                          # Unified Pydantic contracts (A, B, C)
│
├── configs/
│   ├── pipeline.json                   # Pipeline thresholds & confidence weights
│   └── states/                         # State-specific configurations
│       ├── default.json
│       ├── up.json                     # Uttar Pradesh Khatauni & Khasra
│       ├── mp.json                     # Madhya Pradesh Bhoo-Abhilekh
│       ├── maharashtra.json            # Maharashtra 7/12 Satbara
│       └── bihar.json                  # Bihar Jamabandi & Khatiyan
│
├── src/
│   ├── utils/
│   │   ├── hashing.py                  # SHA-256 document & field hashing
│   │   ├── logger.py                   # Structured lifecycle logger
│   │   └── config_loader.py            # State & pipeline config loader
│   ├── extraction/
│   │   ├── extractor.py                # State-driven regex & table extractor
│   │   └── disagreement.py             # OCR consensus & conflict resolver
│   ├── validation/
│   │   ├── normalization.py            # Names, dates, land units normalization
│   │   ├── rules.py                    # Single-field constraint validator
│   │   └── cross_record.py             # Cross-record mathematical consistency
│   ├── database/
│   │   ├── gis.py                      # Cadastral PostGIS spatial validator
│   │   └── duplicates.py               # SHA-256 and vector duplicate detector
│   ├── confidence/
│   │   ├── scorer.py                   # Multi-factor field confidence calculator
│   │   └── evidence.py                 # Audit trail & provenance builder
│   └── integration/
│       ├── person_c_service.py         # extract_and_validate(...) interface
│       └── walking_skeleton.py         # End-to-end 3-person pipeline runner
│
├── evaluation/
│   └── evaluate_pipeline.py            # Comprehensive 10+ doc evaluation benchmark
│
└── tests/
    ├── unit/
    │   ├── test_hashing.py
    │   ├── test_extraction.py
    │   ├── test_normalization.py
    │   ├── test_rules.py
    │   ├── test_cross_record.py
    │   ├── test_gis.py
    │   ├── test_duplicates.py
    │   ├── test_confidence.py
    │   └── test_disagreement.py
    └── integration/
        ├── test_person_c_pipeline.py
        └── test_walking_skeleton.py
```

---

## 3. Quick Start & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest tests/ -v
```

### 3. Run Benchmark Evaluation (10+ Realistic Documents)
```bash
python evaluation/evaluate_pipeline.py
```

---

## 4. Person C API Usage

```python
from schemas import DocumentOCRResult, HandwritingResult
from src.integration.person_c_service import extract_and_validate

# Run extraction and validation on Person A & B outputs
final_doc = extract_and_validate(
    ocr_result=ocr_result,
    handwriting_result=handwriting_result,
    selected_state="UP",  # or None for auto-detection
)

# Access validated structured fields
for field_name, field_obj in final_doc.fields.items():
    print(f"Field: {field_name}")
    print(f"  Raw Value        : {field_obj.raw_value}")
    print(f"  Normalized Value : {field_obj.normalized_value} {field_obj.normalized_unit or ''}")
    print(f"  Confidence       : {field_obj.confidence:.3f}")
    print(f"  Evidence BBox    : {field_obj.bbox}")
    print(f"  Validation Status: {field_obj.validation_status.value}")

# Check GIS and Duplicate Results
print(f"GIS Verified: {final_doc.gis_validation.is_verified}")
print(f"Duplicate Found: {final_doc.duplicate_analysis.is_duplicate}")
print(f"Overall Confidence: {final_doc.overall_confidence:.3f}")
print(f"Requires Human Review: {final_doc.requires_human_review}")
```
