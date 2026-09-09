# Person A Implementation & Team Handoff Document

**System**: AI-Powered Land Record Digitization Pipeline  
**Role**: PERSON A (Vision, Preprocessing, Multilingual OCR, Layout, Tables, Document Classification & Integration)  
**Status**: READY & FULLY TESTED  
**Repository Branch**: `person-a`  
**Package Path**: `person-a/`

---

## 1. Technical Components & Status

| Component | Status | Implementation Details |
| :--- | :--- | :--- |
| **Document Ingestion** | **REAL** | Supports PDF (PyMuPDF/fitz), PNG, JPG, TIFF, raw bytes, streaming `BinaryIO`, and NumPy arrays (`person-a/src/preprocessing/loader.py`). |
| **SHA-256 Hashing** | **REAL** | NIST FIPS 180-4 SHA-256 document hashing for identity and duplicate checking (`person-a/src/utils/hashing.py`). |
| **Image Quality Analysis** | **REAL** | Real Laplacian blur variance, RMS contrast, quadrant illumination uniformity, estimated DPI, and blank page detection (`person-a/src/preprocessing/quality.py`). |
| **Adaptive Preprocessing** | **REAL** | Real OpenCV adaptive pipeline: CLAHE, bilateral/Wiener denoising, unsharp mask sharpening, Radon/Hough deskew (`person-a/src/preprocessing/`). |
| **Real-ESRGAN** | **REAL** | Real PyTorch `RealESRGANMini` (RRDBNet) conditionally triggered on low-resolution/degraded crops (`person-a/src/preprocessing/super_resolution.py`). |
| **Printed OCR (PP-OCRv5)** | **REAL** | Real PaddleOCR / PP-OCRv5 neural engine (`PP-LCNet_x1_0_textline_ori`, `PP-OCRv6_medium_det`, `PP-OCRv6_medium_rec`). |
| **Multilingual OCR** | **REAL** | Config-driven multilingual support (Hindi, Marathi, Tamil, Kannada, English, etc.). |
| **Layout & Table Detection** | **PARTIAL / FALLBACK** | **Honest Classification**: Uses real OpenCV morphological line/grid extraction (cells, row/col matrix, Markdown, JSON table dataframes) and spatial heuristics (`person-a/src/tables/detector.py`). Does *not* download or invoke a heavy neural PP-Structure layout checkpoint. |
| **Document Classification** | **REAL / DETERMINISTIC** | State-config JSON driven rule/keyword/layout signal classifier with evidence tracking (`person-a/src/classification/classifier.py`). |
| **Region-Routed OCR** | **REAL** | Dispatches printed text to PaddleOCR and handwritten crops to Person B adapter (`person-a/src/pipeline.py`). |
| **OCR Consensus Engine** | **REAL** | Multi-candidate edit-distance comparison with provenance tracking and confidence conflict penalization (`person-a/src/ocr/consensus.py`). |
| **Person B Adapter** | **ADAPTER** | Clean pluggable adapter + `StubHandwritingAdapter` (`person-a/src/handwriting/adapter.py`). |
| **Person C Integration** | **ADAPTER / CONTRACT** | Outputs canonical `DocumentOCRResult` matching `person-c/schemas.py` (`person-a/src/integration/person_c_adapter.py`). |

---

## 2. Directory Structure

```
person-a/
├── configs/
│   ├── pipeline.json
│   └── states/
│       ├── default.json
│       ├── maharashtra.json
│       ├── karnataka.json
│       ├── up.json
│       ├── tamilnadu.json
│       ├── bihar.json
│       └── mp.json
├── data/
│   └── samples/
│       ├── sample_land_record_7_12.png
│       ├── sample_skewed_land_record.png
│       ├── sample_low_res_degraded.png
│       └── sample_karnataka_bhoomi_rtc.png
├── docs/
│   ├── PERSON_A_HANDOFF.md
│   └── INTEGRATION_CONTRACT.md
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── schemas.py
│   ├── preprocessing/
│   ├── ocr/
│   ├── layout/
│   ├── tables/
│   ├── classification/
│   ├── config/
│   ├── handwriting/
│   ├── integration/
│   └── utils/
├── tests/
│   ├── test_hashing.py
│   ├── test_preprocessing.py
│   ├── test_consensus.py
│   ├── test_classification.py
│   └── test_person_a_pipeline.py
├── demo.py
├── requirements.txt
└── README.md
```

---

## 3. Integration Guidelines

### For Person B (Handwriting OCR):
Register your model with:
```python
from person_a.src.handwriting.adapter import HandwritingAdapter, register_handwriting_adapter

class LiveTrOCRAdapter(HandwritingAdapter):
    def recognize_region(self, crop, bbox, page_number=1, region_id=None):
        text, conf = my_trocr_model.predict(crop)
        return HandwritingRegionResult(
            region_id=region_id or "hw_001",
            text=text,
            confidence=conf,
            page_number=page_number,
            bbox=bbox,
            model_version="trocr-fine-tuned-v1",
        )

register_handwriting_adapter(LiveTrOCRAdapter())
```

### For Person C (Structured Extraction & Validation):
Person A provides the exact `DocumentOCRResult` required:
```python
from person_a.src.pipeline import process_document
from person_a.src.integration.person_c_adapter import convert_person_a_to_document_ocr_result
from schemas import DocumentOCRResult
from src.integration.person_c_service import extract_and_validate

# 1. Run Person A
ocr_output = process_document("path/to/land_record.png", state_hint="MH")

# 2. Convert to canonical Person C schema
payload = convert_person_a_to_document_ocr_result(ocr_output)
doc_ocr_result = DocumentOCRResult.model_validate(payload)

# 3. Person C Extraction & Validation
final_result = extract_and_validate(doc_ocr_result)
```

---

## 4. Confidence & Multilingual Guarantees

1. **OCR Confidence Score Rationale**:
   `overall_confidence` is an uncalibrated mean aggregation score computed across detected character/word token confidences from the PaddleOCR PP-OCRv5 engine. It is **NOT** a calibrated posterior probability of correctness. Downstream validation in Person C performs schema constraints and mathematical validation for field-level assurance.

2. **Multilingual Fallback Transparency**:
   When a state jurisdiction requests an Indic language (e.g., `mr` for Marathi or `kn` for Kannada) whose specific offline checkpoint is not downloaded in the local environment, Person A transparently routes through the multilingual `PP-OCRv5` Latin/Multilingual model and explicitly records:
   - `requested_language`: e.g. `"mr"`
   - `actual_language`: `"en"`
   - `fallback_occurred`: `True`
   - `fallback_reason`: Detailed reason explanation in `LanguageDetectionResult`.

3. **Table & Grid Detection**:
   Table structures are detected using geometric morphological kernel operations (Otsu binarization, horizontal/vertical morphological structuring elements, contour cell extraction, and row/column spatial interval clustering). Output is structured into `rows_count`, `cols_count`, individual `cells` with bounding boxes, formatted `markdown` tables, and `dataframe_json`.

