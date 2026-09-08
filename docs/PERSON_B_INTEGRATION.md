# Person B Integration & Handover Guide: OCR & Handwriting Recognition Module

## Overview & Ownership

Person B owns the **multimodal OCR and Handwriting Recognition layer** for the Land Record Digitization system.

### What Person B Owns:
1. **Fine-Tuned Kannada Handwriting Model (TrOCR)**: Custom VisionEncoderDecoder trained on native Kannada historical/land-record handwriting.
2. **Printed Regional OCR Engine (PaddleOCR)**: High-accuracy printed Kannada and English document reader.
3. **Multimodal Language & Script Router (`LanguageScriptRouter`)**: Intelligently routes image crops based on language and text type (Handwritten vs. Printed).
4. **Document Processing Orchestrator (`process_document`)**: Performs non-destructive image enhancement, reading-order spatial sorting, length-weighted confidence scoring, and human-review flagging.
5. **Person A Layout Adapter (`PersonAAdapter`)**: Transparently normalizes layout bounding boxes and metadata into typed pipeline requests.
6. **Conservative Kannada Normalizer (`KannadaNormalizer`)**: Unicode NFC normalization, whitespace cleanup, and revenue lexicon suggestions.
7. **Human Correction & Active Learning Contract (`CorrectionService`)**: Collects verified corrections and exports training manifests without auto-retraining.

### What Person B Does NOT Own:
- **Person A**: Document layout analysis, table boundary detection, and YOLO segmentation.
- **Person C**: Information extraction, NER, revenue schema validation, and SQL/NoSQL database storage.

---

## Installation & Environment Requirements

- **Python**: 3.10+
- **PyTorch**: 2.0+ with CUDA support (CPU fallback is automatic).
- **Transformers**: Hugging Face `transformers` library with `sentencepiece` and `protobuf`.
- **PaddleOCR & PaddlePaddle**: Regional printed OCR engine.
- **Pillow & NumPy**: Image manipulation and tensor operations.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers sentencepiece protobuf paddlepaddle paddleocr pydantic Pillow
```

---

## Production Model Path Configuration & Hardware Behavior

- **Default Checkpoint Path**:
  ```text
  models/trocr/kannada_full_checkpoints/best_checkpoint/
  ```
- **Environment Variable Override**:
  You can override the model location without changing code:
  ```bash
  export KANNADA_HANDWRITING_MODEL_PATH="/path/to/custom/checkpoint"
  ```
- **Weights Caching**: Model weights (627.3 MB) are loaded **once lazily** into process memory (`_GLOBAL_MODEL_CACHE`) and shared across all document requests.
- **Hardware Execution**:
  - **CUDA (GPU)**: Automatically detected. Uses `fp16` mixed precision via `torch.amp.autocast('cuda')`. Average latency: ~30 ms per word crop.
  - **CPU Fallback**: If CUDA is unavailable or out of memory, gracefully falls back to FP32 CPU execution without errors.

---

## Public API & Main Entry Point

```python
from src.integration import process_document, DocumentProcessingRequest, RegionRequest
```

### Signature:
```python
def process_document(
    image: Union[str, Path, Image.Image, np.ndarray, bytes, None] = None,
    request: Optional[DocumentProcessingRequest] = None,
    regions: Optional[Sequence[Any]] = None,
    is_handwritten: Optional[bool] = None,
    language: Optional[str] = "kannada",
    script: Optional[str] = "Kannada",
    page_number: int = 1,
    document_id: Optional[str] = None,
    image_path: Optional[str] = None,
    **kwargs: Any,
) -> DocumentProcessingResponse:
```

---

## Data Contracts & Schemas

### 1. Input Schemas (Person A &rarr; Person B)

#### `RegionRequest`
Represents an individual bounding box supplied by Person A layout detection:
```python
from schemas import BoundingBox
from src.integration.schemas import RegionRequest, RegionType

region = RegionRequest(
    region_id="reg_001",
    bbox=BoundingBox(x_min=50, y_min=100, x_max=450, y_max=160),
    language="kannada",               # Default: "kannada"
    script="Kannada",
    is_handwritten=True,              # True=Handwritten, False=Printed, None=Auto
    region_type=RegionType.TEXT,      # TEXT, TABLE_CELL, HEADER, SIGNATURE, STAMP
    layout_confidence=0.96,           # Upstream layout model score
    metadata={"field_name": "owner"} # Arbitrary metadata
)
```

#### `DocumentProcessingRequest`
```python
from src.integration.schemas import DocumentProcessingRequest

doc_request = DocumentProcessingRequest(
    image="path/to/land_record_page1.jpg",
    document_id="DOC_KA_MYS_2026_001",
    page_number=1,
    regions=[region1, region2],
    language="kannada",
    apply_preprocessing=True,         # Applies deskew/contrast/denoise
    apply_normalization=True,         # Applies Unicode NFC and lexicon lookup
)
```

---

### 2. Output Schemas (Person B &rarr; Person C)

#### `RecognizedRegionResult`
Emitted for every recognized sub-region:
```python
class RecognizedRegionResult(BaseModel):
    region_id: str                   # Upstream or assigned ID
    raw_text: str                    # Raw verbatim model output (never modified)
    normalized_text: str             # Cleaned Unicode NFC text
    bbox: Optional[BoundingBox]      # Bounding box coordinates
    page_number: int                 # 1-indexed page number
    language: str                    # 'kannada', 'english', etc.
    script: str                      # 'Kannada', 'Latin', etc.
    is_handwritten: Optional[bool]   # True=Handwritten, False=Printed
    confidence: Optional[float]      # Real posterior probability [0.0 - 1.0]
    model_name: str                  # Engine identifier
    model_version: str               # Version tag (e.g. 'kannada_full_v1')
    inference_time_ms: float         # Execution duration in milliseconds
    preprocessing_metadata: dict     # Filters applied (deskew angle, etc.)
    requires_human_review: bool      # Flagged if conf < 0.60 or unsupported
    status: ProcessingStatus         # SUCCESS, LOW_CONFIDENCE, UNSUPPORTED, EMPTY
    candidate_suggestions: List[str] # Lexicon domain term suggestions
    custom_metadata: dict            # Audit trail & step softmax values
```

#### `DocumentProcessingResponse`
Returned at document level:
```python
class DocumentProcessingResponse(BaseModel):
    document_id: Optional[str]
    page_number: int
    image_path: str
    ordered_regions: List[RecognizedRegionResult]  # Sorted top-to-bottom reading order
    merged_text: str                               # Full page concatenated text
    document_confidence: Optional[float]           # Length-weighted mean confidence
    status: str                                    # 'completed' or 'flagged_for_review'
    requires_human_review: bool                    # True if ANY region requires review
    warnings: List[str]                            # List of review reasons/warnings
    engine_breakdown: Dict[str, int]               # e.g. {'TrOCR-Kannada': 3, 'PaddleOCR': 1}
    processing_time_ms: float
    page: DocumentPage                             # Backwards-compatible schema
```

---

## Integration Code Examples

### Person A Integration Example (Supplying Layout Detections)

Person A can pass raw bounding box dictionaries, tuples, or Pydantic models directly to Person B without custom conversion:

```python
from PIL import Image
from src.integration import process_document

# Load document image
doc_image = Image.open("data/samples/rtc_record_page1.jpg")

# Person A YOLO / LayoutLM bounding boxes output
layout_boxes = [
    {
        "id": "header_01",
        "bbox": [50, 40, 750, 100],
        "label": "printed_header",
        "language": "kannada",
        "is_handwritten": False,
    },
    {
        "id": "owner_hw_02",
        "bbox": [60, 140, 400, 210],
        "label": "handwritten_name",
        "language": "kannada",
        "is_handwritten": True,
    },
]

# Run Person B pipeline
response = process_document(
    image=doc_image,
    regions=layout_boxes,
    document_id="RTC_2026_0987",
)
```

---

### Person C Integration Example (Information Extraction & Database Ingestion)

Person C consumes structured regions and review flags to populate land record tables:

```python
from src.integration import process_document

response = process_document(image="data/samples/rtc_record_page1.jpg")

# 1. Check if document requires human review
if response.requires_human_review:
    print(f"[ALERT] Document {response.document_id} flagged for human review:")
    for warn in response.warnings:
        print(f"  - {warn}")

# 2. Iterate structured regions in reading order
for region in response.ordered_regions:
    print(f"Region {region.region_id} [{region.language}]:")
    print(f"  Raw Text       : {region.raw_text}")
    print(f"  Normalized Text: {region.normalized_text}")
    print(f"  Confidence     : {region.confidence:.2%}")
    print(f"  Engine Used    : {region.model_name}")

    if region.candidate_suggestions:
        print(f"  Lexicon Match  : {region.candidate_suggestions}")

# 3. Access full merged text
full_document_text = response.merged_text
```

---

## Human Correction & Active Learning Workflow

When a region is flagged with `requires_human_review=True`, human operators review and submit corrections using `CorrectionService`.

```python
from src.correction import CorrectionService

service = CorrectionService(storage_path="data/corrections/human_corrections.jsonl")

# 1. Record a verified correction
service.record_correction({
    "document_id": "RTC_2026_0987",
    "region_id": "owner_hw_02",
    "image_path": "crops/RTC_2026_0987_owner_hw_02.png",
    "raw_prediction": "ಸಚಿವರಿದ್ದರೂ",
    "corrected_text": "ಸಚಿವರಾಗಿದ್ದರೂ",
    "ai_confidence": 0.8247,
    "language": "kannada",
    "is_handwritten": True,
    "model_version": "kannada_full_v1",
    "reviewer_id": "REV_OFFICER_42",
})

# 2. Export training manifest for future active learning batch
# Note: This does NOT automatically trigger retraining.
exported_count = service.export_training_manifest(
    output_path="training/datasets/active_learning_kannada.jsonl",
    is_handwritten_only=True,
)
print(f"Exported {exported_count} samples for next training phase.")
```

---

## Known Limitations

1. **Complex Conjunct & Ligature Variations**: Extremely ornate or degraded cursive Kannada conjuncts (e.g. `ಾಗಿದ್ದರೂ` vs `ಿದ್ದರೂ`) may require lexicon-assisted review.
2. **Unsupported Indic Handwriting**: Non-Kannada Indic handwriting (Telugu, Tamil, Hindi, Malayalam) is architected for future checkpoints; currently, the pipeline returns transparent review flags (`requires_human_review=True`, `status="unsupported_language"`) rather than fabricating text.
3. **Upstream Layout Boundary Precision**: Accurate crop recognition relies on bounding boxes encompassing the full character ascenders and descenders (matras).
