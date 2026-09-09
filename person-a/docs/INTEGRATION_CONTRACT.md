# Person A Integration Contract

## 1. Primary Entry Point

```python
from person_a.src.pipeline import process_document, process_document_with_handwriting
from person_a.src.schemas import DocumentInput, OCROutput

# Polymorphic signature
ocr_output: OCROutput = process_document(
    document_input="path/to/land_record.png", # str, Path, bytes, BinaryIO, np.ndarray, DocumentInput
    state_hint="MH",                          # Optional state code (MH, KA, UP, TN, BR, MP, DEFAULT)
    target_languages=["mr", "en"],            # Optional target languages
)
```

## 2. Person C Conversion

```python
from person_a.src.integration.person_c_adapter import convert_person_a_to_document_ocr_result

person_c_payload: dict = convert_person_a_to_document_ocr_result(ocr_output)
# Conforms 100% to DocumentOCRResult in person-c/schemas.py
```

## 3. Backend Processing Result Adapter

```python
from person_a.src.integration.backend_adapter import process_backend_stream_to_result

result: dict = process_backend_stream_to_result(
    file_stream=minio_stream,
    filename="sample_land_record_7_12.png",
    state_hint="MH",
)
# Conforms 100% to backend/app/pipeline/processor.py ProcessingResult
```
