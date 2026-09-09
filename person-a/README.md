# Person A: Land Record Document Intelligence & Vision Engine

Production-grade vision, adaptive preprocessing, printed multilingual OCR (PP-OCRv5), morphological table extraction, document classification, and consensus pipeline for Indian Land Records.

---

## Features

- **Ingestion & Hashing**: Ingests PDF, PNG, JPG, TIFF, raw bytes, and streams with deterministic SHA-256 identity calculation.
- **Image Quality Analysis**: Real blur variance, RMS contrast, quadrant illumination uniformity, and estimated DPI scoring.
- **Adaptive OpenCV Preprocessing**: Selective CLAHE contrast, bilateral denoising, unsharp sharpening, and Radon/Hough deskewing.
- **Selective Real-ESRGAN**: Neural 2x super-resolution triggered when estimated scan DPI is low.
- **Printed Multilingual OCR**: PaddleOCR / PP-OCRv5 multilingual models structuring text into `OCRBlock` -> `OCRLine` -> `OCRWord` hierarchies with exact polygons and confidences.
- **Layout & Tables**: Real OpenCV morphological line & grid segmentation extracting table cells, row/col span, Markdown tables, and JSON dataframes.
- **Document Classification**: State-config JSON driven classification for 7/12 Satbara, Bhoomi RTC, Bhulekh Khatauni, Patta Chitta, Khasra, Jamabandi, etc.
- **Region Routing**: Routes printed blocks to PaddleOCR and handwritten crops to Person B's handwriting adapter.
- **OCR Consensus**: Multi-candidate normalized Levenshtein distance analysis preserving provenance and applying conflict penalization.
- **Downstream Adapters**: Out-of-the-box adapters mapping results to Person C canonical `DocumentOCRResult` and Backend `ProcessingResult`.

---

## Quick Start Demo

```bash
cd person-a
python demo.py --input data/samples/sample_land_record_7_12.png --state MH --output sample_output.json
```

---

## Running Tests

```bash
cd person-a
pytest tests/ -v
```
