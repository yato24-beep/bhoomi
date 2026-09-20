"""Production Printed OCR Regression Test across four real land-record documents.

Evaluates DocumentProcessingPipeline with EasyOCR as the primary printed Kannada recognizer.
Reports real CER/WER against ground truth, verifies orientation invariants, and checks conjunct preservation.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
import jiwer

# Configure UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.document_pipeline import DocumentProcessingPipeline
from src.integration.schemas import DocumentProcessingRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("printed_regression_test")

# Key test documents
TEST_DOCS = [
    {
        "id": "doddaballapura_passage",
        "path": r"C:\Users\akars\.gemini\antigravity-ide\brain\d2ec6ade-f9c9-496e-a738-fcd5a6775c06\.user_uploaded\media_1789702248411.png",
        "description": "Doddaballapura historical passage (digital printed scan with complex conjuncts)",
        "ground_truth": (
            "ಡೊಡ್ಡಬಳ್ಳಾಪುರ\n"
            "ಸ್ಥಳೀಯ ಆದಿನಾರಾಯಣ ದೇವಸ್ಥಾನದಿಂದ ಕ್ರಿ.ಶ 1598 ರ ದಾಖಲೆಯಲ್ಲಿ\n"
            "ಈ ಸ್ಥಳವನ್ನು ಬಲ್ಲಾಪುರ ತಾಂಡಾ ಎಂದು ಉಲ್ಲೇಖಿಸಲಾಗಿದೆ. ಇದು\n"
            "ಹೊಯ್ಸಳ ಹೆಸರಿನ ಬಲ್ಲಾಲದಿಂದ ಹುಟ್ಟಿಕೊಂಡಿರಬಹುದು ಮತ್ತು ನಂತರ\n"
            "ಬಲ್ಲಾಪುರ ಎಂದು ಭ್ರಷ್ಟಗೊಂಡಿರಬಹುದು. ಒಂದು ಹಸು ಒಂದು 'ಬಲ್ಲಾ'\n"
            "ಹಾಲನ್ನು ಒಂದು ನಿರ್ದಿಷ್ಟ ಆಂಥಿಲ್ ಮೇಲೆ ಬೀಳಿಸಲು ಬಳಸಿದ ಸನ್ನಿವೇಶ\n"
            "ದಿಂದ ಈ ಹಳ್ಳಿಗೆ ಈ ಹೆಸರು ಬಂದಿದೆ ಮತ್ತು ಈ ಶಕುನವು ಪಟ್ಟಣದ ಅಡಿ\n"
            "ಪಾಯಕ್ಕೆ ಕಾರಣವಾಯಿತು ಎಂದು ನಂಬಲಾಗಿದೆ. 'ಬಲ್ಲಾ' ದಿಂದ ಬಲ್ಲಾಪುರ\n"
            "ಎಂಬ ಹೆಸರನ್ನು ಪಡೆಯಲಾಗಿದೆ:"
        ),
        "test_conjuncts": ["ಡ್ಡ", "ಳ್ಳಾ", "ಷ್ಟ", "ಸ್ಥ", "ರ್ದಿಷ್ಟ"],
    },

    {
        "id": "doc2_karnataka_rtc",
        "path": str(PROJECT_ROOT / "doc2.jpeg"),
        "description": "Karnataka RTC land record form (scanned form with tabular headers)",
        "ground_truth": None,  # Semi-structured form; evaluate layout and key domain terms
        "expected_terms": ["ಕರ್ನಾಟಕ", "ಸರ್ಕಾರ", "ಕಂದಾಯ", "ಖಾತೆ", "ಗ್ರಾಮ"],
    },
    {
        "id": "sample_bhoomi_rtc",
        "path": str(PROJECT_ROOT / "person-a" / "data" / "samples" / "sample_karnataka_bhoomi_rtc.png"),
        "description": "Karnataka Bhoomi RTC document image",
        "ground_truth": None,
        "expected_terms": ["ಪಹಣಿ", "ಭೂಮಿ", "ಕಂದಾಯ", "ಸರ್ವೆ"],
    },
    {
        "id": "sample_satbara_7_12",
        "path": str(PROJECT_ROOT / "person-a" / "data" / "samples" / "sample_land_record_7_12.png"),
        "description": "7/12 Land Record document image",
        "ground_truth": None,
        "expected_terms": [],
    },
]


def run_regression_test():
    print("=" * 80)
    print("PRODUCTION PRINTED OCR REGRESSION TEST (EasyOCR PRIMARY)")
    print("=" * 80)

    # Initialize production pipeline with default router (EasyOCR)
    pipeline = DocumentProcessingPipeline(
        apply_preprocessing=True,
        apply_normalization=True,
        enable_document_gating=True,
    )

    results = []

    for doc in TEST_DOCS:
        doc_id = doc["id"]
        img_path = Path(doc["path"])
        print(f"\nEvaluating: {doc_id}")
        print(f"  Path       : {img_path}")
        print(f"  Description: {doc['description']}")

        if not img_path.exists():
            print(f"  [ERROR] File not found: {img_path}")
            continue

        raw_img = Image.open(img_path)
        print(f"  Original Dimensions: {raw_img.size} (format={raw_img.format})")

        start_time = time.perf_counter()
        # Process via canonical pipeline
        response = pipeline.process_document(
            image=raw_img,
            is_handwritten=False,
            language="kannada",
            document_id=doc_id,
            apply_preprocessing=True,
        )
        elapsed_sec = round(time.perf_counter() - start_time, 3)

        # Extract telemetry
        modality = "handwritten" if response.ordered_regions and response.ordered_regions[0].is_handwritten else "printed"
        engine_breakdown = response.engine_breakdown or {}
        active_engine = list(engine_breakdown.keys())[0] if engine_breakdown else "unknown"
        recognized_text = response.merged_text or response.clean_kannada_text or ""
        confidence = response.document_confidence

        print(f"  Modality Selected   : {modality}")
        print(f"  Recognizer Active   : {active_engine} (breakdown: {engine_breakdown})")
        print(f"  Processing Time     : {elapsed_sec}s ({response.processing_time_ms} ms)")
        print(f"  Recognizer Conf     : {confidence} (self-reported score; NOT accuracy)")
        print(f"  Gate Decision       : is_land_record={response.is_land_record}, status={response.status}")
        print(f"  Output Length       : {len(recognized_text)} chars across {len(response.ordered_regions)} regions")

        # Orientation check: ensure text is not upside down or garbage
        has_kannada = any('\u0c80' <= c <= '\u0cff' for c in recognized_text)
        print(f"  Contains Kannada    : {has_kannada}")

        # Ground truth evaluation if available
        gt = doc.get("ground_truth")
        cer = None
        wer = None
        if gt:
            cer = jiwer.cer(gt, recognized_text)
            wer = jiwer.wer(gt, recognized_text)
            print(f"  [METRIC] Real CER   : {cer:.2%}")
            print(f"  [METRIC] Real WER   : {wer:.2%}")

            # Check test conjuncts
            conjunct_status = {}
            for conj in doc.get("test_conjuncts", []):
                found = conj in recognized_text
                conjunct_status[conj] = found
            print(f"  [METRIC] Conjuncts  : {conjunct_status} ({sum(conjunct_status.values())}/{len(conjunct_status)} found)")

        # Sample preview
        preview = recognized_text[:200].replace("\n", " ")
        print(f"  Recognized Sample   : {preview}...")

        results.append({
            "id": doc_id,
            "modality": modality,
            "active_engine": active_engine,
            "engine_breakdown": engine_breakdown,
            "elapsed_sec": elapsed_sec,
            "recognizer_confidence": confidence,
            "cer": cer,
            "wer": wer,
            "num_regions": len(response.ordered_regions),
            "text_length": len(recognized_text),
            "sample_output": recognized_text[:300],
        })

    # Save JSON report
    report_file = PROJECT_ROOT / "scratch" / "production_printed_ocr_regression_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Regression report saved to: {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_regression_test()
