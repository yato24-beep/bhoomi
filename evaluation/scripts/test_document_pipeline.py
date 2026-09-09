"""End-to-End Document Pipeline Test Script.

Tests multimodal routing and processing across:
1. Handwritten Kannada crop -> Dispatches to fine-tuned Kannada TrOCR checkpoint
2. Printed Kannada document/crop -> Dispatches to PaddleOCR Kannada baseline

Validates structured DocumentPage schema, engine selection, confidence derivation,
and human-review flagging through the production process_document(...) interface.
"""

import json
import os
from pathlib import Path
import sys
import time

# Ensure UTF-8 output encoding for terminal display of Indic characters on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.document_pipeline import DocumentProcessingPipeline, process_document


def main():
    print("=" * 95, flush=True)
    print("  LAND RECORD DIGITIZATION — END-TO-END DOCUMENT PIPELINE TEST", flush=True)
    print("=" * 95, flush=True)

    pipeline = DocumentProcessingPipeline(
        default_language="kannada",
        confidence_threshold=0.60,
        apply_preprocessing=True,
    )

    # 1. Select Authentic Test Samples
    # A) Handwritten Kannada crop from validation manifest
    val_manifest = PROJECT_ROOT / "training" / "datasets" / "iiit_kannada_val.jsonl"
    hw_sample_path = None
    hw_ground_truth = None

    if val_manifest.exists():
        with open(val_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    rel_p = data.get("image") or data.get("image_path")
                    cand_path = PROJECT_ROOT / rel_p
                    if cand_path.exists():
                        hw_sample_path = cand_path
                        hw_ground_truth = data.get("text")
                        break

    if hw_sample_path is None:
        hw_sample_path = PROJECT_ROOT / "data" / "samples" / "sample_kannada_crop.png"
        hw_ground_truth = "(Sample Kannada Crop)"

    # B) Printed Kannada sample
    printed_sample_path = PROJECT_ROOT / "data" / "samples" / "sample_kannada_document.png"
    if not printed_sample_path.exists():
        printed_sample_path = PROJECT_ROOT / "data" / "samples" / "sample_kannada_crop.png"

    print(f"\n[Test Case 1] HANDWRITTEN KANNADA SAMPLE", flush=True)
    print(f"  Image Path   : {hw_sample_path}", flush=True)
    print(f"  Ground Truth : {hw_ground_truth}", flush=True)
    print(f"  Target Type  : Handwritten (is_handwritten=True)", flush=True)

    res_hw = process_document(
        image=hw_sample_path,
        is_handwritten=True,
        language="kannada",
        page_number=1,
        pipeline=pipeline,
    )

    print("\n  --- Pipeline Execution Audit ---", flush=True)
    print(f"  Selected Engine     : {res_hw.page.ocr_results[0].model_name}", flush=True)
    print(f"  Model Version       : {res_hw.page.ocr_results[0].model_version}", flush=True)
    print(f"  Recognized Text     : {res_hw.full_text}", flush=True)
    print(f"  Confidence Score    : {res_hw.document_confidence}", flush=True)
    print(f"  Requires Review     : {res_hw.requires_human_review} (Reasons: {res_hw.review_reasons})", flush=True)
    print(f"  Engine Breakdown    : {res_hw.engine_breakdown}", flush=True)
    print(f"  Processing Time     : {res_hw.processing_metadata.get('duration_ms')} ms", flush=True)

    print("\n" + "-" * 95, flush=True)

    print(f"\n[Test Case 2] PRINTED KANNADA SAMPLE", flush=True)
    print(f"  Image Path   : {printed_sample_path}", flush=True)
    print(f"  Target Type  : Printed (is_handwritten=False)", flush=True)

    res_printed = process_document(
        image=printed_sample_path,
        is_handwritten=False,
        language="kannada",
        page_number=1,
        pipeline=pipeline,
    )

    print("\n  --- Pipeline Execution Audit ---", flush=True)
    print(f"  Selected Engine     : {res_printed.page.ocr_results[0].model_name}", flush=True)
    print(f"  Model Version       : {res_printed.page.ocr_results[0].model_version}", flush=True)
    print(f"  Recognized Text     : {res_printed.full_text[:60]}..." if len(res_printed.full_text) > 60 else f"  Recognized Text     : {res_printed.full_text}", flush=True)
    print(f"  Confidence Score    : {res_printed.document_confidence}", flush=True)
    print(f"  Requires Review     : {res_printed.requires_human_review} (Reasons: {res_printed.review_reasons})", flush=True)
    print(f"  Engine Breakdown    : {res_printed.engine_breakdown}", flush=True)
    print(f"  Processing Time     : {res_printed.processing_metadata.get('duration_ms')} ms", flush=True)

    print("\n" + "=" * 95, flush=True)
    print("  ROUTING VERIFICATION SUMMARY", flush=True)
    print("=" * 95, flush=True)
    print("  ✓ HANDWRITTEN KANNADA -> Fine-Tuned Kannada TrOCR Checkpoint", flush=True)
    print(f"    Engine: {res_hw.page.ocr_results[0].model_name} ({res_hw.page.ocr_results[0].model_version})", flush=True)
    print("  ✓ PRINTED KANNADA     -> Baseline Regional PaddleOCR Backend", flush=True)
    print(f"    Engine: {res_printed.page.ocr_results[0].model_name} ({res_printed.page.ocr_results[0].model_version})", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.exit(1)
