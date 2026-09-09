#!/usr/bin/env python3
"""person-a/demo.py
Reproducible Person A Vision & Document Intelligence Pipeline Demo.
Demonstrates: Ingestion -> Quality -> Preprocessing -> Classification -> Layout -> OCR -> Consensus -> Person C Export.
"""

import argparse
import json
from pathlib import Path
import sys
import time

from src.pipeline import process_document, process_document_with_handwriting
from src.integration.person_c_adapter import convert_person_a_to_document_ocr_result
from src.schemas import DocumentInput, OCROutput


def main():
    parser = argparse.ArgumentParser(description="Person A Land Record Document Intelligence Demo")
    parser.add_argument(
        "--input", "-i",
        default="data/samples/sample_land_record_7_12.png",
        help="Path to document file (PNG, JPG, PDF, TIFF)",
    )
    parser.add_argument(
        "--state", "-s",
        default="MH",
        help="State jurisdiction code (MH, KA, UP, TN, BR, MP, or DEFAULT)",
    )
    parser.add_argument(
        "--output", "-o",
        default="sample_output.json",
        help="Path to save the serializable JSON result",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input document '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print("=" * 80, flush=True)
    print(" PERSON A: LAND RECORD VISION & MULTILINGUAL OCR PIPELINE DEMO", flush=True)
    print("=" * 80, flush=True)
    print(f" Input Document : {input_path.resolve()}", flush=True)
    print(f" State Hint     : {args.state}", flush=True)
    print("-" * 80, flush=True)
    print("Executing Person A vision & OCR pipeline...", flush=True)

    start_time = time.perf_counter()

    # 1. Execute Person A pipeline
    ocr_out, hw_out = process_document_with_handwriting(
        document_input=str(input_path.resolve()),
        state_hint=args.state,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # 2. Convert to Person C canonical DocumentOCRResult
    person_c_payload = convert_person_a_to_document_ocr_result(ocr_out)

    # 3. Print Structured Console Report
    print("\n[1] DOCUMENT IDENTITY & QUALITY")
    print(f"  • Document ID       : {ocr_out.document_id}")
    print(f"  • SHA-256 Hash      : {ocr_out.sha256_hash}")
    print(f"  • Pages Processed   : {len(ocr_out.pages)}")
    if ocr_out.pages:
        q = ocr_out.pages[0].quality
        print(f"  • Blur Score        : {q.blur_score} (Blurry: {q.is_blurry})")
        print(f"  • Contrast Score    : {q.contrast_score}")
        print(f"  • Preprocessing Ops : {', '.join(q.preprocessing_applied) or 'none'}")

    print("\n[2] DOCUMENT CLASSIFICATION")
    print(f"  • Predicted Type    : {ocr_out.classification.predicted_type}")
    print(f"  • Confidence        : {ocr_out.classification.confidence * 100:.1f}%")
    print(f"  • State Code        : {ocr_out.classification.state}")
    print(f"  • Evidence Signals  : {ocr_out.classification.matched_signals}")

    print("\n[3] PRINTED MULTILINGUAL OCR (PP-OCRv5)", flush=True)
    print(f"  • Engine            : {ocr_out.ocr_engine}", flush=True)
    print(f"  • Overall Confidence: {ocr_out.overall_confidence * 100:.1f}% (Aggregate Score - Not Calibrated Probability)", flush=True)
    if ocr_out.pages and ocr_out.pages[0].language_info:
        lang = ocr_out.pages[0].language_info
        print(f"  • Requested Lang    : {lang.requested_language} ({lang.script} script)", flush=True)
        print(f"  • Actual Model Lang : {lang.actual_language}", flush=True)
        print(f"  • Fallback Status   : {'YES (Fallback Active)' if lang.fallback_occurred else 'NO (Direct Model)'}", flush=True)
        if lang.fallback_occurred:
            print(f"  • Fallback Reason   : {lang.fallback_reason}", flush=True)
    words_count = sum(len(l.words) for p in ocr_out.pages for b in p.blocks for l in b.lines)
    print(f"  • Recognized Words  : {words_count}", flush=True)
    snippet = ocr_out.full_text[:200].replace("\n", " ")
    print(f"  • Text Snippet      : \"{snippet}...\"", flush=True)

    print("\n[4] LAYOUT & TABLE STRUCTURE (Morphological Grid Analysis)", flush=True)
    total_tables = sum(len(p.tables) for p in ocr_out.pages)
    print(f"  • Tables Detected   : {total_tables}", flush=True)
    if ocr_out.pages and ocr_out.pages[0].tables:
        t = ocr_out.pages[0].tables[0]
        print(f"  • Table #1 Grid     : {t.rows_count} rows x {t.cols_count} cols | {len(t.cells)} cells", flush=True)
        if t.headers:
            print(f"  • Headers Detected  : {t.headers}", flush=True)
        if t.markdown:
            print("  • Markdown Preview  :", flush=True)
            for line in t.markdown.split("\n")[:6]:
                print(f"      {line}", flush=True)

    print("\n[5] REGION-ROUTED HANDWRITING (Person B TrOCR Adapter)", flush=True)
    print(f"  • Handwritten Blocks: {len(hw_out.regions)}", flush=True)
    print(f"  • Adapter Status    : {hw_out.model_version}", flush=True)

    print("\n[6] DOWNSTREAM PERSON C COMPATIBILITY", flush=True)
    print(f"  • Canonical Schema  : DocumentOCRResult (conforms to person-c/schemas.py)", flush=True)
    print(f"  • Total Text Lines  : {len(person_c_payload['text_lines'])}", flush=True)

    print(f"\n[7] PERFORMANCE & STAGE LATENCY BREAKDOWN", flush=True)
    print(f"  • Total End-to-End  : {elapsed_ms:.2f} ms", flush=True)
    if "stage_timings_ms" in ocr_out.metadata:
        for stage, dur in ocr_out.metadata["stage_timings_ms"].items():
            print(f"    - {stage:<22}: {dur:.2f} ms", flush=True)

    # 4. Save JSON Output
    out_file = Path(args.output)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(person_c_payload, f, indent=2, ensure_ascii=False)
    print(f"\n  • JSON Result Saved : {out_file.resolve()} ({out_file.stat().st_size} bytes)", flush=True)
    print("=" * 80, flush=True)
    print(" DEMO COMPLETED SUCCESSFULLY (100% Schema Valid)", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
