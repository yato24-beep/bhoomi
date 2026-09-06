"""
src/cli.py
Command Line Interface for Person C: Structured Land Record Extraction & Validation.
Provides interactive commands to process sample documents, run GIS queries, and execute evaluations.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas import (
    BoundingBox,
    DocumentClassificationResult,
    DocumentOCRResult,
    DocumentType,
    ExtractedField,
    OCREngineType,
    OCRTextLine,
)
from src.database.duplicates import DuplicateDetector
from src.database.gis import GISValidator
from src.integration.person_c_service import extract_and_validate
from src.utils.config_loader import ConfigLoader


def print_result(res):
    """Formats and prints the FinalDocumentResult clearly to terminal."""
    print("\n" + "=" * 70)
    print("PERSON C: EXTRACTION & VALIDATION RESULT")
    print("=" * 70)
    print(f"Document ID       : {res.document_id}")
    print(f"State             : {res.state}")
    print(f"Document Type     : {res.document_type.value}")
    print(f"Validation Status : {res.validation_status.value.upper()}")
    print(f"Overall Confidence: {res.overall_confidence:.3f}")
    print(f"Requires Review   : {res.requires_human_review}")
    if res.review_reasons:
        print(f"Review Reasons    : {res.review_reasons}")
    print("\nExtracted Fields:")
    for fname, fobj in res.fields.items():
        unit_str = f" {fobj.normalized_unit}" if fobj.normalized_unit else ""
        print(f"  - {fname:<22}: {str(fobj.normalized_value) + unit_str:<25} (Raw: '{fobj.raw_value}', Conf: {fobj.confidence:.2f})")
    print("\nCadastral GIS Verification:")
    print(f"  - Status          : {res.gis_validation.gis_status.value}")
    print(f"  - Verified        : {res.gis_validation.is_verified}")
    print(f"  - Recorded GIS Ha : {res.gis_validation.gis_recorded_area_hectares}")
    if res.gis_validation.flag_reasons:
        print(f"  - GIS Flags       : {res.gis_validation.flag_reasons}")
    print("=" * 70 + "\n")


def run_cli():
    parser = argparse.ArgumentParser(description="Person C: Land Record Extraction & Validation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: process-file
    file_parser = subparsers.add_parser("process-file", help="Process an OCR JSON sample file through Person C pipeline")
    file_parser.add_argument("file_path", help="Path to input OCR JSON file")

    # Command: process-sample
    proc_parser = subparsers.add_parser("process-sample", help="Run full Person C pipeline on sample text")
    proc_parser.add_argument("--state", default="UP", help="Target state code (e.g. UP, MP, MH, BR, KA, TN)")
    proc_parser.add_argument("--khasra", default="142/1", help="Sample Khasra/Plot number")
    proc_parser.add_argument("--owner", default="राम प्रसाद", help="Sample owner name")
    proc_parser.add_argument("--area", default="0.4500", help="Sample land area in hectares")
    proc_parser.add_argument("--village", default="मऊ", help="Sample village name")

    # Command: test-gis
    gis_parser = subparsers.add_parser("test-gis", help="Validate cadastral parcel coordinates against GIS")
    gis_parser.add_argument("--state", default="UP")
    gis_parser.add_argument("--district", default="LUCKNOW")
    gis_parser.add_argument("--tehsil", default="MOHANLALGANJ")
    gis_parser.add_argument("--village", default="MAU")
    gis_parser.add_argument("--khasra", default="142/1")
    gis_parser.add_argument("--area", type=float, default=0.4500)

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run multi-state document evaluation benchmark")

    args = parser.parse_args()

    if args.command == "process-file":
        with open(args.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ocr_res = DocumentOCRResult.model_validate(data)
        res = extract_and_validate(ocr_res, selected_state=data.get("selected_state"))
        print_result(res)

    elif args.command == "process-sample":
        print(f"\n[PERSON C] Processing sample {args.state} document...")
        lines = [
            OCRTextLine(
                text=f"ग्राम: {args.village}  तहसील: मोहनलालगंज  जिला: लखनऊ",
                confidence=0.98,
                bbox=BoundingBox(x_min=10, y_min=10, x_max=400, y_max=40),
            ),
            OCRTextLine(
                text=f"खातेदार का नाम: {args.owner}  पिता: श्याम लाल",
                confidence=0.96,
                bbox=BoundingBox(x_min=10, y_min=50, x_max=400, y_max=80),
            ),
            OCRTextLine(
                text=f"गाटा संख्या: {args.khasra}  क्षेत्रफल: {args.area} हेक्टेयर",
                confidence=0.97,
                bbox=BoundingBox(x_min=10, y_min=90, x_max=450, y_max=120),
            ),
        ]
        ocr_res = DocumentOCRResult(
            document_id="CLI_DEMO_001",
            sha256_hash="hash_cli_demo_001",
            classification=DocumentClassificationResult(document_type=DocumentType.KHATAUNI, state=args.state),
            text_lines=lines,
            raw_full_text="\n".join(l.text for l in lines),
        )

        res = extract_and_validate(ocr_res, selected_state=args.state)
        print_result(res)

    elif args.command == "test-gis":
        print(f"\n[PERSON C] Querying Cadastral GIS for {args.state}/{args.district}/{args.tehsil}/{args.village}/{args.khasra}...")
        validator = GISValidator()
        fields = {
            "district": ExtractedField(field_name="district", raw_value=args.district, normalized_value=args.district, confidence=1.0),
            "tehsil": ExtractedField(field_name="tehsil", raw_value=args.tehsil, normalized_value=args.tehsil, confidence=1.0),
            "village": ExtractedField(field_name="village", raw_value=args.village, normalized_value=args.village, confidence=1.0),
            "khasra_number": ExtractedField(field_name="khasra_number", raw_value=args.khasra, normalized_value=args.khasra, confidence=1.0),
            "land_area": ExtractedField(field_name="land_area", raw_value=str(args.area), normalized_value=args.area, confidence=1.0),
        }
        res = validator.validate_gis(fields, state_code=args.state)
        print(f"GIS Status        : {res.gis_status.value}")
        print(f"GIS Verified      : {res.is_verified}")
        print(f"Mismatch Detected : {res.has_mismatch}")
        print(f"Cadastral Area Ha : {res.gis_recorded_area_hectares}")
        print(f"Flags/Reasons     : {res.flag_reasons}")

    elif args.command == "evaluate":
        from evaluation.evaluate_pipeline import run_benchmark
        run_benchmark()
    else:
        parser.print_help()


if __name__ == "__main__":
    run_cli()
