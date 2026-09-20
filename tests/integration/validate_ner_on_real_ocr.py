"""Phase 6: Validate NER on Real OCR Outputs.

Evaluates LandRecordFieldExtractor against real OCR outputs from the
production pipeline across real test documents.

Separates:
- OCR Errors (e.g. degraded anchor/value recognition by OCR engine)
- Extraction Errors (e.g. anchor present but parser failed to extract or associate value)

Computes field-level precision, recall, and F1 where ground truth exists.
"""

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.extraction.land_record_ner import LandRecordFieldExtractor, FIELD_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("validate_ner_on_real_ocr")

# Define ground truth for structured test documents
GROUND_TRUTH = {
    "sample_bhoomi_rtc": {
        "is_cadastral_record": True,
        "fields": {
            "survey_number": "45/1",
            "owner_name": "Manjunath Gowda",
            "extent_area": "2-14 Acres",
        }
    },
    "sample_satbara_7_12": {
        "is_cadastral_record": True,
        "fields": {
            "district": "Pune",
            "taluk": "Haveli",
            "village": "Shirur",
            "survey_number": "124/2A",
        }
    },
    "doddaballapura_passage": {
        "is_cadastral_record": False,
        "fields": {}  # Should extract 0 cadastral fields (tests false positive suppression)
    },
    "doc2_karnataka_rtc": {
        "is_cadastral_record": True,
        "fields": {
            # Low resolution real RTC - evaluating degradation handling
            "document_type": "RTC Form 16",
        }
    }
}


def run_ner_validation():
    report_path = Path("scratch/production_printed_ocr_regression_report.json")
    if not report_path.exists():
        logger.error(f"OCR regression report missing: {report_path}")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        ocr_results = json.load(f)

    extractor = LandRecordFieldExtractor(confidence_threshold=0.50)
    validation_results = []

    total_gt_fields = 0
    total_extracted_fields = 0
    correctly_extracted_fields = 0

    print("\n" + "=" * 80)
    print("PHASE 6: VALIDATING NER ON REAL OCR OUTPUTS (EASYOCR PRODUCTION OUTPUT)")
    print("=" * 80)

    for doc in ocr_results:
        doc_id = doc["id"]
        gt_info = GROUND_TRUTH.get(doc_id, {"is_cadastral_record": False, "fields": {}})
        raw_text = doc.get("sample_output", "")
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

        extracted = extractor.extract_fields(lines)
        extracted_dict = {k: v.to_dict() for k, v in extracted.items()}

        gt_fields = gt_info["fields"]
        doc_eval = {
            "id": doc_id,
            "is_cadastral_record": gt_info["is_cadastral_record"],
            "lines_count": len(lines),
            "extracted_fields": extracted_dict,
            "ground_truth_fields": gt_fields,
            "field_matches": {},
            "error_analysis": [],
        }

        print(f"\n--- Document: {doc_id} ---")
        print(f"Lines passed to NER: {len(lines)}")
        print(f"Fields extracted: {list(extracted.keys())}")

        if not gt_info["is_cadastral_record"]:
            # Negative test: narrative document should produce 0 cadastral extractions
            fp_count = len(extracted)
            if fp_count == 0:
                print("  PASS: False-positive suppression successful (0 fields extracted from non-cadastral text).")
                doc_eval["false_positive_suppression"] = "PASS"
            else:
                print(f"  WARNING: {fp_count} false-positive fields extracted from non-cadastral narrative:")
                for k, v in extracted.items():
                    print(f"    - {k}: {v.raw_value} (Evidence: {v.evidence_text})")
                doc_eval["false_positive_suppression"] = f"FAIL ({fp_count} false positives)"
        else:
            # Cadastral document: evaluate fields
            for field_name, expected_val in gt_fields.items():
                total_gt_fields += 1
                if field_name in extracted:
                    total_extracted_fields += 1
                    act = extracted[field_name].normalized_value
                    # Check semantic match or partial match
                    if expected_val.lower() in act.lower() or act.lower() in expected_val.lower():
                        correctly_extracted_fields += 1
                        doc_eval["field_matches"][field_name] = {
                            "status": "EXACT_OR_SUBSTRING_MATCH",
                            "expected": expected_val,
                            "actual": act,
                            "confidence": extracted[field_name].confidence,
                        }
                        print(f"  MATCH [{field_name}]: Expected '{expected_val}', Got '{act}' (conf: {extracted[field_name].confidence:.2f})")
                    else:
                        # Value mismatch - diagnose cause
                        error_type = "OCR_ERROR"
                        # Check if expected value was present anywhere in OCR text
                        if any(expected_val.lower() in l.lower() for l in lines):
                            error_type = "EXTRACTION_PARSER_ERROR"
                        
                        doc_eval["field_matches"][field_name] = {
                            "status": "VALUE_MISMATCH",
                            "expected": expected_val,
                            "actual": act,
                            "root_cause": error_type,
                        }
                        doc_eval["error_analysis"].append({
                            "field": field_name,
                            "root_cause": error_type,
                            "detail": f"Expected '{expected_val}', but OCR gave '{act}'",
                        })
                        print(f"  MISMATCH [{field_name}] ({error_type}): Expected '{expected_val}', Got '{act}'")
                else:
                    # Field missed
                    # Check if anchor was present in OCR lines
                    anchors = FIELD_ANCHORS.get(field_name, [])
                    anchor_in_ocr = any(any(a.lower() in l.lower() for a in anchors) for l in lines)
                    root_cause = "EXTRACTION_PARSER_ERROR" if anchor_in_ocr else "OCR_ERROR (Anchor corrupted/absent in OCR output)"

                    doc_eval["field_matches"][field_name] = {
                        "status": "MISSED",
                        "expected": expected_val,
                        "root_cause": root_cause,
                    }
                    doc_eval["error_analysis"].append({
                        "field": field_name,
                        "root_cause": root_cause,
                        "detail": f"Anchor missing or corrupted in OCR output: {anchors[:3]}",
                    })
                    print(f"  MISSED [{field_name}] ({root_cause})")

        validation_results.append(doc_eval)

    # Compute overall metrics
    precision = (correctly_extracted_fields / total_extracted_fields) if total_extracted_fields > 0 else 0.0
    recall = (correctly_extracted_fields / total_gt_fields) if total_gt_fields > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    summary = {
        "total_cadastral_gt_fields": total_gt_fields,
        "total_extracted_fields": total_extracted_fields,
        "correctly_extracted_fields": correctly_extracted_fields,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "documents": validation_results,
    }

    print("\n" + "=" * 80)
    print("NER EVALUATION ON REAL OCR OUTPUT SUMMARY:")
    print(f"  Total GT Fields evaluated: {total_gt_fields}")
    print(f"  Correctly Extracted: {correctly_extracted_fields}")
    print(f"  Field Precision: {precision * 100:.2f}%")
    print(f"  Field Recall:    {recall * 100:.2f}%")
    print(f"  Field F1-Score:  {f1 * 100:.2f}%")
    print("=" * 80)

    out_dir = Path("tests/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "real_ocr_ner_validation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {out_file}")


if __name__ == "__main__":
    run_ner_validation()
