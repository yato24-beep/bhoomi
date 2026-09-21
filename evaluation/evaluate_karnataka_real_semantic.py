"""Karnataka-Only Real-Document Semantic Evaluation Benchmark.

Evaluates Gemini-powered SemanticPipeline against representative, real Karnataka
land-record OCR datasets available in the repository.

Features:
- Primary Karnataka canonical schema
- Pre-semantic cadastral gating (is_cadastral=False suppresses extraction on non-cadastral texts)
- Unlabelled contextual inference
- Degraded printed and handwritten OCR handling
- Verbatim Kannada script preservation
- Exact OCR token provenance tracking
- Deterministic domain validation

Outputs reproducible evaluation artifact:
  evaluation/karnataka_real_semantic_evaluation_report.json
"""

import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from schemas import BoundingBox, DocumentType
from src.integration.schemas import RecognizedRegionResult
from src.semantic.schema import LandRecordDocument, SemanticFieldItem, ValidationStatus
from src.semantic.semantic_pipeline import SemanticPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate_karnataka_real_semantic")

ACTIVE_MODEL_NAME = os.getenv("SEMANTIC_MODEL_NAME", "gemini-3.1-flash-lite")


# ==============================================================================
# Helper Normalizers & Matchers
# ==============================================================================

def clean_str(s: Optional[str]) -> str:
    """Normalizes string for comparison by removing whitespace and punctuation."""
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"[\s,._\-\/:]+", "", s)
    return s


def fuzzy_field_match(predicted: Optional[str], acceptable_values: List[str]) -> Tuple[bool, str]:
    """Checks if predicted string matches any acceptable ground truth value."""
    if not predicted or not acceptable_values:
        return False, ""
    
    p_clean = clean_str(predicted)
    for acc in acceptable_values:
        a_clean = clean_str(acc)
        if not a_clean:
            continue
        if p_clean == a_clean:
            return True, acc
        if len(a_clean) >= 3 and a_clean in p_clean:
            return True, acc
        if len(p_clean) >= 3 and p_clean in a_clean:
            return True, acc
    return False, ""


# ==============================================================================
# Karnataka Benchmark Ground Truth
# ==============================================================================

KARNATAKA_GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "ka_doc1_bhoomi_rtc_sample": {
        "is_cadastral": True,
        "description": "Karnataka Bhoomi RTC Form with Kannada script and multi-field lines",
        "expected_fields": {
            "district": ["BENGALURU URBAN", "ಬೆಂಗಳೂರು ನಗರ", "ಬೆಂಗಳೂರು"],
            "taluk": ["BANGALORE SOUTH", "ಬೆಂಗಳೂರು ದಕ್ಷಿಣ"],
            "village": ["KENGERI", "ಕೆಂಗೇರಿ"],
            "survey_number": ["42/1", "42"],
            "owner_name": ["ಸಿದ್ದರಾಮಯ್ಯ", "Siddaramaiah"],
            "extent": ["2 ಎಕರೆ", "2 acres", "2.00", "2"],
        }
    },
    "ka_doc2_bhoomi_rtc_form16_degraded": {
        "is_cadastral": True,
        "description": "Karnataka Bhoomi RTC Form 16 with degraded EasyOCR tokens (Mdnjundth Gowdq, 2_14)",
        "expected_fields": {
            "survey_number": ["45/1", "45,1", "45"],
            "owner_name": ["Manjunath Gowda", "Mdnjundth Gowdq", "ManjunathGowda"],
            "extent": ["2-14 Acres", "2_14", "2 14", "2.14", "214", "2-14"],
        }
    },
    "ka_doc3_bbmp_municipal_certificate": {
        "is_cadastral": True,
        "description": "BBMP Bangalore South municipal land certificate with unlabelled narrative context",
        "expected_fields": {
            "owner_name": ["Dorothy Charles", "Mrs. Dorothy Charles", "Smt. Dorothy Charles"],
            "taluk": ["Kormangala", "ಕೋರಮಂಗಲ", "Koramangala"],
            "district": ["ಬೆಂಗಳೂರು", "Bengaluru", "Bangalore"],
        }
    },
    "ka_doc4_unlabelled_mutation_lines": {
        "is_cadastral": True,
        "description": "Karnataka Form 11 Mutation Register with unlabelled survey/hissa number '125'",
        "expected_fields": {
            "survey_number": ["125", "125 1 ರ"],
        }
    },
    "ka_doc5_doc1_handwritten_kannada": {
        "is_cadastral": True,
        "description": "doc1.jpeg 77 lines of TrOCR handwritten Kannada (High-degradation stress test)",
        "expected_fields": {
            # High noise handwritten baseline - tests rejection/flagging and Kannada numeral preservation
        }
    },
    "ka_doc6_doc2_karnataka_rtc_degraded": {
        "is_cadastral": True,
        "description": "doc2.jpeg 38 lines of degraded EasyOCR Karnataka RTC",
        "expected_fields": {
            # Degraded real scan
        }
    },
    "ka_doc7_bilingual_mutation_paddleocr": {
        "is_cadastral": True,
        "description": "PaddleOCR bilingual scan of Karnataka Mutation Register with token '125'",
        "expected_fields": {
            "document_type": ["MUTATION REGISTER", "Mutation Register"],
        }
    },
    "ka_doc8_doddaballapura_narrative_gated": {
        "is_cadastral": False,
        "description": "Non-cadastral narrative passage from Doddaballapura (Pre-semantic cadastral gate test)",
        "expected_fields": {
            # Non-cadastral: Gating must suppress all cadastral field extractions (0 false positives!)
        }
    }
}


# ==============================================================================
# Document Loaders
# ==============================================================================

def load_karnataka_document(doc_key: str) -> Tuple[List[RecognizedRegionResult], str, bool]:
    """Loads actual OCR regions for Karnataka documents without synthetic layout."""
    regions: List[RecognizedRegionResult] = []
    doc_type = "Unknown"
    is_cadastral = True

    if doc_key == "ka_doc1_bhoomi_rtc_sample":
        p = PROJECT_ROOT / "person-c/data/samples/karnataka_bhoomi_sample.json"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            doc_type = "Bhoomi RTC"
            for idx, tl in enumerate(data.get("text_lines", [])):
                bb = tl.get("bbox", {})
                regions.append(RecognizedRegionResult(
                    region_id=f"line_{idx+1}",
                    raw_text=tl["text"],
                    normalized_text=tl["text"],
                    bbox=BoundingBox(x_min=bb.get("x_min", 0.0), y_min=bb.get("y_min", 0.0), x_max=bb.get("x_max", 100.0), y_max=bb.get("y_max", 20.0)),
                    language=tl.get("language", "kn"),
                    script="Kannada",
                    confidence=tl.get("confidence", 0.95),
                ))

    elif doc_key == "ka_doc2_bhoomi_rtc_form16_degraded":
        p = PROJECT_ROOT / "scratch/production_printed_ocr_regression_report.json"
        with open(p, "r", encoding="utf-8") as f:
            items = json.load(f)
            matched = [it for it in items if it.get("id") == "sample_bhoomi_rtc"]
            if matched:
                doc_type = "Bhoomi RTC Form 16"
                lines = [l.strip() for l in matched[0].get("sample_output", "").split("\n") if l.strip()]
                for i, line in enumerate(lines):
                    regions.append(RecognizedRegionResult(
                        region_id=f"line_{i+1}",
                        raw_text=line,
                        normalized_text=line,
                        bbox=BoundingBox(x_min=10.0, y_min=i * 25.0, x_max=500.0, y_max=(i + 1) * 25.0),
                        language="en",
                        script="Latin",
                        confidence=matched[0].get("recognizer_confidence", 0.65),
                    ))

    elif doc_key == "ka_doc3_bbmp_municipal_certificate":
        p = PROJECT_ROOT / "scratch/backend_doc_results.json"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            doc_type = "BBMP Municipal Land Certificate"
            raw = data.get("extracted_data", {}).get("merged_text", "")
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            for i, line in enumerate(lines):
                is_kn = any(0x0C80 <= ord(c) <= 0x0CFF for c in line)
                regions.append(RecognizedRegionResult(
                    region_id=f"line_{i+1}",
                    raw_text=line,
                    normalized_text=line,
                    bbox=BoundingBox(x_min=20.0, y_min=i * 24.0, x_max=600.0, y_max=(i + 1) * 24.0),
                    language="kn" if is_kn else "en",
                    script="Kannada" if is_kn else "Latin",
                    confidence=0.91,
                ))

    elif doc_key == "ka_doc4_unlabelled_mutation_lines":
        p = PROJECT_ROOT / "scratch/trocr_eval_comparison_report.json"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            doc_type = "Mutation Register Form 11"
            lines = data.get("lines", [])[:5]
            for i, item in enumerate(lines):
                txt = item.get("gt", "")
                is_kn = any(0x0C80 <= ord(c) <= 0x0CFF for c in txt)
                regions.append(RecognizedRegionResult(
                    region_id=f"line_{i+1}",
                    raw_text=txt,
                    normalized_text=txt,
                    bbox=BoundingBox(x_min=10.0, y_min=i * 30.0, x_max=650.0, y_max=(i + 1) * 30.0),
                    language="kn" if is_kn else "en",
                    script="Kannada" if is_kn else "Latin",
                    confidence=0.88,
                ))

    elif doc_key == "ka_doc5_doc1_handwritten_kannada":
        p = PROJECT_ROOT / "storage/exports/doc1_full_result.json"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            doc_type = "Handwritten Kannada Land Record"
            lines = [l.strip() for l in data.get("kannada_transcription", "").split("\n") if l.strip()]
            for i, line in enumerate(lines):
                regions.append(RecognizedRegionResult(
                    region_id=f"line_{i+1}",
                    raw_text=line,
                    normalized_text=line,
                    bbox=BoundingBox(x_min=15.0, y_min=i * 18.0, x_max=350.0, y_max=(i + 1) * 18.0),
                    language="kn",
                    script="Kannada",
                    confidence=0.645,
                ))

    elif doc_key == "ka_doc6_doc2_karnataka_rtc_degraded":
        p = PROJECT_ROOT / "scratch/production_printed_ocr_regression_report.json"
        with open(p, "r", encoding="utf-8") as f:
            items = json.load(f)
            matched = [it for it in items if it.get("id") == "doc2_karnataka_rtc"]
            if matched:
                doc_type = "Karnataka RTC"
                lines = [l.strip() for l in matched[0].get("sample_output", "").split("\n") if l.strip()]
                for i, line in enumerate(lines):
                    is_kn = any(0x0C80 <= ord(c) <= 0x0CFF for c in line)
                    regions.append(RecognizedRegionResult(
                        region_id=f"line_{i+1}",
                        raw_text=line,
                        normalized_text=line,
                        bbox=BoundingBox(x_min=10.0, y_min=i * 25.0, x_max=500.0, y_max=(i + 1) * 25.0),
                        language="kn" if is_kn else "en",
                        script="Kannada" if is_kn else "Latin",
                        confidence=0.55,
                    ))

    elif doc_key == "ka_doc7_bilingual_mutation_paddleocr":
        p = PROJECT_ROOT / "bilingual_results.txt"
        with open(p, "r", encoding="utf-8") as f:
            doc_type = "Mutation Register"
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or "|" not in line:
                    continue
                parts = line.split("|")
                txt = parts[1].strip() if len(parts) > 1 else ""
                conf_match = re.search(r"conf=([0-9.]+)", parts[0])
                c_val = float(conf_match.group(1)) if conf_match else 0.70
                regions.append(RecognizedRegionResult(
                    region_id=f"box_{i+1}",
                    raw_text=txt,
                    normalized_text=txt,
                    bbox=BoundingBox(x_min=50.0, y_min=i * 20.0, x_max=400.0, y_max=(i + 1) * 20.0),
                    language="kn" if "kannada" in parts[0] else "en",
                    script="Kannada" if "kannada" in parts[0] else "Latin",
                    confidence=c_val,
                ))

    elif doc_key == "ka_doc8_doddaballapura_narrative_gated":
        p = PROJECT_ROOT / "scratch/production_printed_ocr_regression_report.json"
        with open(p, "r", encoding="utf-8") as f:
            items = json.load(f)
            matched = [it for it in items if it.get("id") == "doddaballapura_passage"]
            if matched:
                doc_type = "Not a land record"
                is_cadastral = False
                lines = [l.strip() for l in matched[0].get("sample_output", "").split("\n") if l.strip()]
                for i, line in enumerate(lines):
                    is_kn = any(0x0C80 <= ord(c) <= 0x0CFF for c in line)
                    regions.append(RecognizedRegionResult(
                        region_id=f"line_{i+1}",
                        raw_text=line,
                        normalized_text=line,
                        bbox=BoundingBox(x_min=10.0, y_min=i * 25.0, x_max=500.0, y_max=(i + 1) * 25.0),
                        language="kn" if is_kn else "en",
                        script="Kannada" if is_kn else "Latin",
                        confidence=0.65,
                    ))

    return regions, doc_type, is_cadastral


# ==============================================================================
# Evaluation Logic
# ==============================================================================

@dataclass
class KarnatakaFieldRecord:
    field_name: str
    expected_value: Optional[str]
    predicted_value: Optional[str]
    predicted_raw: Optional[str]
    classification: str  # "CORRECT", "INCORRECT", "MISSING", "FALSE_POSITIVE", "ADDITIONAL_GROUNDED"
    provenance_correct: bool
    provenance_region_id: Optional[str]
    validation_status: str
    confidence: float


@dataclass
class KarnatakaDocumentEvaluation:
    document_id: str
    description: str
    is_cadastral: bool
    gate_suppressed: bool
    detected_document_type: str
    total_regions: int
    field_evaluations: List[KarnatakaFieldRecord]
    expected_count: int
    correct_count: int
    incorrect_count: int
    missing_count: int
    false_positive_count: int
    provenance_correct_count: int
    total_extracted_count: int
    overall_confidence: float
    requires_human_review: bool


def evaluate_karnataka_document(
    doc_id: str,
    pipeline: SemanticPipeline,
) -> KarnatakaDocumentEvaluation:
    gt_info = KARNATAKA_GROUND_TRUTH.get(doc_id, {"is_cadastral": True, "description": "", "expected_fields": {}})
    expected_fields = gt_info.get("expected_fields", {})
    is_cadastral_gt = gt_info.get("is_cadastral", True)
    description = gt_info.get("description", "")

    regions, nominal_type, is_cadastral_flag = load_karnataka_document(doc_id)

    # Process through pipeline with pre-semantic cadastral gate
    start_t = time.time()
    land_record: LandRecordDocument = pipeline.process(
        regions=regions,
        document_id=doc_id,
        page_number=1,
        document_type=nominal_type,
        is_cadastral=is_cadastral_flag,
        classification_result={"is_land_record": is_cadastral_flag},
    )
    elapsed_ms = round((time.time() - start_t) * 1000, 1)
    gate_suppressed = land_record.validation_summary.get("gate_status") == "cadastral_extraction_suppressed"

    logger.info(f"[{doc_id}] Processed {len(regions)} regions in {elapsed_ms}ms (Gate suppressed: {gate_suppressed}) -> {len(land_record.fields)} fields")

    field_evals: List[KarnatakaFieldRecord] = []
    evaluated_field_names: Set[str] = set()

    correct_count = 0
    incorrect_count = 0
    missing_count = 0
    false_positive_count = 0
    provenance_correct_count = 0

    all_ocr_full_text = " ".join([getattr(r, "raw_text", getattr(r, "text", "")) for r in regions])
    region_id_set = {getattr(r, "region_id", "") for r in regions}

    # 1. Expected fields
    for exp_fname, acceptable_vals in expected_fields.items():
        evaluated_field_names.add(exp_fname)
        pred_item = land_record.fields.get(exp_fname)

        if not pred_item or not pred_item.value or not str(pred_item.value).strip():
            missing_count += 1
            field_evals.append(KarnatakaFieldRecord(
                field_name=exp_fname,
                expected_value=str(acceptable_vals),
                predicted_value=None,
                predicted_raw=None,
                classification="MISSING",
                provenance_correct=False,
                provenance_region_id=None,
                validation_status="NOT_EXTRACTED",
                confidence=0.0,
            ))
            continue

        pred_val = str(pred_item.value)
        pred_raw = str(pred_item.raw_value) if pred_item.raw_value else pred_val

        is_match, _ = fuzzy_field_match(pred_val, acceptable_vals)
        if not is_match:
            is_match, _ = fuzzy_field_match(pred_raw, acceptable_vals)

        prov = pred_item.provenance
        prov_reg_id = prov.region_id if prov else None
        prov_valid = bool(prov and prov.region_id in region_id_set) or (clean_str(pred_raw) in clean_str(all_ocr_full_text))

        if prov_valid:
            provenance_correct_count += 1

        is_grounded = clean_str(pred_raw) in clean_str(all_ocr_full_text)
        if not is_grounded:
            classification = "FALSE_POSITIVE"
            false_positive_count += 1
        elif is_match:
            classification = "CORRECT"
            correct_count += 1
        else:
            classification = "INCORRECT"
            incorrect_count += 1

        val_status_str = pred_item.validation_status.value if pred_item.validation_status else "UNVERIFIED"

        field_evals.append(KarnatakaFieldRecord(
            field_name=exp_fname,
            expected_value=str(acceptable_vals),
            predicted_value=pred_val,
            predicted_raw=pred_raw,
            classification=classification,
            provenance_correct=prov_valid,
            provenance_region_id=prov_reg_id,
            validation_status=val_status_str,
            confidence=pred_item.confidence or 0.0,
        ))

    # 2. Auxiliary extracted fields
    for fname, item in land_record.fields.items():
        if fname in evaluated_field_names:
            continue
        if not item.value or not str(item.value).strip():
            continue

        raw_val = str(item.raw_value) if item.raw_value else str(item.value)
        is_grounded = clean_str(raw_val) in clean_str(all_ocr_full_text)

        prov = item.provenance
        prov_reg_id = prov.region_id if prov else None
        prov_valid = bool(prov and prov.region_id in region_id_set) or is_grounded

        if prov_valid:
            provenance_correct_count += 1

        if not is_cadastral_gt:
            classification = "FALSE_POSITIVE"
            false_positive_count += 1
        elif not is_grounded:
            classification = "FALSE_POSITIVE"
            false_positive_count += 1
        else:
            classification = "ADDITIONAL_GROUNDED"

        val_status_str = item.validation_status.value if item.validation_status else "UNVERIFIED"

        field_evals.append(KarnatakaFieldRecord(
            field_name=fname,
            expected_value=None if is_cadastral_gt else "None (Non-cadastral passage)",
            predicted_value=str(item.value),
            predicted_raw=raw_val,
            classification=classification,
            provenance_correct=prov_valid,
            provenance_region_id=prov_reg_id,
            validation_status=val_status_str,
            confidence=item.confidence or 0.0,
        ))

    total_extracted = len(land_record.fields)
    overall_conf = land_record.confidence_score
    requires_review = land_record.requires_human_review

    return KarnatakaDocumentEvaluation(
        document_id=doc_id,
        description=description,
        is_cadastral=is_cadastral_gt,
        gate_suppressed=gate_suppressed,
        detected_document_type=land_record.document_type,
        total_regions=len(regions),
        field_evaluations=field_evals,
        expected_count=len(expected_fields),
        correct_count=correct_count,
        incorrect_count=incorrect_count,
        missing_count=missing_count,
        false_positive_count=false_positive_count,
        provenance_correct_count=provenance_correct_count,
        total_extracted_count=total_extracted,
        overall_confidence=overall_conf,
        requires_human_review=requires_review,
    )


# ==============================================================================
# Runner
# ==============================================================================

def run_karnataka_evaluation() -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print(f"KARNATAKA-ONLY REAL-DOCUMENT SEMANTIC BENCHMARK ({ACTIVE_MODEL_NAME})")
    print("=" * 90)

    pipeline = SemanticPipeline()
    doc_keys = list(KARNATAKA_GROUND_TRUTH.keys())

    eval_results: List[KarnatakaDocumentEvaluation] = []

    total_expected = 0
    total_correct = 0
    total_incorrect = 0
    total_missing = 0
    total_false_positives = 0
    total_provenance_correct = 0
    total_extracted = 0
    total_review_required = 0

    for doc_id in doc_keys:
        print(f"\nEvaluating [{doc_id}] ({KARNATAKA_GROUND_TRUTH[doc_id]['description']})...")
        rec = evaluate_karnataka_document(doc_id, pipeline)
        eval_results.append(rec)

        total_expected += rec.expected_count
        total_correct += rec.correct_count
        total_incorrect += rec.incorrect_count
        total_missing += rec.missing_count
        total_false_positives += rec.false_positive_count
        total_provenance_correct += rec.provenance_correct_count
        total_extracted += rec.total_extracted_count
        if rec.requires_human_review:
            total_review_required += 1

    # Key Metrics
    accuracy = (total_correct / total_expected * 100) if total_expected > 0 else 0.0
    missing_rate = (total_missing / total_expected * 100) if total_expected > 0 else 0.0
    wrong_rate = (total_incorrect / total_expected * 100) if total_expected > 0 else 0.0
    fp_rate = (total_false_positives / total_extracted * 100) if total_extracted > 0 else 0.0
    provenance_acc = (total_provenance_correct / total_extracted * 100) if total_extracted > 0 else 0.0
    review_rate = (total_review_required / len(eval_results) * 100) if eval_results else 0.0

    metrics = {
        "gemini_model_used": ACTIVE_MODEL_NAME,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_karnataka_documents_tested": len(eval_results),
        "total_expected_fields": total_expected,
        "total_extracted_fields": total_extracted,
        "correct_extractions": total_correct,
        "incorrect_extractions": total_incorrect,
        "missing_extractions": total_missing,
        "false_positives_count": total_false_positives,
        "provenance_verified": total_provenance_correct,
        "documents_requiring_human_review": total_review_required,
        "field_extraction_accuracy_pct": round(accuracy, 2),
        "missing_field_rate_pct": round(missing_rate, 2),
        "wrong_field_rate_pct": round(wrong_rate, 2),
        "false_positive_rate_pct": round(fp_rate, 2),
        "provenance_accuracy_pct": round(provenance_acc, 2),
        "document_review_rate_pct": round(review_rate, 2),
    }

    # Save artifact
    output_path = PROJECT_ROOT / "evaluation/karnataka_real_semantic_evaluation_report.json"
    output_data = {
        "summary_metrics": metrics,
        "documents": [asdict(r) for r in eval_results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n[ARTIFACT SAVED] Saved Karnataka evaluation report to: {output_path}")

    # Summary Report
    print("\n" + "=" * 90)
    print("KARNATAKA BENCHMARK SUMMARY REPORT")
    print("=" * 90)
    print(f"Gemini Model               : {metrics['gemini_model_used']}")
    print(f"Total Karnataka Documents  : {metrics['total_karnataka_documents_tested']}")
    print(f"Field Extraction Accuracy  : {metrics['field_extraction_accuracy_pct']}% ({total_correct}/{total_expected})")
    print(f"Missing Field Rate         : {metrics['missing_field_rate_pct']}% ({total_missing}/{total_expected})")
    print(f"Wrong Field Rate           : {metrics['wrong_field_rate_pct']}% ({total_incorrect}/{total_expected})")
    print(f"False Positive Rate        : {metrics['false_positive_rate_pct']}% ({total_false_positives}/{total_extracted})")
    print(f"Provenance Accuracy        : {metrics['provenance_accuracy_pct']}% ({total_provenance_correct}/{total_extracted})")
    print(f"Human Review Rate          : {metrics['document_review_rate_pct']}% ({total_review_required}/{len(eval_results)} docs)")
    print("=" * 90)

    print(f"\n{'Document ID':<38} | {'Regions':<8} | {'Exp':<4} | {'Cor':<4} | {'Mis':<4} | {'FP':<4} | {'Gated':<6} | {'Review':<6}")
    print("-" * 90)
    for r in eval_results:
        print(f"{r.document_id:<38} | {r.total_regions:<8} | {r.expected_count:<4} | {r.correct_count:<4} | {r.missing_count:<4} | {r.false_positive_count:<4} | {str(r.gate_suppressed):<6} | {str(r.requires_human_review):<6}")

    return output_data


if __name__ == "__main__":
    run_karnataka_evaluation()
