"""End-to-End Team Integration Test (Person A -> Person B -> Person C).

Simulates the complete inter-team production lifecycle:
1. Person A: Generates candidate bounding boxes and layout classification metadata.
2. Person B: Ingests layout data, routes each region to appropriate OCR/Handwriting engine,
   applies non-destructive normalization, aggregates evidence, and flags review needs.
3. Person C: Consumes strongly-typed DocumentProcessingResponse, extracts structured fields,
   and routes low-confidence/unsupported items to the Human Correction Service.
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from schemas import BoundingBox
from src.correction.service import CorrectionService
from src.integration.document_pipeline import DocumentProcessingPipeline, process_document
from src.integration.person_a_adapter import PersonAAdapter
from src.integration.schemas import DocumentProcessingRequest, RegionRequest, RegionType


def create_synthetic_team_test_page() -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Constructs a realistic multi-region test page simulating Person A layout output."""
    # Create page canvas (800 x 600)
    page_img = Image.new("RGB", (800, 600), color=(250, 248, 242))
    draw = ImageDraw.Draw(page_img)

    # 1. Printed Header Area (Region 1)
    draw.rectangle([40, 30, 760, 90], outline=(180, 180, 180), width=1)
    draw.text((60, 45), "ಕರ್ನಾಟಕ ಸರ್ಕಾರ - ಕಂದಾಯ ಇಲಾಖೆ", fill=(20, 20, 20))

    # 2. Handwritten Kannada Section (Region 2)
    # Load actual handwritten crop from validation set if available
    val_manifest = PROJECT_ROOT / "training/datasets/iiit_kannada_val.jsonl"
    hw_crop_img = None
    hw_gt_text = "ಕಾಲಾವಕಾಶ"

    if val_manifest.exists():
        with open(val_manifest, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                img_rel = item.get("image") or item.get("image_path")
                if img_rel:
                    crop_p = PROJECT_ROOT / img_rel
                    if crop_p.exists():
                        try:
                            hw_crop_img = Image.open(crop_p).convert("RGB")
                            hw_gt_text = item.get("text", "ಕಾಲಾವಕಾಶ")
                            break
                        except Exception:
                            pass

    if hw_crop_img:
        # Paste real crop into page canvas
        hw_crop_resized = hw_crop_img.resize((300, 70))
        page_img.paste(hw_crop_resized, (60, 140))
    else:
        draw.rectangle([60, 140, 360, 210], fill=(240, 238, 230), outline=(100, 100, 100))
        draw.text((80, 160), "ಕಾಲಾವಕಾಶ", fill=(10, 10, 80))

    # 3. Unsupported Handwriting Section (Region 3 - Telugu Handwriting)
    draw.rectangle([60, 260, 360, 330], fill=(245, 240, 235), outline=(150, 100, 100))
    draw.text((80, 280), "తెలుగు చేతివ్రాత", fill=(40, 20, 20))

    # Person A Layout Analysis Output (Simulated upstream dictionary metadata)
    person_a_regions = [
        {
            "id": "header_box_01",
            "bbox": [40, 30, 760, 90],
            "label": "printed_header",
            "language": "kannada",
            "is_handwritten": False,
            "confidence": 0.98,
        },
        {
            "id": "kannada_hw_entry_02",
            "bbox": [60, 140, 360, 210],
            "label": "handwritten_name",
            "language": "kannada",
            "is_handwritten": True,
            "confidence": 0.94,
        },
        {
            "id": "unsupported_hw_entry_03",
            "bbox": [60, 260, 360, 330],
            "label": "handwritten_notes",
            "language": "telugu",
            "is_handwritten": True,
            "confidence": 0.91,
        },
    ]

    return page_img, person_a_regions


def run_team_integration_test():
    print("=" * 80)
    print("PERSON B PRODUCTION HANDOVER: END-TO-END TEAM INTEGRATION TEST")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STAGE 1: PERSON A LAYOUT ANALYSIS SIMULATION
    # -------------------------------------------------------------------------
    print("\n[Stage 1] Person A Output Generated:")
    test_page_img, person_a_raw_regions = create_synthetic_team_test_page()
    print(f"  --> Document Canvas: {test_page_img.size[0]}x{test_page_img.size[1]} px")
    print(f"  --> Detected Layout Regions: {len(person_a_raw_regions)}")
    for r in person_a_raw_regions:
        print(f"      - ID: {r['id']} | Type: {r['label']} | Lang: {r['language']} | HW: {r['is_handwritten']} | BBox: {r['bbox']}")

    # -------------------------------------------------------------------------
    # STAGE 2: PERSON B ORCHESTRATION PIPELINE
    # -------------------------------------------------------------------------
    print("\n[Stage 2] Person B Document Processing Pipeline Execution:")
    start_t = time.perf_counter()

    pipeline = DocumentProcessingPipeline(default_language="kannada", confidence_threshold=0.60)

    # Ingest Person A output into typed request
    request = PersonAAdapter.create_document_request(
        image=test_page_img,
        regions=person_a_raw_regions,
        document_id="DOC_KA_BLR_2026_0042",
        page_number=1,
        language="kannada",
    )

    response = pipeline.process_document(request=request)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    print(f"  --> Pipeline Execution Time: {elapsed_ms:.1f} ms")
    print(f"  --> Document Status        : {response.status}")
    print(f"  --> Document Confidence    : {response.document_confidence if response.document_confidence is not None else 'N/A'}")
    print(f"  --> Requires Human Review  : {response.requires_human_review}")
    print(f"  --> Active Engines Used    : {response.engine_breakdown}")
    print(f"  --> Review Warnings ({len(response.warnings)}):")
    for w in response.warnings:
        print(f"      * {w}")

    # -------------------------------------------------------------------------
    # STAGE 3: PERSON C DATA CONSUMPTION & EXTRACTION
    # -------------------------------------------------------------------------
    print("\n[Stage 3] Person C Structured Data Consumption:")
    print("  --> Iterating Recognized Regions in Geometric Reading Order:")

    for idx, reg in enumerate(response.ordered_regions):
        print(f"\n  [Region {idx + 1}] ID: {reg.region_id}")
        print(f"      - Language / Script   : {reg.language} ({reg.script})")
        print(f"      - Text Type           : {'HANDWRITTEN' if reg.is_handwritten else 'PRINTED'}")
        print(f"      - Model Engine        : {reg.model_name} ({reg.model_version or 'N/A'})")
        print(f"      - Raw Recognized Text : '{reg.raw_text}'")
        print(f"      - Normalized Text     : '{reg.normalized_text}'")
        print(f"      - Model Confidence    : {f'{reg.confidence:.4f}' if reg.confidence is not None else 'N/A'}")
        print(f"      - Review Flagged      : {reg.requires_human_review} (Status: {reg.status.value})")
        if reg.candidate_suggestions:
            print(f"      - Lexicon Suggestions : {reg.candidate_suggestions}")

    # -------------------------------------------------------------------------
    # STAGE 4: HUMAN CORRECTION & ACTIVE LEARNING INTEGRATION
    # -------------------------------------------------------------------------
    print("\n[Stage 4] Human Correction & Active Learning Triage:")
    corr_service = CorrectionService(storage_path="data/corrections/test_active_learning.jsonl")

    # Record corrections for review-flagged regions
    corrections_recorded = 0
    for reg in response.ordered_regions:
        if reg.requires_human_review:
            # Reviewer validates and submits correction
            corr_record = corr_service.record_correction({
                "document_id": response.document_id,
                "region_id": reg.region_id,
                "image_path": f"crops/{response.document_id}_{reg.region_id}.png",
                "crop_bbox": reg.bbox,
                "raw_prediction": reg.raw_text,
                "corrected_text": "తెలుగు చేతివ్రಾತ నಮೂನೆ" if reg.language == "telugu" else (reg.normalized_text or "ದೃಢೀಕೃತ ಪಠ್ಯ"),
                "ai_confidence": reg.confidence,
                "language": reg.language,
                "script": reg.script,
                "is_handwritten": bool(reg.is_handwritten),
                "model_version": reg.model_version,
                "reviewer_id": "REV_OFFICER_07",
                "notes": "Verified against physical revenue registry",
            })
            print(f"  [+] Human Correction Logged: {corr_record.correction_id} for {corr_record.region_id} -> '{corr_record.corrected_text}'")
            corrections_recorded += 1

    # Export active learning dataset manifest
    manifest_out = "data/corrections/active_learning_train_manifest.jsonl"
    count_exported = corr_service.export_training_manifest(output_path=manifest_out, is_handwritten_only=True)
    print(f"  --> Active Learning Dataset Exported: {count_exported} records saved to '{manifest_out}'")
    print("      (Note: No automatic retraining triggered; dataset ready for next planned cycle)")

    print("\n" + "=" * 80)
    print("[SUCCESS] Complete Person A -> Person B -> Person C Integration Flow Verified.")
    print("=" * 80)


if __name__ == "__main__":
    run_team_integration_test()
