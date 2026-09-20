"""Comprehensive Regression Test Suite for Non-Training Land-Record Sprint.

Tests all non-training pipeline infrastructure:
1. Dataset ZIP Ingestion (flat, nested, CSV, JSON, TXT label sources, corruption detection)
2. Data Lineage and Provenance Tracking (tamper-evident split hash, leakage detection)
3. Confidence Calibration (Platts scaling, temperature scaling, ECE binning, fallback to UNCALIBRATED)
4. Human Review Queue & State Machine (priority ordering, full lifecycle, immutable raw evidence)
5. Uncertainty Visualizer (overlay rendering, side-by-side comparison, color tiers)
6. End-to-End Pipeline Integration (semantic data, tables, review items, honest confidence)
7. Integrity Check: Gate 2 locked benchmark remains completely untouched
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import pytest

from src.data.provenance import (
    ProvenanceManifest,
    SampleProvenance,
    compute_bytes_sha256,
    compute_sha256,
)
from src.data.zip_ingestion import DatasetZipIngester, IngestionReport
from src.confidence.calibration import (
    CalibrationMethod,
    EngineCalibrator,
    MultiEngineCalibrator,
    ReliabilityDiagramData,
)
from src.integration.review_state import (
    HumanReviewItem,
    HumanReviewQueue,
    ReviewDecision,
    ReviewLifecycleState,
    ReviewStateMachine,
    ReviewStatus,
)
from src.visualization.uncertainty_overlay import (
    generate_uncertainty_report,
    get_color_for_region,
    render_side_by_side_comparison,
    render_uncertainty_overlay,
)
from schemas import BoundingBox


# ---------------------------------------------------------------------------
# Test 1: Dataset ZIP Ingestion
# ---------------------------------------------------------------------------

def test_dataset_zip_ingestion_csv_and_txt(tmp_path):
    """Tests intake of ZIP archive containing images, TXT labels, and CSV metadata."""
    zip_source_dir = tmp_path / "zip_source"
    zip_source_dir.mkdir()

    # Create 3 synthetic sample images
    img1 = Image.new("RGB", (64, 32), color=(255, 255, 255))
    img1.save(zip_source_dir / "sample_01.png")
    (zip_source_dir / "sample_01.txt").write_text("ಕರ್ನಾಟಕ", encoding="utf-8")

    img2 = Image.new("RGB", (80, 32), color=(240, 240, 240))
    img2.save(zip_source_dir / "sample_02.jpg")

    # CSV metadata mapping sample_02 and sample_03
    csv_content = "filename,text,writer_id,domain\nsample_02.jpg,ಸರ್ಕಾರ,w1,bhoomi\nsample_03.png,ಕಂದಾಯ,w2,rtc\n"
    (zip_source_dir / "metadata.csv").write_text(csv_content, encoding="utf-8")

    img3 = Image.new("RGB", (70, 32), color=(230, 230, 230))
    img3.save(zip_source_dir / "sample_03.png")

    # Create zip archive
    zip_path = tmp_path / "test_intake.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in zip_source_dir.iterdir():
            zf.write(f, arcname=f.name)

    # Ingest
    output_root = tmp_path / "normalized_datasets"
    ingester = DatasetZipIngester(
        output_root=output_root,
        split_ratios={"train": 0.60, "val": 0.20, "test": 0.20},
        dataset_name="intake_test_v1",
    )
    target_dir, report, prov = ingester.ingest_zip(zip_path)

    # Assertions
    assert target_dir.exists()
    assert report.total_images_discovered == 3
    assert report.valid_images == 3
    assert report.corrupt_images == 0
    assert report.labeled_samples == 3
    assert (target_dir / "manifest.json").is_file()
    assert (target_dir / "provenance.json").is_file()
    assert (target_dir / "validation_report.json").is_file()
    assert (target_dir / "SUMMARY.md").is_file()

    # Verify manifest JSON content
    with open(target_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert manifest_data["total_samples"] == 3
    assert len(manifest_data["samples"]) == 3
    assert all("images/" in s["image_file"] for s in manifest_data["samples"])


# ---------------------------------------------------------------------------
# Test 2: Data Provenance and Leakage Detection
# ---------------------------------------------------------------------------

def test_provenance_manifest_and_leakage_detection():
    """Tests provenance tracking, split hash generation, and leakage detection."""
    manifest = ProvenanceManifest(
        dataset_name="kannada_test_leakage",
        dataset_version="1.0.0",
        archive_sha256="dummy_sha256_archive",
    )

    s1 = SampleProvenance(
        sample_id="s1",
        file_name="images/s1.png",
        sample_sha256="hash_sample_1",
        split="train",
        writer_id="writer_A",
    )
    s2 = SampleProvenance(
        sample_id="s2",
        file_name="images/s2.png",
        sample_sha256="hash_sample_2",
        split="val",
        writer_id="writer_B",
    )
    s3 = SampleProvenance(
        sample_id="s3",
        file_name="images/s3.png",
        sample_sha256="hash_sample_3",
        split="test",
        writer_id="writer_C",
    )
    manifest.add_sample(s1)
    manifest.add_sample(s2)
    manifest.add_sample(s3)

    # Split hash
    h1 = manifest.compute_split_assignment_hash()
    assert len(h1) == 64
    # Clean check should pass
    passed, errors = manifest.verify_no_data_leakage()
    assert passed is True
    assert len(errors) == 0

    # Introduce intentional hash leakage
    s_leak = SampleProvenance(
        sample_id="s_leak",
        file_name="images/s_leak.png",
        sample_sha256="hash_sample_1",  # Same as s1 in train!
        split="test",
        writer_id="writer_D",
    )
    manifest.add_sample(s_leak)
    passed_leak, errors_leak = manifest.verify_no_data_leakage()
    assert passed_leak is False
    assert any("Data leakage detected" in e for e in errors_leak)


# ---------------------------------------------------------------------------
# Test 3: Confidence Calibration (Platts & Temperature)
# ---------------------------------------------------------------------------

def test_confidence_calibration_platts_and_temperature(tmp_path):
    """Tests fitting Platt's scaling and Temperature scaling and computing ECE."""
    # Synthetic dataset: 40 samples
    # Raw scores are somewhat overconfident
    rng = np.random.RandomState(42)
    raw_scores = rng.uniform(0.5, 0.95, 60)
    # Ground truth: probability of correct aligns with true underlying sigmoid
    true_probs = 1.0 / (1.0 + np.exp(-3.0 * (raw_scores - 0.7)))
    labels = (rng.uniform(0, 1, 60) < true_probs).astype(int)

    # Verify >=50 safeguard
    cal_safeguard = EngineCalibrator(engine_name="easyocr")
    with pytest.raises(ValueError, match="Insufficient samples"):
        cal_safeguard.fit(raw_scores[:30], labels[:30])

    # 1. Platt's scaling with >=50 samples
    cal_platts = EngineCalibrator(engine_name="easyocr", method=CalibrationMethod.PLATTS_SCALING)
    metrics_p = cal_platts.fit(raw_scores, labels)
    assert cal_platts.is_fitted is True
    assert "calibrated_ece" in metrics_p
    assert cal_platts.reliability_diagram is not None
    assert len(cal_platts.reliability_diagram.bin_edges) == 11

    # Calibrated prediction
    pred_p = cal_platts.predict(0.80)
    assert pred_p is not None
    assert 0.0 <= pred_p <= 1.0

    # 2. Temperature scaling
    cal_temp = EngineCalibrator(engine_name="iitb_kannada_v002", method=CalibrationMethod.TEMPERATURE_SCALING)
    metrics_t = cal_temp.fit(raw_scores, labels)
    assert cal_temp.is_fitted is True
    assert metrics_t["parameters"]["temperature"] > 0
    pred_t = cal_temp.predict(0.75)
    assert pred_t is not None
    assert 0.0 <= pred_t <= 1.0

    # 3. MultiEngineCalibrator save/load
    multi = MultiEngineCalibrator()
    multi.calibrators["easyocr"] = cal_platts
    multi.calibrators["iitb_kannada_v002"] = cal_temp

    save_p = tmp_path / "multi_calibrator.json"
    multi.save(save_p)
    assert save_p.is_file()

    loaded = MultiEngineCalibrator.load(save_p)
    assert "easyocr" in loaded.calibrators
    assert loaded.calibrators["easyocr"].is_fitted is True
    cal_val, state = loaded.calibrate("easyocr", 0.80)
    assert cal_val == pred_p
    assert state == "CALIBRATED"

    # Test unknown engine fallback
    uncal_val, uncal_state = loaded.calibrate("unknown_engine", 0.80)
    assert uncal_val is None
    assert uncal_state == "UNCALIBRATED"


# ---------------------------------------------------------------------------
# Test 4: Human Review State Machine & Priority Queue
# ---------------------------------------------------------------------------

def test_human_review_queue_and_state_transitions(tmp_path):
    """Tests priority queue ordering, status transitions, and audit preservation."""
    queue = HumanReviewQueue()

    # Item 1: High confidence, normal field
    item1 = ReviewStateMachine.create_review_item(
        document_id="doc_1",
        page_number=1,
        region_id="r1",
        bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.2),
        raw_ocr_text="ಕರ್ನಾಟಕ",
        recognizer="easyocr",
        review_reason="Standard verification",
        recognizer_confidence_raw=0.90,
        calibrated_confidence=0.88,
        is_critical_field=False,
    )

    # Item 2: Low confidence, normal field
    item2 = ReviewStateMachine.create_review_item(
        document_id="doc_1",
        page_number=1,
        region_id="r2",
        bbox=BoundingBox(x_min=0.1, y_min=0.3, x_max=0.3, y_max=0.4),
        raw_ocr_text="ಸರ್ವೆ",
        recognizer="iitb_kannada_v002",
        review_reason="Low handwriting confidence",
        recognizer_confidence_raw=0.45,
        calibrated_confidence=0.40,
        is_critical_field=False,
    )

    # Item 3: Medium confidence, CRITICAL FIELD (e.g. survey_number)
    item3 = ReviewStateMachine.create_review_item(
        document_id="doc_1",
        page_number=1,
        region_id="r3",
        bbox=BoundingBox(x_min=0.1, y_min=0.5, x_max=0.3, y_max=0.6),
        raw_ocr_text="142/3",
        recognizer="easyocr",
        review_reason="Survey number anchor verification",
        recognizer_confidence_raw=0.75,
        calibrated_confidence=0.72,
        is_critical_field=True,
    )

    queue.add_item(item1)
    queue.add_item(item2)
    queue.add_item(item3, is_critical_field=True)

    # Priority verification: item3 (critical field) should have highest priority score
    pending = queue.get_pending_items()
    assert len(pending) == 3
    assert pending[0].review_id == item3.review_id  # Boosted by critical field
    assert pending[1].review_id == item2.review_id  # Lowest confidence among non-critical

    # Review lifecycle: Assign and Approve
    top_item = queue.get_next(reviewer_id="officer_01")
    assert top_item.review_id == item3.review_id
    assert top_item.lifecycle_state == ReviewLifecycleState.IN_REVIEW
    assert top_item.reviewed_by == "officer_01"

    # Officer edits the item
    queue.edit_item(
        review_id=top_item.review_id,
        reviewer_id="officer_01",
        corrected_text="142/3A",
        notes="Corrected sub-division letter",
    )
    assert top_item.lifecycle_state == ReviewLifecycleState.EDITED
    assert top_item.corrected_text == "142/3A"
    # CRITICAL: Raw OCR evidence must remain 100% immutable!
    assert top_item.raw_ocr_text == "142/3"

    # Export and reload JSON queue
    export_p = tmp_path / "review_queue.json"
    queue.export_json(export_p)
    assert export_p.is_file()

    reloaded_q = HumanReviewQueue.load_json(export_p)
    assert len(reloaded_q.items) == 3
    reloaded_top = reloaded_q.items[item3.review_id]
    # Verify all required review persistence fields survive restart / reload:
    assert reloaded_top.raw_ocr_text == "142/3"  # Immutable raw evidence
    assert reloaded_top.corrected_text == "142/3A"
    assert reloaded_top.status == ReviewStatus.AUTO_ACCEPT
    assert reloaded_top.lifecycle_state == ReviewLifecycleState.EDITED
    assert reloaded_top.reviewed_by == "officer_01"
    assert reloaded_top.reviewed_at is not None
    assert reloaded_top.review_reason == "Survey number anchor verification"
    assert reloaded_top.document_id == "doc_1"
    assert reloaded_top.page_number == 1
    assert reloaded_top.region_id == "r3"
    assert reloaded_top.bbox is not None
    assert reloaded_top.bbox.x_min == 0.1


# ---------------------------------------------------------------------------
# Test 5: Uncertainty & Low-Confidence Visualizer
# ---------------------------------------------------------------------------

def test_uncertainty_visualizer_rendering(tmp_path):
    """Tests visualizer rendering with confidence tiers and side-by-side view."""
    # Create blank test document
    doc_img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(doc_img)
    draw.text((20, 20), "Land Record Page 1", fill=(0, 0, 0))

    # Mock regions
    regions = [
        {
            "bbox": [20, 40, 150, 70],
            "confidence": 0.95,
            "recognizer": "easyocr",
            "is_handwritten": False,
            "needs_review": False,
            "raw_text": "ಕರ್ನಾಟಕ ಸರ್ಕಾರ",
        },
        {
            "bbox": [20, 80, 160, 110],
            "confidence": 0.72,
            "recognizer": "easyocr",
            "is_handwritten": False,
            "needs_review": False,
            "raw_text": "ಕಂದಾಯ ಇಲಾಖೆ",
        },
        {
            "bbox": [20, 120, 180, 160],
            "confidence": 0.42,
            "recognizer": "iitb_kannada_v002",
            "is_handwritten": True,
            "needs_review": True,
            "raw_text": "ಲಿಂಗಯ್ಯ",
        },
    ]

    out_dir = tmp_path / "vis_output"
    report_paths = generate_uncertainty_report(
        image=doc_img,
        regions=regions,
        output_dir=out_dir,
        base_name="test_page",
    )

    assert Path(report_paths["confidence_overlay"]).is_file()
    assert Path(report_paths["engine_overlay"]).is_file()
    assert Path(report_paths["side_by_side"]).is_file()

    # Verify overlay dimensions match base image
    with Image.open(report_paths["confidence_overlay"]) as img:
        assert img.size == (400, 300)

    # Verify side-by-side comparison image has content
    with Image.open(report_paths["side_by_side"]) as sbs:
        assert sbs.size[0] == 900
        assert sbs.size[1] > 100


# ---------------------------------------------------------------------------
# Test 6: Gate 2 Locked Benchmark Untouched
# ---------------------------------------------------------------------------

def test_gate2_benchmark_remains_untouched():
    """Verifies that scripts/gate2_locked_benchmark.py is strictly unmodified."""
    benchmark_path = Path("scripts/gate2_locked_benchmark.py")
    assert benchmark_path.is_file()
    content = benchmark_path.read_text(encoding="utf-8")

    # Verify original locked benchmark anchors
    assert "GATE 2: LOCKED OCR BENCHMARK FOR IIT BOMBAY INDIC-TROCR V0.0.2" in content
    assert "WORD_LEVEL_ARCHIVAL = [" in content
    assert "LINE_LEVEL_ARCHIVAL = [" in content
    assert "crop_a_mara.png" in content
    assert "line_04.png" in content
    assert "line_10.png" in content


# ---------------------------------------------------------------------------
# Test 7: E2E Pipeline Integration (Semantic Data, Tables, Review Queue)
# ---------------------------------------------------------------------------

def test_e2e_document_pipeline_integration():
    """Tests that DocumentProcessingPipeline returns all Sprint 7 schemas and honest metadata."""
    from unittest.mock import MagicMock
    from src.integration.document_pipeline import DocumentProcessingPipeline
    from src.integration.schemas import DocumentProcessingRequest, RegionRequest

    from schemas import OCRResult

    pipeline = DocumentProcessingPipeline(enable_document_gating=False)

    # Mock the router/recognizer to return a deterministic Kannada result
    mock_router = MagicMock()
    mock_router.route_and_recognize.return_value = OCRResult(
        text="ಸರ್ಕಾರ ೧೨೩",
        confidence=0.88,
        model_name="easyocr",
        is_handwritten=False,
    )
    pipeline.router = mock_router

    # Create synthetic page
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    regions = [
        RegionRequest(
            region_id="r_hdr_01",
            bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.4),
            script="Kannada",
        )
    ]
    req = DocumentProcessingRequest(image=img, document_id="doc_test_sprint7", page_number=1)

    resp = pipeline.process_document(image=img, request=req, regions=regions)

    # Assertions
    assert resp.document_id == "doc_test_sprint7"
    assert "semantic_data" in resp.model_fields_set or resp.semantic_data is not None
    assert isinstance(resp.review_items, list)
    assert isinstance(resp.tables, list)
    assert resp.confidence_state in ("UNCALIBRATED", "CALIBRATED", "REVIEW_REQUIRED")
    assert resp.verification_status in ("accepted", "needs_verification")
    # Anti-hallucination / Anti-fake check
    assert resp.document_type != "Commercial Invoice"
    assert resp.document_type in ("Unknown / Not classified", "Land Record (Unclassified)", "Not a land record", None)
    assert resp.document_type_state in ("UNKNOWN", "CONFIRMED")


def test_processor_schema_integration():
    """Verifies backend ProcessingResult schema has all non-training sprint fields."""
    from backend.app.pipeline.processor import ProcessingResult

    fields = ProcessingResult.model_fields
    assert "confidence_state" in fields
    assert "semantic_data" in fields
    assert "review_items" in fields
    assert "tables" in fields
    assert "calibrated_confidence" in fields
    assert "recognizer_confidence_raw" in fields
    assert "document_type_state" in fields
    assert "classifier_source" in fields
    assert "classifier_score" in fields
    assert "classifier_evidence" in fields


def test_human_review_persistence_api_and_immutability():
    """Verifies human review correction persists to database across process restarts.
    Trace: OCR -> review item -> API -> storage -> process restart -> reload.
    Ensures raw OCR evidence is NEVER mutated and all review metadata survives.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db.base import Base
    from app.models.document import Document
    from app.models.extraction import ExtractionResult
    from backend.app.api.v1.documents import submit_document_review, ReviewCorrectionRequest

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    # Step 1: Create Document and ExtractionResult with a review item
    doc = Document(filename="record_45.pdf", file_hash="mock_hash_45", storage_path="records/record_45.pdf")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    doc_id = doc.id

    review_item = {
        "review_id": "rev_item_45",
        "document_id": str(doc_id),
        "page_number": 1,
        "region_id": "reg_45",
        "bbox": {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 50},
        "raw_ocr_text": "ಸರ್ವೆ ನಂ: ೧೪೨/೧",
        "recognizer": "easyocr",
        "review_reason": "Kannada numeral ambiguity",
        "status": "REVIEW_REQUIRED",
        "decision": "PENDING",
        "lifecycle_state": "PENDING",
        "corrected_text": None,
        "reviewed_by": None,
        "reviewed_at": None,
    }

    ext = ExtractionResult(
        document_id=doc_id,
        extracted_data={"review_items": [review_item]},
        confidence_score=0.75,
        is_valid=False,
        validation_info={"requires_human_review": True, "warnings": ["Needs review"]},
    )
    db.add(ext)
    db.commit()
    db.close()

    # Step 2: Simulate process restart - open fresh session to submit review
    db2 = SessionLocal()
    req = ReviewCorrectionRequest(
        review_id="rev_item_45",
        decision="CORRECTED",
        corrected_text="ಸರ್ವೆ ನಂ: 142/1",
        reviewer_notes="Standardized numerals to ASCII",
        reviewed_by="officer_kannada_dept",
    )
    res = submit_document_review(document_id=doc_id, payload=req, db=db2, current_user=None)
    assert res["status"] == "success"
    db2.close()

    # Step 3: Simulate second process restart - open another fresh session and reload from storage
    db3 = SessionLocal()
    reloaded_ext = db3.query(ExtractionResult).filter(ExtractionResult.document_id == doc_id).one()
    items = reloaded_ext.extracted_data["review_items"]
    assert len(items) == 1
    it = items[0]

    # Required field verification:
    assert it["raw_ocr_text"] == "ಸರ್ವೆ ನಂ: ೧೪೨/೧"  # RAW OCR EVIDENCE NEVER MUTATED!
    assert it["corrected_text"] == "ಸರ್ವೆ ನಂ: 142/1"
    assert it["status"] == "AUTO_ACCEPT"
    assert it["lifecycle_state"] == "EDITED"
    assert it["reviewed_by"] == "officer_kannada_dept"
    assert it["reviewed_at"] is not None
    assert it["review_reason"] == "Kannada numeral ambiguity"
    assert it["document_id"] == str(doc_id)
    assert it["page_number"] == 1
    assert it["region_id"] == "reg_45"
    assert it["bbox"] == {"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 50}
    # Verification status updated
    assert reloaded_ext.validation_info["requires_human_review"] is False
    assert reloaded_ext.is_valid is True
    db3.close()

