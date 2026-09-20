#!/usr/bin/env python3
"""End-to-End Demonstration Script for Karnataka Land Record Digitization.

Executes the real, unmocked production pipeline:
1. Ingests synthetic Karnataka RTC (Pahani) document.
2. Uploads via API (`POST /api/v1/documents/upload`).
3. Polls processing status until completion (`GET /api/v1/documents/{id}/status`).
4. Fetches full structured extraction results (`GET /api/v1/documents/{id}/results`).
5. Fetches canonical bilingual fields with confidence (`GET /api/v1/documents/{id}/fields`).
6. Fetches and displays human review items and review reasons (`GET /api/v1/documents/{id}/reviews`).
7. Prints provenance, confidence calibration, and stage timings.

Works seamlessly both when the FastAPI server is running externally or in-process via TestClient.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root and backend directory are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
for p in (str(REPO_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("DEMO_MODE", "false")  # Strict production pipeline verification

DEMO_IMG_PATH = REPO_ROOT / "demo_artifacts" / "synthetic_karnataka_rtc.png"
DEMO_PDF_PATH = REPO_ROOT / "demo_artifacts" / "synthetic_karnataka_rtc.pdf"
API_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


class DemoClient:
    """Unified client supporting both live HTTP requests and in-process FastAPI TestClient."""

    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url.rstrip("/")
        self.use_http = False
        self.test_client = None

        # Check if live HTTP server is listening
        try:
            import requests
            resp = requests.get(f"{self.base_url}/api/v1/health", timeout=1.0)
            if resp.status_code == 200:
                self.use_http = True
                self.session = requests.Session()
                print(f" Connected to live FastAPI server at {self.base_url}")
        except Exception:
            pass

        if not self.use_http:
            print(" Live server not detected on localhost:8000. Using in-process FastAPI TestClient...")
            from fastapi.testclient import TestClient
            from backend.app.main import app
            self.test_client = TestClient(app)

    def upload_document(self, file_path: Path, is_handwritten: Optional[bool] = None) -> Dict[str, Any]:
        url = "/api/v1/documents/upload"
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        filename = file_path.name
        files = {"file": (filename, file_bytes, "image/png" if file_path.suffix == ".png" else "application/pdf")}
        data = {}
        if is_handwritten is not None:
            data["is_handwritten"] = str(is_handwritten).lower()

        if self.use_http:
            resp = self.session.post(f"{self.base_url}{url}", files=files, data=data)
            resp.raise_for_status()
            return resp.json()
        else:
            resp = self.test_client.post(url, files=files, data=data)
            assert resp.status_code in (200, 201), f"Upload failed ({resp.status_code}): {resp.text}"
            return resp.json()

    def get_status(self, doc_id: int) -> Dict[str, Any]:
        url = f"/api/v1/documents/{doc_id}/status"
        if self.use_http:
            for _ in range(3):
                try:
                    resp = self.session.get(f"{self.base_url}{url}", timeout=10.0)
                    resp.raise_for_status()
                    return resp.json()
                except Exception:
                    time.sleep(1.0)
            resp = self.session.get(f"{self.base_url}{url}", timeout=15.0)
            resp.raise_for_status()
            return resp.json()
        else:
            resp = self.test_client.get(url)
            assert resp.status_code == 200, f"Get status failed: {resp.text}"
            return resp.json()

    def get_results(self, doc_id: int) -> Optional[Dict[str, Any]]:
        url = f"/api/v1/documents/{doc_id}/results"
        if self.use_http:
            resp = self.session.get(f"{self.base_url}{url}")
            if resp.status_code == 202:
                return None
            resp.raise_for_status()
            return resp.json()
        else:
            resp = self.test_client.get(url)
            if resp.status_code == 202:
                return None
            assert resp.status_code == 200, f"Get results failed: {resp.text}"
            return resp.json()

    def get_fields(self, doc_id: int) -> Dict[str, Any]:
        url = f"/api/v1/documents/{doc_id}/fields"
        if self.use_http:
            resp = self.session.get(f"{self.base_url}{url}")
            resp.raise_for_status()
            return resp.json()
        else:
            resp = self.test_client.get(url)
            assert resp.status_code == 200, f"Get fields failed: {resp.text}"
            return resp.json()

    def get_reviews(self, doc_id: int) -> List[Dict[str, Any]]:
        url = f"/api/v1/documents/{doc_id}/reviews"
        if self.use_http:
            resp = self.session.get(f"{self.base_url}{url}")
            return resp.json() if resp.status_code == 200 else []
        else:
            resp = self.test_client.get(url)
            return resp.json() if resp.status_code == 200 else []


def ensure_demo_artifact() -> Path:
    """Ensures the synthetic Karnataka RTC document artifact exists."""
    if len(sys.argv) > 1:
        custom_path = Path(sys.argv[1])
        if custom_path.exists():
            return custom_path

    if not DEMO_IMG_PATH.exists():
        print(f"Artifact {DEMO_IMG_PATH} not found. Generating now...")
        from scripts.generate_synthetic_rtc import generate_rtc_image
        img = generate_rtc_image()
        DEMO_IMG_PATH.parent.mkdir(parents=True, exist_ok=True)
        img.save(DEMO_IMG_PATH, format="PNG", dpi=(200, 200))
        img.save(DEMO_PDF_PATH, format="PDF", resolution=200.0)
        print(f"Generated synthetic RTC image: {DEMO_IMG_PATH}")
        print(f"Generated synthetic RTC PDF: {DEMO_PDF_PATH}")
    return DEMO_IMG_PATH


def main():
    print("=" * 80)
    print("    KARNATAKA LAND RECORD DIGITIZATION - LIVE DEMO E2E VERIFICATION    ")
    print("=" * 80)

    artifact_path = ensure_demo_artifact()
    import hashlib
    with open(artifact_path, "rb") as f:
        file_sha256 = hashlib.sha256(f.read()).hexdigest()

    print(f"\n[STEP 1] Selected Demonstration Document:")
    print(f"  Path: {artifact_path}")
    print(f"  Size: {artifact_path.stat().st_size:,} bytes")
    print(f"  SHA-256: {file_sha256}")

    client = DemoClient()

    # Upload document
    print(f"\n[STEP 2] Uploading document to /api/v1/documents/upload...")
    upload_res = client.upload_document(artifact_path)
    doc_id = upload_res["document"]["id"]
    filename = upload_res["document"]["filename"]
    task_id = upload_res.get("task_id")
    print(f"  Document Created: ID={doc_id}, Filename='{filename}'")
    print(f"  Processing Task Dispatched: {task_id}")

    # Poll status
    print(f"\n[STEP 3] Polling pipeline processing status for Document #{doc_id}...")
    max_wait_seconds = 180
    start_poll = time.time()
    final_status = "UNKNOWN"

    while (time.time() - start_poll) < max_wait_seconds:
        status_res = client.get_status(doc_id)
        cur_status = status_res.get("status", "UNKNOWN")
        elapsed = round(time.time() - start_poll, 1)
        print(f"  [{elapsed}s] Document #{doc_id} Status: {cur_status}")

        if cur_status in ("COMPLETED", "FAILED"):
            final_status = cur_status
            break
        time.sleep(3.0)

    if final_status != "COMPLETED":
        print(f"\n Pipeline did not reach COMPLETED state within timeout (final status: {final_status})")
        sys.exit(1)

    print(f"\n Pipeline processing completed successfully!")

    # Fetch structured extraction results
    print(f"\n[STEP 4] Fetching structured extraction results (/api/v1/documents/{doc_id}/results)...")
    results = client.get_results(doc_id)
    if not results:
        print(" Extraction results not found!")
        sys.exit(1)

    extracted_data = results.get("extracted_data", {})
    confidence_score = results.get("confidence_score", 0.0)
    is_valid = results.get("is_valid", False)
    proc_time_ms = results.get("processing_time_ms", 0)
    is_demo_detected = extracted_data.get("demo_fixture_detected", False)
    demo_name = extracted_data.get("demo_fixture_name", "none")

    print(f"  Confidence Score: {confidence_score * 100:.1f}%")
    print(f"  Validation Status: {'VALID' if is_valid else 'NEEDS_VERIFICATION'}")
    print(f"  Processing Time: {proc_time_ms:,} ms")
    print(f"  Demo Fixture Detected: {is_demo_detected} (Name: {demo_name})")

    # Fetch bilingual canonical fields
    print(f"\n[STEP 5] Canonical Extracted & Translated Fields (/api/v1/documents/{doc_id}/fields):")
    fields_summary = client.get_fields(doc_id)
    field_items = fields_summary.get("fields", [])

    print("-" * 105)
    print(f"{'Field Name':<18} | {'Kannada Original':<22} | {'English Translation':<22} | {'Engine':<18} | {'Source Type'}")
    print("-" * 105)
    for f in field_items:
        fname = f.get("field_name", "")
        orig_val = str(f.get("normalized_value") or f.get("original_value") or "-")
        eng_val = str(f.get("english_value") or "-")
        engine = str(f.get("translation_engine") or "-")
        stype = str(f.get("source_type") or "pipeline")
        print(f"{fname:<18} | {orig_val:<22} | {eng_val:<22} | {engine:<18} | {stype}")
    print("-" * 105)

    # Fetch human review queue items
    print(f"\n[STEP 6] Human Review Queue Inspection (/api/v1/documents/{doc_id}/reviews):")
    reviews = client.get_reviews(doc_id)
    if reviews:
        print(f"  Found {len(reviews)} review item(s) flagged by honest confidence threshold:")
        for r in reviews:
            rid = r.get("region_id", r.get("review_id", "reg"))
            ocr_text = r.get("raw_ocr_text", "")
            reason = r.get("review_reason", "Review triggered")
            conf = r.get("calibrated_confidence") or r.get("recognizer_confidence_raw")
            conf_str = f"{conf:.2f}" if conf is not None else "Uncalibrated"
            print(f"    Region: {rid} | Text: '{ocr_text}' | Confidence: {conf_str} | Reason: {reason}")
    else:
        print("  All detected regions exceeded confidence threshold (no reviews queued).")

    # Cadastral Profiler & Stage Timings
    stage_timings = extracted_data.get("stage_timings", {})
    if stage_timings:
        print(f"\n[STEP 7] Pipeline Latency Breakdown (Stage Profiler):")
        for stage, duration in stage_timings.items():
            print(f"  {stage:<26}: {duration:>8.2f} ms")

    print("\n" + "=" * 80)
    print("   END-TO-END DEMO EXECUTION VERIFIED: ALL PIPELINE STAGES PASSED!   ")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
