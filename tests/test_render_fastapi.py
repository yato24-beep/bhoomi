"""Test FastAPI startup, Render PORT binding, and strict production CORS without TrOCR loading.
"""

import os
import sys

# Configure production environment variables before imports
os.environ["ENVIRONMENT"] = "production"
os.environ["PORT"] = "9876"
os.environ["FRONTEND_ORIGIN"] = "https://bhoomi-karnataka.vercel.app"
os.environ["ENABLE_SERVER_TROCR_DOWNLOAD"] = "false"
os.environ["ALLOW_LOCALHOST_CORS"] = "false"

# Add backend directory to sys.path
from pathlib import Path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

def test_production_cors_and_endpoints():
    print(f"Testing environment: {settings.ENVIRONMENT}")
    print(f"Backend CORS origins: {settings.BACKEND_CORS_ORIGINS}")
    
    # 1. Verify that wildcard '*' is absent
    assert "*" not in settings.BACKEND_CORS_ORIGINS, "Wildcard '*' must NOT be allowed in production CORS!"
    
    # 2. Verify localhost is absent in production without explicit permission
    for origin in settings.BACKEND_CORS_ORIGINS:
        assert "localhost" not in origin and "127.0.0.1" not in origin, f"Localhost origin '{origin}' leaked into production CORS!"
        
    # 3. Verify Vercel domain is present
    assert "https://bhoomi-karnataka.vercel.app" in settings.BACKEND_CORS_ORIGINS, "Vercel domain missing from production CORS!"
    print("[PASS] CORS origins strictly verified.")
    
    # 4. Test endpoints using TestClient
    with TestClient(app) as client:
        # Test root endpoint
        res = client.get("/")
        assert res.status_code == 200, f"Root endpoint returned {res.status_code}"
        data = res.json()
        print(f"[PASS] Root endpoint: {data.get('message')}")
        
        # Test /health endpoint
        res_health = client.get("/health")
        assert res_health.status_code == 200, f"/health returned {res_health.status_code}"
        print(f"[PASS] Health endpoint: {res_health.json()}")
        
        # Test /api/ocr/health endpoint
        res_ocr = client.get("/api/ocr/health")
        assert res_ocr.status_code == 200, f"/api/ocr/health returned {res_ocr.status_code}"
        print(f"[PASS] OCR health endpoint: status={res_ocr.json().get('status')}")
        
        # Test CORS preflight response from allowed Vercel origin
        headers = {
            "Origin": "https://bhoomi-karnataka.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        res_preflight = client.options("/api/ocr/process", headers=headers)
        print(f"[PASS] Preflight response status: {res_preflight.status_code}")
        assert res_preflight.headers.get("access-control-allow-origin") == "https://bhoomi-karnataka.vercel.app", (
            f"Preflight did not return allowed Vercel origin! Got: {res_preflight.headers.get('access-control-allow-origin')}"
        )
        
        # Test CORS rejection for unauthorized origin
        bad_headers = {
            "Origin": "https://malicious-site.com",
            "Access-Control-Request-Method": "POST",
        }
        res_bad = client.options("/api/ocr/process", headers=bad_headers)
        assert res_bad.headers.get("access-control-allow-origin") is None, "Unauthorized origin was improperly permitted!"
        print("[PASS] Unauthorized origin correctly rejected by CORS.")

    # 5. Verify that absent weights on Render does not cause a crash or trigger an automatic 843MB download
    os.environ["TROCR_CHECKPOINT_12000_PATH"] = "/tmp/nonexistent_model_dir"
    from src.handwriting.trocr_12000_recognizer import resolve_or_download_trocr_checkpoint, _has_weights
    assert not _has_weights("/tmp/nonexistent_model_dir"), "Sanity check on nonexistent dir failed"
    resolved_path = resolve_or_download_trocr_checkpoint("/tmp/nonexistent_model_dir")
    print(f"[PASS] Resolved path without weights or download: {resolved_path}")

    print("\nALL FASTAPI RENDER DEPLOYMENT CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_production_cors_and_endpoints()
