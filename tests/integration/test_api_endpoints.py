"""API integration tests for FastAPI endpoints using async client:
- GET /health
- POST /api/ocr/process (all 4 modalities + error scenarios)
"""

import io
import json
import unittest
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import httpx
from src.api.main import create_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _ensure_sample_images(samples_dir: Path) -> dict:
    """Ensures test images for all 4 modalities exist."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    images = {
        "kannada_printed": samples_dir / "sample_kannada_crop.png",
        "kannada_handwritten": samples_dir / "sample_handwritten_crop.png",
        "english_printed": samples_dir / "sample_english_printed.png",
        "english_handwritten": samples_dir / "sample_english_handwritten.png",
    }

    if not images["english_printed"].exists():
        img = Image.new("RGB", (600, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 32)
        except Exception:
            font = ImageFont.load_default()
        draw.text((20, 30), "SURVEY NO 142 VILLAGE RECORD", fill=(0, 0, 0), font=font)
        img.save(images["english_printed"])

    if not images["english_handwritten"].exists():
        img = Image.new("RGB", (600, 100), color=(252, 250, 242))
        draw = ImageDraw.Draw(img)
        hw_font = None
        for f in ["C:/Windows/Fonts/segoepr.ttf", "C:/Windows/Fonts/comic.ttf", "C:/Windows/Fonts/arial.ttf"]:
            if os.path.exists(f):
                try:
                    hw_font = ImageFont.truetype(f, 32)
                    break
                except Exception:
                    pass
        if not hw_font:
            hw_font = ImageFont.load_default()
        draw.text((20, 30), "Survey Number 142", fill=(15, 25, 60), font=hw_font)
        img.save(images["english_handwritten"])

    return images


class TestOcrApiEndpoints(unittest.IsolatedAsyncioTestCase):
    """Test suite for FastAPI endpoints using async ASGI transport."""

    @classmethod
    def setUpClass(cls):
        cls.samples_dir = PROJECT_ROOT / "data" / "samples"
        cls.images = _ensure_sample_images(cls.samples_dir)
        cls.app = create_app()
        cls.transport = httpx.ASGITransport(app=cls.app)

    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=self.transport, base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_health_check_endpoint(self):
        """Verify GET /health returns 200 and expected service info."""
        response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("supported_modalities", data)
        self.assertIn("printed_kannada", data["supported_modalities"])
        self.assertIn("handwritten_kannada", data["supported_modalities"])
        self.assertIn("printed_english", data["supported_modalities"])
        self.assertIn("handwritten_english", data["supported_modalities"])

    async def test_process_printed_kannada(self):
        """Verify POST /api/ocr/process with Printed Kannada."""
        img_path = self.images["kannada_printed"]
        with open(img_path, "rb") as f:
            file_bytes = f.read()

        response = await self.client.post(
            "/api/ocr/process",
            files={"file": ("sample_kannada_crop.png", file_bytes, "image/png")},
            data={"language": "kannada", "is_handwritten": "false"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "completed")
        self.assertGreater(len(data["ordered_regions"]), 0)
        self.assertEqual(data["ordered_regions"][0]["model_name"], "paddleocr-kannada")
        self.assertIsNotNone(data["merged_text"])
        self.assertGreater(len(data["merged_text"].strip()), 0)

    async def test_process_handwritten_kannada(self):
        """Verify POST /api/ocr/process with Handwritten Kannada."""
        img_path = self.images["kannada_handwritten"]
        with open(img_path, "rb") as f:
            file_bytes = f.read()

        response = await self.client.post(
            "/api/ocr/process",
            files={"file": ("sample_handwritten_crop.png", file_bytes, "image/png")},
            data={"language": "kannada", "is_handwritten": "true"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["status"], ["completed", "flagged_for_review"])
        self.assertGreater(len(data["ordered_regions"]), 0)
        self.assertIn("kannada", data["ordered_regions"][0]["model_name"].lower())
        self.assertIsNotNone(data["merged_text"])

    async def test_process_printed_english(self):
        """Verify POST /api/ocr/process with Printed English."""
        img_path = self.images["english_printed"]
        with open(img_path, "rb") as f:
            file_bytes = f.read()

        response = await self.client.post(
            "/api/ocr/process",
            files={"file": ("sample_english_printed.png", file_bytes, "image/png")},
            data={"language": "english", "is_handwritten": "false"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "completed")
        self.assertGreater(len(data["ordered_regions"]), 0)
        self.assertEqual(data["ordered_regions"][0]["model_name"], "paddleocr-english")
        self.assertIn("SURVEY NO", data["merged_text"].upper())

    async def test_process_handwritten_english(self):
        """Verify POST /api/ocr/process with Handwritten English."""
        img_path = self.images["english_handwritten"]
        with open(img_path, "rb") as f:
            file_bytes = f.read()

        response = await self.client.post(
            "/api/ocr/process",
            files={"file": ("sample_english_handwritten.png", file_bytes, "image/png")},
            data={"language": "english", "is_handwritten": "true"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "completed")
        self.assertGreater(len(data["ordered_regions"]), 0)
        self.assertEqual(data["ordered_regions"][0]["model_name"], "microsoft/trocr-small-handwritten")
        self.assertIsNotNone(data["merged_text"])

    async def test_process_with_sub_regions_json(self):
        """Verify POST /api/ocr/process with structured candidate sub-regions."""
        img_path = self.images["english_printed"]
        regions_payload = json.dumps([
            {"bbox": [10, 10, 300, 90], "language": "english", "is_handwritten": False},
        ])
        with open(img_path, "rb") as f:
            file_bytes = f.read()

        response = await self.client.post(
            "/api/ocr/process",
            files={"file": ("sample_english_printed.png", file_bytes, "image/png")},
            data={"regions": regions_payload, "document_id": "test_doc_api_001"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document_id"], "test_doc_api_001")
        self.assertEqual(len(data["ordered_regions"]), 1)

    async def test_error_handling_invalid_file_bytes(self):
        """Verify 400 response on non-image file upload."""
        response = await self.client.post(
            "/api/ocr/process",
            files={"file": ("corrupt.txt", b"not an image file at all", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid or unreadable image format", response.json()["detail"])

    async def test_error_handling_invalid_regions_json(self):
        """Verify 400 response on malformed regions JSON."""
        img_path = self.images["english_printed"]
        with open(img_path, "rb") as f:
            file_bytes = f.read()

        response = await self.client.post(
            "/api/ocr/process",
            files={"file": ("sample_english_printed.png", file_bytes, "image/png")},
            data={"regions": "{invalid_json_format}"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid 'regions' JSON", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
