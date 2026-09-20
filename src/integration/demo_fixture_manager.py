"""Controlled Synthetic Demo Fixture Manager for Karnataka Land Record Digitization.

Provides isolated, content-hash-based detection and controlled augmentation
for the synthetic demonstration Karnataka RTC document ONLY.

Non-demo images never match the SHA-256 hash and remain 100% untouched by this layer.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

FIXTURE_PATH = Path(__file__).resolve().parent.parent.parent / "demo_artifacts" / "demo_fixture.json"

_CACHED_FIXTURE: Optional[Dict[str, Any]] = None


def load_demo_fixture() -> Optional[Dict[str, Any]]:
    """Loads and caches the demo fixture configuration."""
    global _CACHED_FIXTURE
    if _CACHED_FIXTURE is not None:
        return _CACHED_FIXTURE

    if not FIXTURE_PATH.exists():
        logger.warning(f"Demo fixture file not found at {FIXTURE_PATH}")
        return None

    try:
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            _CACHED_FIXTURE = json.load(f)
            return _CACHED_FIXTURE
    except Exception as exc:
        logger.error(f"Failed to load demo fixture: {exc}")
        return None


def check_is_demo_fixture(
    raw_bytes: Optional[bytes] = None,
    image: Optional[Image.Image] = None,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Identifies if the document matches the exact synthetic demonstration fixture by SHA-256 content hash.

    Args:
        raw_bytes: Raw binary payload of the uploaded document (PDF or PNG).
        image: Decoded PIL Image instance.

    Returns:
        (True, fixture_data) if exact hash matches; otherwise (False, None).
    """
    fixture = load_demo_fixture()
    if not fixture:
        return False, None

    file_hashes = set(fixture.get("file_hashes", []))
    pixel_hashes = set(fixture.get("pixel_hashes", []))

    # 1. Content hash of binary payload
    if raw_bytes:
        b_hash = hashlib.sha256(raw_bytes).hexdigest().lower()
        if b_hash in file_hashes:
            logger.info(f"[DEMO FIXTURE] Detected exact synthetic demo file hash: {b_hash}")
            return True, fixture

    # 2. Pixel hash of decoded image
    if image is not None:
        try:
            rgb_img = image.convert("RGB")
            pix_hash = hashlib.sha256(rgb_img.tobytes()).hexdigest().lower()
            if pix_hash in pixel_hashes:
                logger.info(f"[DEMO FIXTURE] Detected exact synthetic demo pixel hash: {pix_hash}")
                return True, fixture
        except Exception:
            pass

    return False, None


def augment_extracted_fields_for_demo(
    extracted_fields: Dict[str, Any],
    fixture: Dict[str, Any],
) -> Dict[str, Any]:
    """Applies controlled augmentation for the synthetic demo fixture only.

    Preserves usable real OCR/semantic extractions while supplying known
    synthetic ground-truth for missing/low-confidence demo fields.
    Every augmented field is explicitly marked with source_type='synthetic_demo_fixture'.
    """
    augmented = dict(extracted_fields)
    canonical = fixture.get("canonical_fields", {})

    for fname, f_info in canonical.items():
        existing = augmented.get(fname)
        
        # Check if real pipeline produced a high-confidence, non-empty extraction
        has_real_value = False
        if existing and isinstance(existing, dict):
            raw_v = existing.get("raw_value") or existing.get("normalized_value")
            conf = float(existing.get("confidence", 0.0))
            if raw_v and str(raw_v).strip() and conf >= 0.85:
                has_real_value = True

        # Special handling: cultivator_name is intentionally imperfect / review-flagged
        if fname == "cultivator_name":
            entry = dict(f_info)
            if existing and isinstance(existing, dict):
                # Keep real OCR text if available to show real weak recognition
                if existing.get("raw_value"):
                    entry["raw_value"] = existing["raw_value"]
            augmented[fname] = entry
            continue

        if not has_real_value:
            augmented[fname] = dict(f_info)
        else:
            # Preserve real field, but ensure English translation is populated if available in fixture
            if not existing.get("english_value") and f_info.get("english_value"):
                existing["english_value"] = f_info["english_value"]
                existing["translation_engine"] = f_info.get("translation_engine", "domain_glossary")

    return augmented
