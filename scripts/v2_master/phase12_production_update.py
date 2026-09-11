"""Phase 12 — Production Pipeline Update.

Updates DEFAULT_KANNADA_MODEL_PATH in trocr_recognizer.py to point to V2 checkpoint.
Only runs after V2 passes the 90% accuracy gate.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

V1_PATH = PROJECT_ROOT / "models" / "trocr" / "kannada_generalized_checkpoints" / "best_checkpoint"
V2_PATH = PROJECT_ROOT / "models" / "trocr" / "kannada_generalized_v2_checkpoints" / "best_checkpoint"
RECOGNIZER_FILE = PROJECT_ROOT / "src" / "handwriting" / "trocr_recognizer.py"

# Safety gate: V2 evaluation report must show ≥90% character accuracy
REPORT_PATH = PROJECT_ROOT / "evaluation" / "v2_evaluation_report.json"


def verify_gate():
    """Check if V2 passes the 90% accuracy gate."""
    if not REPORT_PATH.exists():
        print("[FATAL] V2 evaluation report not found. Run Phase 9 first.")
        return False

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    if "V2" not in report:
        print("[FATAL] V2 results not found in evaluation report.")
        return False

    char_acc = report["V2"]["char_accuracy"]
    print(f"  V2 Character Accuracy: {char_acc:.2f}%")

    if char_acc < 90.0:
        print(f"  [BLOCKED] V2 does not meet 90% threshold. Cannot promote.")
        return False

    print(f"  [PASS] V2 meets promotion criteria (≥90%)")
    return True


def verify_v1_intact():
    """Verify V1 checkpoint hasn't been modified."""
    preservation = PROJECT_ROOT / "models" / "trocr" / "kannada_generalized_v1_preserved" / "v1_preservation_record.json"
    if not preservation.exists():
        print("[WARN] V1 preservation record not found.")
        return True  # Proceed but warn

    with open(preservation, "r", encoding="utf-8") as f:
        record = json.load(f)

    expected_hash = record["model_safetensors_sha256"]
    model_file = V1_PATH / "model.safetensors"
    if not model_file.exists():
        print("[FATAL] V1 model file missing!")
        return False

    h = hashlib.sha256()
    with open(model_file, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    actual_hash = h.hexdigest()

    if actual_hash != expected_hash:
        print("[FATAL] V1 checkpoint has been modified! Aborting.")
        return False

    print(f"  V1 integrity verified (hash matches)")
    return True


def update_recognizer():
    """Update DEFAULT_KANNADA_MODEL_PATH in trocr_recognizer.py."""
    if not RECOGNIZER_FILE.exists():
        print(f"[FATAL] Recognizer file not found: {RECOGNIZER_FILE}")
        return False

    content = RECOGNIZER_FILE.read_text(encoding="utf-8")

    # Find the current DEFAULT_KANNADA_MODEL_PATH
    pattern = r'(DEFAULT_KANNADA_MODEL_PATH\s*=\s*)"([^"]*)"'
    match = re.search(pattern, content)
    if not match:
        pattern = r"(DEFAULT_KANNADA_MODEL_PATH\s*=\s*)'([^']*)'"
        match = re.search(pattern, content)

    if not match:
        print("[FATAL] Could not find DEFAULT_KANNADA_MODEL_PATH in recognizer file.")
        return False

    old_path = match.group(2)
    new_path = "models/trocr/kannada_generalized_v2_checkpoints/best_checkpoint"

    if old_path == new_path:
        print(f"  Already pointing to V2: {old_path}")
        return True

    print(f"  Old path: {old_path}")
    print(f"  New path: {new_path}")

    new_content = content.replace(f'"{old_path}"', f'"{new_path}"')
    if new_content == content:
        new_content = content.replace(f"'{old_path}'", f"'{new_path}'")

    RECOGNIZER_FILE.write_text(new_content, encoding="utf-8")
    print(f"  Updated: {RECOGNIZER_FILE}")
    return True


def main():
    print("=" * 70)
    print("  PHASE 12 — PRODUCTION PIPELINE UPDATE")
    print("=" * 70)

    # Step 1: Verify V2 passes gate
    print("\n  Step 1: Accuracy Gate Verification")
    if not verify_gate():
        print("\n  [ABORTED] V2 does not meet promotion criteria.")
        sys.exit(1)

    # Step 2: Verify V1 is intact
    print("\n  Step 2: V1 Integrity Check")
    if not verify_v1_intact():
        print("\n  [ABORTED] V1 integrity failure.")
        sys.exit(1)

    # Step 3: Verify V2 checkpoint exists
    print("\n  Step 3: V2 Checkpoint Verification")
    v2_model = V2_PATH / "model.safetensors"
    if not v2_model.exists():
        print(f"  [FATAL] V2 model not found at: {v2_model}")
        sys.exit(1)
    print(f"  V2 model: {v2_model} ({v2_model.stat().st_size / 1024 / 1024:.1f} MB)")

    # Step 4: Update recognizer
    print("\n  Step 4: Update Production Recognizer Path")
    if not update_recognizer():
        print("\n  [ABORTED] Failed to update recognizer.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  PHASE 12 COMPLETE — Production Pipeline Updated to V2")
    print(f"{'='*70}")
    print(f"  V1 preserved at: {V1_PATH}")
    print(f"  V2 active at:    {V2_PATH}")
    print(f"  Recognizer:      {RECOGNIZER_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
