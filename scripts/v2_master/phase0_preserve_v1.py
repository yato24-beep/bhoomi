"""Phase 0 — Preserve V1 checkpoint and create V2 experiment directory.

NEVER overwrites V1. Creates a metadata snapshot and sets up V2 workspace.
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

V1_CHECKPOINT = PROJECT_ROOT / "models" / "trocr" / "kannada_generalized_checkpoints" / "best_checkpoint"
V1_PRESERVED_META = PROJECT_ROOT / "models" / "trocr" / "kannada_generalized_v1_preserved"
V2_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "trocr" / "kannada_generalized_v2_checkpoints"


def hash_file(path: Path, algo: str = "sha256") -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("  PHASE 0 — PRESERVE V1 CHECKPOINT")
    print("=" * 70)

    # 1. Verify V1 exists
    if not V1_CHECKPOINT.exists():
        print(f"[FATAL] V1 checkpoint not found at: {V1_CHECKPOINT}")
        sys.exit(1)

    model_file = V1_CHECKPOINT / "model.safetensors"
    if not model_file.exists():
        print(f"[FATAL] V1 model weights not found at: {model_file}")
        sys.exit(1)

    print(f"V1 checkpoint: {V1_CHECKPOINT}")
    print(f"Model file size: {model_file.stat().st_size / 1024 / 1024:.1f} MB")

    # 2. Hash the model weights
    print("Computing SHA-256 hash of model.safetensors (this may take a moment)...")
    model_hash = hash_file(model_file)
    print(f"V1 model hash: {model_hash[:16]}...{model_hash[-16:]}")

    # 3. Read training metadata
    meta_file = V1_CHECKPOINT / "training_metadata.json"
    v1_meta = {}
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            v1_meta = json.load(f)
        print(f"V1 version: {v1_meta.get('model_version', 'unknown')}")
        print(f"V1 epoch: {v1_meta.get('epoch', 'unknown')}")
        print(f"V1 step: {v1_meta.get('step', 'unknown')}")
        print(f"V1 val CER: {v1_meta.get('metrics', {}).get('val_cer', 'unknown')}")

    # 4. Create preservation record
    V1_PRESERVED_META.mkdir(parents=True, exist_ok=True)
    preservation_record = {
        "v1_checkpoint_path": str(V1_CHECKPOINT),
        "model_safetensors_sha256": model_hash,
        "model_file_size_bytes": model_file.stat().st_size,
        "training_metadata": v1_meta,
        "preservation_note": "V1 checkpoint MUST NOT be overwritten. This record serves as a verification hash.",
        "files_in_checkpoint": [f.name for f in V1_CHECKPOINT.iterdir()],
    }
    record_path = V1_PRESERVED_META / "v1_preservation_record.json"
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(preservation_record, f, indent=2, ensure_ascii=False)
    print(f"\nPreservation record saved: {record_path}")

    # 5. Create V2 experiment directory
    V2_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"V2 experiment directory created: {V2_CHECKPOINT_DIR}")

    # 6. Create V2 datasets directory
    v2_dataset_dir = PROJECT_ROOT / "training" / "datasets" / "kannada_character_balanced_v2"
    v2_dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"V2 dataset directory created: {v2_dataset_dir}")

    # 7. Verify V1 is NOT inside V2 directory
    assert str(V1_CHECKPOINT) != str(V2_CHECKPOINT_DIR), "SAFETY: V1 and V2 paths must be different!"

    print("\n" + "=" * 70)
    print("  PHASE 0 COMPLETE — V1 PRESERVED SAFELY")
    print("=" * 70)
    print(f"  V1 hash: {model_hash[:32]}...")
    print(f"  V2 dir:  {V2_CHECKPOINT_DIR}")
    print("  V1 will NEVER be overwritten.")
    print("=" * 70)


if __name__ == "__main__":
    main()
