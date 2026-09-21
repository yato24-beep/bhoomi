"""Generates versioned model_manifest.json with real SHA-256 checksums and exact file sizes.
"""

import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_ONNX_DIR = REPO_ROOT / "experiments" / "onnx_kannada_trocr" / "models"
TOKENIZER_DIR = MODELS_ONNX_DIR / "tokenizer"
CHECKPOINT_DIR = REPO_ROOT / "models" / "trocr" / "checkpoint-12000"
CONFIGS_DIR = REPO_ROOT / "src" / "handwriting" / "configs" / "iitb_kannada_v002"

# Hugging Face Repository identifier where these files will be hosted
DEFAULT_HF_REPO = "bhoomi-karnataka/iitb-indic-trocr-kannada-onnx"
VERSION = "1.0.0"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 4):  # 4MB chunks
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    target_files = {
        "encoder.onnx": MODELS_ONNX_DIR / "encoder.onnx",
        "encoder.onnx.data": MODELS_ONNX_DIR / "encoder.onnx.data",
        "decoder.onnx": MODELS_ONNX_DIR / "decoder.onnx",
        "decoder.onnx.data": MODELS_ONNX_DIR / "decoder.onnx.data",
        "tokenizer.json": TOKENIZER_DIR / "tokenizer.json",
        "tokenizer_config.json": TOKENIZER_DIR / "tokenizer_config.json",
        "preprocessor_config.json": CONFIGS_DIR / "preprocessor_config.json",
        "config.json": CHECKPOINT_DIR / "config.json",
        "generation_config.json": CHECKPOINT_DIR / "generation_config.json",
    }

    manifest_files = {}
    total_size = 0

    print("Computing SHA-256 hashes and file sizes...")
    for filename, filepath in target_files.items():
        if not filepath.exists():
            print(f"ERROR: Missing file {filepath}", file=sys.stderr)
            sys.exit(1)
        size = filepath.stat().st_size
        sha256 = compute_sha256(filepath)
        total_size += size
        manifest_files[filename] = {
            "filename": filename,
            "size": size,
            "sha256": sha256,
            "url": f"https://huggingface.co/{DEFAULT_HF_REPO}/resolve/main/{filename}",
            "fallback_url": f"https://huggingface.co/{DEFAULT_HF_REPO}/raw/main/{filename}",
            "required": True,
        }
        print(f"  {filename:25s} : {size:12,d} bytes | SHA-256: {sha256[:16]}...")

    manifest = {
        "manifest_version": "1.0.0",
        "model_name": "IIT Bombay Indic-TrOCR Kannada (Checkpoint-12000)",
        "model_version": VERSION,
        "format": "ONNX-FP32",
        "huggingface_repo": DEFAULT_HF_REPO,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "signatures": {
            "encoder": {
                "inputs": [{"name": "pixel_values", "shape": [1, 3, 224, 224], "type": "float32"}],
                "outputs": [{"name": "last_hidden_state", "shape": [1, 197, 768], "type": "float32"}],
            },
            "decoder": {
                "inputs": [
                    {"name": "input_ids", "shape": [1, "seq_len"], "type": "int64"},
                    {"name": "encoder_hidden_states", "shape": [1, 197, 768], "type": "float32"},
                ],
                "outputs": [{"name": "logits", "shape": [1, "seq_len", 100000], "type": "float32"}],
            },
        },
        "generation": {
            "bos_token_id": 0,
            "decoder_start_token_id": 0,
            "eos_token_id": 2,
            "pad_token_id": 1,
            "max_length": 64,
        },
        "files": manifest_files,
    }

    # Save to frontend public directory and scripts directory
    frontend_manifest_path = REPO_ROOT / "frontend" / "public" / "model_manifest.json"
    frontend_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(frontend_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest successfully written to: {frontend_manifest_path}")

    # Also save a copy in experiments
    exp_manifest_path = REPO_ROOT / "experiments" / "onnx_kannada_trocr" / "model_manifest.json"
    with open(exp_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest copy written to: {exp_manifest_path}")


if __name__ == "__main__":
    main()
