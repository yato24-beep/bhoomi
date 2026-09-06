"""
src/utils/hashing.py
Deterministic SHA-256 document hashing for identity, auditability, and duplicate tracking.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


def compute_sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw binary bytes."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def compute_sha256_file(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for hashing: {file_path}")
    
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# Alias for calculate_sha256
calculate_sha256 = compute_sha256_file


def compute_sha256_text(text: str) -> str:
    """Compute SHA-256 hex digest of a UTF-8 string."""
    return compute_sha256_bytes(text.encode("utf-8"))


def compute_structured_field_hash(fields_dict: Dict[str, Any]) -> str:
    """
    Compute deterministic SHA-256 hash of normalized structured fields
    ignoring timestamp/volatile metadata.
    """
    # Sort keys and normalize values to ensure deterministic hashing
    canonical_data = {
        k: str(v) for k, v in sorted(fields_dict.items())
    }
    json_bytes = json.dumps(canonical_data, sort_keys=True).encode("utf-8")
    return compute_sha256_bytes(json_bytes)
