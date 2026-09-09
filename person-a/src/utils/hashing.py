"""person-a/src/utils/hashing.py
NIST FIPS 180-4 compliant SHA-256 cryptographic hashing for document identity.
"""

import hashlib
import io
from pathlib import Path
from typing import BinaryIO, Union
import numpy as np


def compute_document_hash(
    data: Union[bytes, str, Path, BinaryIO, np.ndarray],
    chunk_size: int = 65536,
) -> str:
    """Compute the hexadecimal SHA-256 hash of document data.

    Supports raw bytes, file paths, stream objects, and numpy image arrays.
    """
    hasher = hashlib.sha256()

    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.is_file():
            raise FileNotFoundError(f"Cannot hash non-existent file: {path}")
        with path.open("rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)

    elif isinstance(data, bytes):
        hasher.update(data)

    elif isinstance(data, (io.BytesIO, io.BufferedReader)) or hasattr(data, "read"):
        current_pos = data.tell() if hasattr(data, "tell") else None
        while chunk := data.read(chunk_size):
            hasher.update(chunk)
        if current_pos is not None and hasattr(data, "seek"):
            data.seek(current_pos)

    elif isinstance(data, np.ndarray):
        hasher.update(data.tobytes())

    else:
        raise TypeError(f"Unsupported data type for SHA-256 calculation: {type(data)}")

    return hasher.hexdigest()
