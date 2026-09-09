"""person-a/tests/test_hashing.py
Unit tests for cryptographic SHA-256 document hashing.
"""

import io
from pathlib import Path
import numpy as np
import pytest

from src.utils.hashing import compute_document_hash


def test_hash_empty_bytes():
    assert compute_document_hash(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_hash_known_ascii():
    assert compute_document_hash(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_hash_stream():
    stream = io.BytesIO(b"land_record_payload_12345")
    h1 = compute_document_hash(stream)
    stream.seek(0)
    h2 = compute_document_hash(stream.read())
    assert h1 == h2


def test_hash_numpy_array():
    arr = np.zeros((50, 50, 3), dtype=np.uint8)
    h = compute_document_hash(arr)
    assert len(h) == 64
