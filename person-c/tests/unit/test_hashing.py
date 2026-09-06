"""
Unit tests for SHA-256 document and field hashing.
"""

import pytest
from src.utils.hashing import (
    compute_sha256_bytes,
    compute_sha256_text,
    compute_structured_field_hash,
)


def test_sha256_bytes():
    data = b"Sample Land Record Document"
    h1 = compute_sha256_bytes(data)
    h2 = compute_sha256_bytes(data)
    assert len(h1) == 64
    assert h1 == h2


def test_sha256_text():
    text = "उत्तर प्रदेश राजस्व परिषद"
    h = compute_sha256_text(text)
    assert len(h) == 64
    assert h == compute_sha256_text(text)


def test_deterministic_structured_field_hash():
    dict1 = {"khasra_number": "142/1", "owner_name": "Ram Prasad", "land_area": "0.45"}
    dict2 = {"land_area": "0.45", "owner_name": "Ram Prasad", "khasra_number": "142/1"}
    
    # Hashes should be identical regardless of insertion key order
    h1 = compute_structured_field_hash(dict1)
    h2 = compute_structured_field_hash(dict2)
    assert h1 == h2
