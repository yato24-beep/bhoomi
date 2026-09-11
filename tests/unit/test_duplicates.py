"""
Unit tests for duplicate document detection (SHA-256 hash match and vector semantic similarity).
"""

import pytest
from schemas import DuplicateMatchType, ExtractedField
from src.database.duplicates import DuplicateDetector


@pytest.fixture
def detector():
    return DuplicateDetector(vector_similarity_threshold=0.85)


def test_sha256_exact_duplicate(detector):
    # Register doc 1
    detector.register_document(
        document_id="DOC_001",
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        full_text="Sample text of land record 1",
    )

    # Check incoming doc 2 with identical hash
    res = detector.check_duplicate(
        document_id="DOC_002",
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        full_text="Sample text of land record 1",
    )

    assert res.is_duplicate is True
    assert res.match_type == DuplicateMatchType.EXACT_HASH
    assert res.matched_document_id == "DOC_001"
    assert res.sha256_match is True


def test_vector_similarity_near_duplicate(detector):
    text_1 = "उत्तर प्रदेश राजस्व परिषद ग्राम मऊ तहसील मोहनलालगंज खाता 124 राम प्रसाद गाटा 142/1 क्षेत्रफल 0.4500"
    text_2 = "उत्तर प्रदेश राजस्व परिषद ग्राम मऊ तहसील मोहनलालगंज खाता 124 राम प्रसाद गाटा 142/1 क्षेत्रफल 0.4500 प्रतिलिपि"

    detector.register_document(
        document_id="DOC_ORIGINAL",
        sha256_hash="hash_original_111",
        full_text=text_1,
    )

    res = detector.check_duplicate(
        document_id="DOC_NEAR_DUP",
        sha256_hash="hash_different_222",
        full_text=text_2,
    )

    assert res.is_duplicate is True
    assert res.match_type == DuplicateMatchType.VECTOR_SIMILARITY
    assert res.matched_document_id == "DOC_ORIGINAL"
    assert res.similarity_score >= 0.85
