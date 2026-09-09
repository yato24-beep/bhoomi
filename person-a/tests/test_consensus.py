import pytest

from src.ocr.consensus import (
    OCRConsensusCandidate,
    OCRConsensusEngine,
    normalized_levenshtein_distance,
)


def test_normalized_levenshtein():
    assert normalized_levenshtein_distance("124/2A", "124/2A") == 0.0
    assert normalized_levenshtein_distance("124/2A", "124/2B") == pytest.approx(1 / 6, 0.01)
    assert normalized_levenshtein_distance("", "test") == 1.0


def test_consensus_unanimous():
    engine = OCRConsensusEngine()
    c1 = OCRConsensusCandidate(engine_name="Paddle", raw_text="124/2A", confidence=0.95)
    c2 = OCRConsensusCandidate(engine_name="TrOCR", raw_text="124/2A", confidence=0.92)
    res = engine.evaluate_consensus([c1, c2])
    assert res.selected_text == "124/2A"
    assert res.disagreement_detected is False
    assert res.selected_confidence == 0.95


def test_consensus_disagreement_penalty():
    engine = OCRConsensusEngine(disagreement_threshold=0.20, penalty_factor=0.20)
    c1 = OCRConsensusCandidate(engine_name="Paddle", raw_text="124/2A", confidence=0.90)
    c2 = OCRConsensusCandidate(engine_name="TrOCR", raw_text="999/9Z", confidence=0.80)
    res = engine.evaluate_consensus([c1, c2])
    assert res.disagreement_detected is True
    assert res.selected_confidence < 0.90
