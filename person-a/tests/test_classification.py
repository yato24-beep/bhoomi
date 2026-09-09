"""person-a/tests/test_classification.py
Unit tests for rule-based land record document classification.
"""

from src.classification.classifier import DocumentClassifier
from src.config.loader import get_state_config
from src.schemas import TableStructure, BoundingBox


def test_classify_maharashtra_7_12():
    cfg = get_state_config("MH")
    classifier = DocumentClassifier(cfg)
    text = "महाराष्ट्र शासन गाव नमुना सात (७/१२) अधिकार अभिलेख पत्रक गट क्रमांक १२४/२"
    res = classifier.classify(text)
    assert res.predicted_type == "land_record_7_12"
    assert res.state == "MH"
    assert res.confidence >= 0.60
    assert len(res.matched_signals) >= 1


def test_classify_karnataka_rtc():
    cfg = get_state_config("KA")
    classifier = DocumentClassifier(cfg)
    text = "GOVERNMENT OF KARNATAKA BHOOMI RTC FORM NO 16 RECORD OF RIGHTS SURVEY NO 45/1 KHATHEDAR"
    res = classifier.classify(text)
    assert res.predicted_type == "bhoomi_rtc"
    assert res.state == "KA"
    assert res.confidence >= 0.60


def test_classify_unknown_empty():
    cfg = get_state_config("DEFAULT")
    classifier = DocumentClassifier(cfg)
    res = classifier.classify("")
    assert res.predicted_type == "unknown"
    assert res.confidence == 0.0
