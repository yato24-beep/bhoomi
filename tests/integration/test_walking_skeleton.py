"""
Integration tests for the complete 3-person walking skeleton pipeline.
"""

import pytest
from schemas import DocumentInput, FinalDocumentResult, ValidationStatus
from src.integration.walking_skeleton import EndToEndPipeline


def test_walking_skeleton_end_to_end():
    pipeline = EndToEndPipeline()

    doc_input = DocumentInput(
        document_id="SKELETON_TEST_001",
        selected_state="UP",
        metadata={
            "sample_text": """
उत्तर प्रदेश शासन - राजस्व परिषद
खातौनी (अधिकार अभिलेख)
ग्राम का नाम: मऊ  परगना: मोहनलालगंज  तहसील: मोहनलालगंज  जनपद: लखनऊ
फसली वर्ष: 1428-1433
खाता संख्या: 00124
खातेदार का नाम: श्री राम प्रसाद
पिता का नाम: श्याम लाल
गाटा संख्या: 142/1
क्षेत्रफल (हेक्टेयर): 0.4500
            """
        },
    )

    result = pipeline.process(doc_input)

    assert isinstance(result, FinalDocumentResult)
    assert result.document_id == "SKELETON_TEST_001"
    assert len(result.sha256_hash) == 64
    assert result.state == "UP"
    assert "extraction" in result.pipeline_stages_completed
    assert "gis_validation" in result.pipeline_stages_completed
    assert "confidence_scoring" in result.pipeline_stages_completed

    # Validate extracted fields
    assert "owner_name" in result.fields
    assert result.fields["owner_name"].normalized_value == "राम प्रसाद"
    assert result.fields["khasra_number"].normalized_value == "142/1"
    assert result.fields["land_area"].normalized_value == 0.4500
    assert result.gis_validation.is_verified is True
    assert result.overall_confidence > 0.80
