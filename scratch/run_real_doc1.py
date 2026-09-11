import os
import sys
import json
import time

sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import src
from PIL import Image

from src.preprocessing.image_enhancement import preprocess_document_image
from src.integration.document_pipeline import DocumentProcessingPipeline
from src.integration.person_c_adapter import PersonCAdapter

DOC_PATH = r"c:\Land Record\doc1.jpeg"

print(f"=== PROCESSING REAL DOCUMENT: {DOC_PATH} ===")
start_time = time.perf_counter()

# Step 1: Document Processing Pipeline (Person B)
pipeline = DocumentProcessingPipeline(
    confidence_threshold=0.60,
    apply_preprocessing=True,
    apply_normalization=True,
    apply_segmentation=True,
)

response = pipeline.process_document(image=DOC_PATH)

duration = time.perf_counter() - start_time
print(f"Pipeline finished in {duration:.2f}s")
print(f"Status: {response.status}")
print(f"Confidence: {response.document_confidence}")
print(f"Regions: {len(response.ordered_regions)}")
print(f"Engine breakdown: {response.engine_breakdown}")
print(f"Requires human review: {response.requires_human_review}")
print(f"Warnings ({len(response.warnings)}): {response.warnings[:5]}")

kn_text = response.original_kannada_text or response.merged_text or ""
en_text = response.translated_text or ""

with open(r"c:\Land Record\scratch\doc1_kannada_text.txt", "w", encoding="utf-8") as f:
    f.write(kn_text)

with open(r"c:\Land Record\scratch\doc1_english_translation.txt", "w", encoding="utf-8") as f:
    f.write(en_text)

print("\n--- ORIGINAL KANNADA RECONSTRUCTION (first 500 chars) ---")
try:
    print(kn_text[:500])
except Exception:
    print(kn_text[:500].encode("unicode_escape").decode("ascii"))

print("\n--- ENGLISH TRANSLATION (first 500 chars) ---")
try:
    print(en_text[:500])
except Exception:
    print(en_text[:500].encode("unicode_escape").decode("ascii"))

# Step 2: Person C Extraction
with open(DOC_PATH, "rb") as f:
    raw_bytes = f.read()

c_res = PersonCAdapter.execute_person_c(b_response=response, file_bytes=raw_bytes)
print("\n--- PERSON C EXTRACTED LAND RECORD FIELDS ---")
print("Document Type:", c_res.document_type.value)
print("Validation Status:", c_res.validation_status.value)
print("Overall Confidence:", c_res.overall_confidence)
print("Needs Review:", c_res.requires_human_review)

fields_extracted = {}
for k, v in c_res.fields.items():
    val = v.normalized_value or v.raw_value
    fields_extracted[k] = str(val) if val else "Requires verification"
    print(f"  {k}: {fields_extracted[k]} (conf={v.confidence:.2f})")

# Step 3: Save Full Audit Artifact
audit_data = {
    "document": "doc1.jpeg",
    "status": response.status,
    "document_confidence": response.document_confidence,
    "requires_review": response.requires_human_review,
    "regions_count": len(response.ordered_regions),
    "engine_breakdown": response.engine_breakdown,
    "original_kannada_text": response.original_kannada_text or response.merged_text,
    "translated_text": response.translated_text,
    "extracted_fields": fields_extracted,
    "warnings": response.warnings,
}

with open(r"c:\Land Record\scratch\doc1_real_pipeline_result.json", "w", encoding="utf-8") as f:
    json.dump(audit_data, f, indent=2, ensure_ascii=False)

print("\nArtifact saved to scratch/doc1_real_pipeline_result.json")
