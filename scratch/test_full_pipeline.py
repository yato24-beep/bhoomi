import os
import sys
from pathlib import Path

os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from PIL import Image

repo_root = Path("c:/Land Record")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.integration.document_pipeline import DocumentProcessingPipeline
from src.integration.person_c_adapter import PersonCAdapter

doc2_path = Path("C:/Users/achyu/Downloads/doc2.jpeg")
with open(doc2_path, "rb") as f:
    raw_bytes = f.read()

img = Image.open(doc2_path)
print("Image size:", img.size)

pipeline = DocumentProcessingPipeline()
resp = pipeline.process_document(
    image=img,
    document_id="doc2",
    image_path="doc2.jpeg",
    apply_preprocessing=True,
    apply_normalization=True,
)

print("\n--- Pipeline Response ---")
print("Document ID:", resp.document_id)
print("Regions Count:", len(resp.ordered_regions))
print("Engine Breakdown:", resp.engine_breakdown)
print("Document Confidence:", resp.document_confidence)
print("Requires Human Review:", resp.requires_human_review)
print("Warnings:", resp.warnings)
print("Merged Text Length:", len(resp.merged_text))

with open("c:/Land Record/scratch/doc2_full_pipeline_merged_text.txt", "w", encoding="utf-8") as f:
    f.write(resp.merged_text)

# Execute Person C
print("\n--- Executing Person C ---")
c_res = PersonCAdapter.execute_person_c(
    b_response=resp,
    file_bytes=raw_bytes,
)

print("Person C Validation Status:", c_res.validation_status.value)
print("Person C Overall Confidence:", c_res.overall_confidence)
print("Person C Document Type:", c_res.document_type.value)
print("Person C Requires Human Review:", c_res.requires_human_review)
print("Person C Review Reasons:", c_res.review_reasons)
print(f"Person C Extracted Fields Count: {len(c_res.fields)}")
for fname, fval in c_res.fields.items():
    print(f"  Field '{fname}': raw='{fval.raw_value}', norm='{fval.normalized_value}', conf={fval.confidence:.4f}")
