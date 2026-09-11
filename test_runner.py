import sys
import io
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"c:\Land Record")
import src
from src.integration.document_pipeline import DocumentProcessingPipeline
from src.integration.person_c_adapter import PersonCAdapter

pipeline = DocumentProcessingPipeline()
resp = pipeline.process_document(image="doc2.jpeg")

print("--- PERSON B OCR RESPONSE ---")
print("Status:", resp.status)
print("Document Confidence:", resp.document_confidence)
print("Regions Count:", len(resp.ordered_regions))
print("Requires Human Review:", resp.requires_human_review)
print("Warnings:", resp.warnings[:3])
print("Original Kannada Text Sample:", (resp.original_kannada_text or "")[:120].encode("unicode_escape").decode("ascii"))
print("Translated Text Sample:", (resp.translated_text or "")[:120].encode("unicode_escape").decode("ascii"))

with open("doc2.jpeg", "rb") as f:
    raw_bytes = f.read()

c_res = PersonCAdapter.execute_person_c(b_response=resp, file_bytes=raw_bytes)
print("--- PERSON C EXTRACTED FIELDS ---")
print("Doc Type:", c_res.document_type.value)
print("Overall Confidence:", c_res.overall_confidence)
for k, v in c_res.fields.items():
    print(f"  {k}: raw='{v.raw_value}' norm='{v.normalized_value}' conf={v.confidence:.2f}")

# Test PDF and DOCX exports
from backend.app.services.export import build_pdf_export, build_docx_export
fields_dict = {k: str(v.normalized_value or v.raw_value) for k, v in c_res.fields.items()}
pdf = build_pdf_export(
    document_id="doc_2",
    filename="doc2.jpeg",
    overall_confidence=float(c_res.overall_confidence),
    status=c_res.validation_status.value,
    fields=fields_dict,
    kannada_text=resp.original_kannada_text or resp.merged_text,
    english_translation=resp.translated_text or resp.merged_text,
    requires_review=c_res.requires_human_review,
    review_reasons=c_res.review_reasons,
)
docx = build_docx_export(
    document_id="doc_2",
    filename="doc2.jpeg",
    overall_confidence=float(c_res.overall_confidence),
    status=c_res.validation_status.value,
    fields=fields_dict,
    kannada_text=resp.original_kannada_text or resp.merged_text,
    english_translation=resp.translated_text or resp.merged_text,
    requires_review=c_res.requires_human_review,
    review_reasons=c_res.review_reasons,
)
with open("test_out.pdf", "wb") as f:
    f.write(pdf)
with open("test_out.docx", "wb") as f:
    f.write(docx)
print(f"Export Success! Generated test_out.pdf ({len(pdf)} bytes) and test_out.docx ({len(docx)} bytes)")
