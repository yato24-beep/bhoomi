import json

with open("doc2_api_result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("=== OCR Pipeline Results ===")
print("document_confidence:", data.get("document_confidence"))
print("overall_confidence:", data.get("overall_confidence"))
print("requires_human_review:", data.get("requires_human_review"))
print("state:", data.get("state"))
print("document_type:", data.get("document_type"))
print("engine_breakdown:", data.get("engine_breakdown"))
print("pipeline_stages_completed:", data.get("pipeline_stages_completed"))

print("\n=== Extracted Fields ===")
print(json.dumps(data.get("extracted_fields", {}), indent=2, ensure_ascii=False))

print("\n=== Validation ===")
print(json.dumps(data.get("validation", {}), indent=2, ensure_ascii=False))

print("\n=== Merged Text (First 600 chars) ===")
print(data.get("merged_text", "")[:600])

print("\n=== Total Merged Text Length ===")
print(len(data.get("merged_text", "")))
