import json

with open("doc2_api_result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys()))
print("Status:", data.get("status"))
print("Confidence:", data.get("confidence"), data.get("overall_confidence"))
print("Engine breakdown:", data.get("engine_breakdown"))
print("Warnings:", data.get("warnings"))
print("Regions count:", len(data.get("regions", [])) if "regions" in data else data.get("regions_count"))
print("Extracted Data:", json.dumps(data.get("data", {}), indent=2, ensure_ascii=True))
print("Person C Status:", data.get("data", {}).get("status") if "data" in data else None)
print("\nFirst 500 chars of merged_text (escaped):")
print(ascii(data.get("merged_text", "")[:500]))
