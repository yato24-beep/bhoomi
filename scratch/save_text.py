import json

with open("doc2_api_result.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("doc2_merged_text.txt", "w", encoding="utf-8") as out_f:
    out_f.write(data.get("merged_text", ""))

print("Wrote doc2_merged_text.txt. Total length:", len(data.get("merged_text", "")))
