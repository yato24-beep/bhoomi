import requests
import json
import os

url = "http://127.0.0.1:8000/api/ocr/process"
file_path = "doc2.jpeg"

print(f"Testing live API POST {url} with {file_path}...")
with open(file_path, "rb") as f:
    files = {"file": (os.path.basename(file_path), f, "image/jpeg")}
    response = requests.post(url, files=files, timeout=60)

print(f"Status Code: {response.status_code}")
data = response.json()
with open("doc2_api_result.json", "w", encoding="utf-8") as out_f:
    json.dump(data, out_f, indent=2, ensure_ascii=False)

print("Saved response to doc2_api_result.json")
print("Top-level keys:", list(data.keys()))
for k in ["status", "success", "confidence", "overall_confidence", "warnings", "engine_breakdown"]:
    if k in data:
        print(f"{k}: {data[k]}")

if "regions" in data:
    print(f"Number of regions: {len(data['regions'])}")
    for i, r in enumerate(data['regions'][:5]):
        print(f"  Region {i}: engine={r.get('engine')} lang={r.get('language')} conf={r.get('confidence')} text={repr(r.get('text'))}")

print("Person C data keys:", list(data.get("data", {}).keys()))
print("Person C data:")
print(json.dumps(data.get("data", {}), indent=2, ensure_ascii=True))
