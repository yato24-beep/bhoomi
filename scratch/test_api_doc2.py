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
try:
    data = response.json()
    print("=== OCR Pipeline Response Summary ===")
    print(f"Overall Confidence: {data.get('confidence')}")
    print(f"Regions Count: {data.get('regions_count')}")
    print(f"Engine Breakdown: {data.get('engine_breakdown')}")
    print(f"Warnings: {data.get('warnings')}")
    print(f"Merged Text Sample (first 300 chars):\n{data.get('merged_text', '')[:300]}")
    print("\n=== Person C Extraction (data) ===")
    ext_data = data.get("data", {})
    print(json.dumps(ext_data, indent=2, ensure_ascii=False))
    
    with open("doc2_api_result.json", "w", encoding="utf-8") as out_f:
        json.dump(data, out_f, indent=2, ensure_ascii=False)
    print("\nSaved full response to doc2_api_result.json")
except Exception as e:
    print(f"Error parsing response: {e}")
    print(response.text)
