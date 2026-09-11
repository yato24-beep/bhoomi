import os
import requests
import json
import time

base_url = "http://127.0.0.1:8000"
file_path = r"C:\Users\achyu\Downloads\doc2.jpeg"

# 1. Upload doc2.jpeg
print("Uploading doc2.jpeg to backend...")
with open(file_path, "rb") as f:
    files = {"file": ("doc2.jpeg", f, "image/jpeg")}
    upload_resp = requests.post(f"{base_url}/api/v1/documents/upload", files=files)

print("Upload response status:", upload_resp.status_code)
if upload_resp.status_code not in (200, 201):
    print("Upload failed:", upload_resp.text)
    exit(1)

upload_data = upload_resp.json()
doc = upload_data.get("document", {})
doc_id = doc.get("id")
print(f"Document ID: {doc_id}, initial status: {doc.get('status')}")

# 2. Wait for processing
max_wait = 60
start_t = time.time()
while time.time() - start_t < max_wait:
    res_resp = requests.get(f"{base_url}/api/v1/documents/{doc_id}")
    if res_resp.status_code == 200:
        d_status = res_resp.json().get("status")
        print(f"Polling doc status: {d_status} (elapsed: {int(time.time() - start_t)}s)")
        if d_status in ("COMPLETED", "FAILED"):
            break
    time.sleep(3)

# 3. Fetch full results
results_resp = requests.get(f"{base_url}/api/v1/documents/{doc_id}/results")
print("\n--- Document Results Endpoint ---")
print("Results HTTP status:", results_resp.status_code)
if results_resp.status_code == 200:
    res = results_resp.json()
    print("Result Confidence:", res.get("confidence_score"))
    print("Result is_valid:", res.get("is_valid"))
    print("Result processing_time_ms:", res.get("processing_time_ms"))
    extracted = res.get("extracted_data", {})
    print("Merged text length:", len(extracted.get("merged_text", "")))
    print("Regions count:", extracted.get("regions_count"))
    print("Engine breakdown:", extracted.get("engine_breakdown"))
    print("Status:", extracted.get("status"))
    print("Warnings:", extracted.get("warnings"))
    
    fields = res.get("fields", [])
    print(f"Total Extracted Fields: {len(fields)}")
    for f in fields[:10]:
        print(f"  Field: {f.get('field_name')}, value: {f.get('original_value')}, conf: {f.get('confidence_score')}")

    # Save to file
    with open("c:/Land Record/scratch/backend_doc_results.json", "w", encoding="utf-8") as out:
        json.dump(res, out, indent=2, ensure_ascii=False)
    print("Saved to c:/Land Record/scratch/backend_doc_results.json")
