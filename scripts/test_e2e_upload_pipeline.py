"""
End-to-End Upload, OCR, Person-C Pipeline & Verification Test Script
"""
import os
import sys
import json
import time
import urllib.request
import urllib.parse
from PIL import Image

sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_test():
    print("=" * 80)
    print("LAND RECORD DIGITIZATION — END-TO-END PIPELINE TEST")
    print("=" * 80)
    
    # 1. Login as Admin
    login_url = "http://localhost:8000/api/v1/auth/login"
    login_data = json.dumps({"email": "admin@docplatform.com", "password": "admin123"}).encode("utf-8")
    req = urllib.request.Request(login_url, data=login_data, headers={"Content-Type": "application/json"})
    
    with urllib.request.urlopen(req, timeout=10) as res:
        auth_resp = json.loads(res.read().decode("utf-8"))
        token = auth_resp["access_token"]
        print(f"[✓] Authenticated as Admin (token: {token[:20]}...)")
        
    # 2. Upload sample document (doc1.jpeg)
    upload_url = "http://localhost:8000/api/v1/documents/upload"
    doc_path = r"c:\Land Record\doc1.jpeg"
    if not os.path.exists(doc_path):
        doc_path = r"c:\Land Record\doc2.jpeg"
        
    print(f"[*] Uploading test document: {doc_path} ({os.path.getsize(doc_path)} bytes)")
    
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(doc_path, "rb") as f:
        file_bytes = f.read()
        
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="file"; filename="doc1.jpeg"\r\n')
    body.extend(b'Content-Type: image/jpeg\r\n\r\n')
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    
    req = urllib.request.Request(
        upload_url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {token}"
        }
    )
    
    with urllib.request.urlopen(req, timeout=30) as res:
        upload_res = json.loads(res.read().decode("utf-8"))
        doc_info = upload_res.get("document", {})
        doc_id = doc_info.get("id")
        print(f"[✓] Document Uploaded Successfully! (Document ID: {doc_id})")
        print(f"    Initial Status: {doc_info.get('status')}")
        
    # 3. Poll for status
    status_url = f"http://localhost:8000/api/v1/documents/{doc_id}/status"
    print("\n[*] Polling document processing pipeline...")
    
    final_status = None
    for attempt in range(40):
        time.sleep(1)
        req = urllib.request.Request(status_url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as res:
                status_data = json.loads(res.read().decode("utf-8"))
                st = status_data.get("status")
                print(f"    Attempt {attempt+1}: Status = {st}")
                if st in ("COMPLETED", "PROCESSED", "REVIEW_REQUIRED", "FAILED"):
                    final_status = st
                    break
        except Exception as e:
            print(f"    Attempt {attempt+1} notice: {e}")
                
    # 4. Fetch document record
    doc_url = f"http://localhost:8000/api/v1/documents/{doc_id}"
    req = urllib.request.Request(doc_url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=10) as res:
        doc_record = json.loads(res.read().decode("utf-8"))
        
    # 5. Fetch extraction results
    results_url = f"http://localhost:8000/api/v1/documents/{doc_id}/results"
    req = urllib.request.Request(results_url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=10) as res:
        results_data = json.loads(res.read().decode("utf-8"))
        
    # 6. Fetch extracted fields
    fields_url = f"http://localhost:8000/api/v1/documents/{doc_id}/fields"
    req = urllib.request.Request(fields_url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=10) as res:
        fields_data = json.loads(res.read().decode("utf-8"))
        
    print("\n" + "=" * 80)
    print("DOCUMENT PROCESSING VERIFICATION REPORT")
    print("=" * 80)
    print(f"Document ID:             {doc_id}")
    print(f"Filename:                {doc_record.get('filename')}")
    print(f"Final Status:            {doc_record.get('status')}")
    print(f"Overall Confidence:      {results_data.get('confidence_score') * 100:.1f}%")
    print(f"Is Valid:                {results_data.get('is_valid')}")
    
    val_info = results_data.get("validation_info", {})
    print(f"Validation Status:       {val_info.get('validation_status', 'N/A')}")
    print(f"GIS Verified:            {val_info.get('gis_verified', 'N/A')}")
    print(f"Requires Human Review:   {val_info.get('requires_human_review', False)}")
    print(f"Warnings:                {val_info.get('warnings', [])}")
    
    extracted_data = results_data.get("extracted_data", {})
    kannada_text = extracted_data.get("original_kannada_text") or extracted_data.get("merged_text") or ""
    translated_text = extracted_data.get("translated_text") or ""
    
    print(f"\n--- Extracted Kannada OCR Text (Sample) ---")
    print(kannada_text[:400] if kannada_text else "[None]")
    
    print(f"\n--- English Translation Text (Sample) ---")
    print(translated_text[:400] if translated_text else "[None]")
    
    fields_list = fields_data.get("fields", [])
    print(f"\n--- Person-C Extracted Structured Fields ({len(fields_list)} total) ---")
    for f_item in fields_list[:12]:
        print(f"  • {f_item.get('field_name')}: '{f_item.get('normalized_value') or f_item.get('original_value')}' (Conf: {f_item.get('confidence_score')*100:.1f}%)")
        
    has_kannada = bool(kannada_text and len(kannada_text.strip()) > 5)
    has_fields = bool(len(fields_list) > 0)
    
    print("\n" + "=" * 80)
    print("SUMMARY VERIFICATION CHECKLIST")
    print("=" * 80)
    print(f"UPLOAD TEST:             PASS")
    print(f"OCR PROCESSING TEST:     {'PASS' if has_kannada or final_status == 'COMPLETED' else 'FAIL'}")
    print(f"KANNADA/ENGLISH TEXT:    {'PASS' if bool(kannada_text or translated_text) else 'FAIL'}")
    print(f"PERSON-C EXTRACTION:     {'PASS' if has_fields else 'FAIL'}")
    print(f"FINAL RESULTS PAGE API:  PASS")
    print("=" * 80)
    return True

if __name__ == "__main__":
    run_test()
