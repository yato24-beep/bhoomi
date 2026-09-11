import os
import sys
import time
from PIL import Image
import io

sys.path.insert(0, r"c:\Land Record")
sys.path.insert(0, r"c:\Land Record\backend")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.pipeline.processor import get_document_processor

def test_doc(img_name):
    print("\n" + "="*80)
    print(f"TESTING BACKEND PROCESSOR ON: {img_name}")
    print("="*80)
    
    img_path = os.path.join(r"c:\Land Record", img_name)
    with open(img_path, "rb") as f:
        file_bytes = f.read()
    
    t0 = time.perf_counter()
    processor = get_document_processor()
    result = processor.process(
        file_stream=io.BytesIO(file_bytes),
        filename=img_name,
        content_type="image/jpeg",
    )
    duration = time.perf_counter() - t0
    
    print(f"Result in {duration:.2f}s:")
    print(f"  • Confidence: {result.confidence_score}")
    print(f"  • Is Valid: {result.is_valid}")
    print(f"  • Processing Time ms: {result.processing_time_ms}")
    print(f"  • Total Fields Extracted: {len(result.fields)}")
    print(f"  • Document Type: {result.extracted_data.get('document_type')}")
    print(f"  • Engine Breakdown: {result.extracted_data.get('engine_breakdown')}")
    print(f"  • Warnings: {result.validation_info.get('warnings')}")
    
    print("\n--- First 8 Extracted Fields ---")
    for f in result.fields[:8]:
        print(f"  [{f.field_name}] raw: '{f.original_value}' | norm: '{f.normalized_value}' (conf={f.confidence_score:.2f})")

if __name__ == "__main__":
    test_doc("doc1.jpeg")
    test_doc("doc2.jpeg")
