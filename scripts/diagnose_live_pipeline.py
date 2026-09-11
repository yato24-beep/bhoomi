"""
Detailed Diagnostic of Live Processing Pipeline on doc1.jpeg and doc2.jpeg
Measures time of each stage:
1. Load & Preprocessing (Coarse Orientation, Upscaling, CLAHE, Deskew)
2. Line Segmentation (Paddle line detection vs morphology)
3. Handwriting Routing & TrOCR Recognition (Generalized Checkpoint)
4. Person-C Field Extraction & Normalization
"""
import os
import sys
import time
from PIL import Image

sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.integration.document_pipeline import DocumentProcessingPipeline
from src.integration.person_c_adapter import PersonCAdapter
from src.preprocessing.image_enhancement import preprocess_document_image
from src.preprocessing.line_segmentation import DocumentLineSegmenter

def diagnose_doc(img_path):
    print("\n" + "=" * 80)
    print(f"DIAGNOSING PIPELINE FOR: {img_path}")
    print("=" * 80)
    
    img = Image.open(img_path)
    print(f"Original size: {img.size}")
    
    # 1. Preprocessing stage
    t0 = time.perf_counter()
    prep_res = preprocess_document_image(img, apply_thresholding=False)
    t_prep = time.perf_counter() - t0
    print(f"[Stage 1] Preprocessing took {t_prep:.2f}s:")
    print(f"  Processed size: {prep_res.image.size}")
    print(f"  Audit: {prep_res.audit_metadata.get('pipeline_steps')}")
    print(f"  Coarse Orientation: {prep_res.audit_metadata.get('coarse_orientation')}")
    
    # 2. Line Segmentation stage
    t0 = time.perf_counter()
    segmenter = DocumentLineSegmenter()
    lines = segmenter.segment_into_lines(prep_res.image, page_number=1)
    t_seg = time.perf_counter() - t0
    print(f"\n[Stage 2] Line Segmentation took {t_seg:.2f}s:")
    print(f"  Total lines discovered: {len(lines)}")
    if lines:
        print(f"  Sample line bboxes: {[l.bbox for l in lines[:3]]}")
        
    # 3. Full DocumentProcessingPipeline
    t0 = time.perf_counter()
    pipeline = DocumentProcessingPipeline()
    resp = pipeline.process_document(
        image=img,
        document_id=os.path.basename(img_path),
        apply_preprocessing=True,
        apply_normalization=True,
    )
    t_pipe = time.perf_counter() - t0
    print(f"\n[Stage 3] Full Document Pipeline took {t_pipe:.2f}s:")
    print(f"  Total ordered regions: {len(resp.ordered_regions)}")
    print(f"  Document confidence: {resp.document_confidence}")
    print(f"  Requires human review: {resp.requires_human_review}")
    print(f"  Engine breakdown: {resp.engine_breakdown}")
    
    print("\n--- First 5 Recognized Line Crops ---")
    for r in resp.ordered_regions[:5]:
        print(f"  • [{r.region_id}] lang={r.language}, model={r.model_name}, conf={r.confidence:.2f}: '{r.normalized_text}'")
        
    # 4. Person C Execution
    t0 = time.perf_counter()
    with open(img_path, "rb") as f:
        file_bytes = f.read()
    c_res = PersonCAdapter.execute_person_c(b_response=resp, file_bytes=file_bytes)
    t_c = time.perf_counter() - t0
    print(f"\n[Stage 4] Person-C Extraction took {t_c:.2f}s:")
    print(f"  Validation status: {c_res.validation_status.value}")
    print(f"  Overall confidence: {c_res.overall_confidence}")
    print(f"  Extracted fields count: {len(c_res.fields)}")
    for fname, fval in list(c_res.fields.items())[:5]:
        print(f"    - {fname}: '{fval.normalized_value or fval.raw_value}' (conf: {fval.confidence:.2f})")
        
    total_time = t_pipe + t_c
    print(f"\n>>> Total Processing Time: {total_time:.2f}s <<<")
    return resp, c_res

if __name__ == "__main__":
    diagnose_doc("doc1.jpeg")
    if os.path.exists("doc2.jpeg"):
        diagnose_doc("doc2.jpeg")
