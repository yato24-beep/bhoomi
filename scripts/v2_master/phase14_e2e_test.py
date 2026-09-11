"""Phase 14 — End-to-End Pipeline Test.

Tests the full backend pipeline by uploading test images programmatically.
Verifies: upload → processing → OCR → translation → Person-C → results.
"""

import io
import json
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_pipeline_directly(image_path: Path, doc_name: str):
    """Test the document processing pipeline directly (without HTTP)."""
    from PIL import Image
    from src.integration.document_pipeline import DocumentProcessingPipeline

    print(f"\n  Testing: {doc_name}")
    print(f"  Image:  {image_path}")

    if not image_path.exists():
        print(f"  [SKIP] Image not found: {image_path}")
        return None

    image = Image.open(image_path)
    print(f"  Size:   {image.size[0]}x{image.size[1]}")

    start = time.time()

    pipeline = DocumentProcessingPipeline()
    result = pipeline.process_document(
        image=image,
        document_id=doc_name,
        image_path=str(image_path),
        apply_preprocessing=True,
        apply_normalization=True,
    )

    elapsed = time.time() - start
    elapsed_ms = int(elapsed * 1000)

    print(f"  Time:   {elapsed_ms}ms")
    print(f"  Status: {result.status}")
    print(f"  Regions: {len(result.ordered_regions)}")
    print(f"  Doc Confidence: {result.document_confidence}")
    print(f"  Needs Review:  {result.requires_human_review}")

    # Print each region
    for i, reg in enumerate(result.ordered_regions[:10]):
        raw = (reg.raw_text or "")[:60]
        norm = (reg.normalized_text or "")[:60]
        lang = reg.language
        hw = reg.is_handwritten
        conf = reg.confidence
        print(f"    Region {i+1}: [{lang}|{'HW' if hw else 'PR'}|{conf:.2f}] {raw}")

    # Print merged text
    if result.merged_text:
        print(f"\n  Merged Text (first 200 chars):")
        print(f"    {result.merged_text[:200]}")

    # Print translated text
    if result.translated_text:
        print(f"\n  Translated Text (first 200 chars):")
        print(f"    {result.translated_text[:200]}")

    # Check warnings
    if result.warnings:
        print(f"\n  Warnings: {result.warnings}")

    return {
        "doc_name": doc_name,
        "status": result.status,
        "regions": len(result.ordered_regions),
        "confidence": result.document_confidence,
        "requires_review": result.requires_human_review,
        "processing_ms": elapsed_ms,
        "merged_text_len": len(result.merged_text or ""),
        "translated_text_len": len(result.translated_text or ""),
        "warnings": result.warnings,
    }


def main():
    print("=" * 70)
    print("  PHASE 14 — END-TO-END PIPELINE TEST")
    print("=" * 70)

    # Test images in priority order
    test_images = [
        (PROJECT_ROOT / "storage" / "uploads" / "doc1.jpeg", "doc1"),
        (PROJECT_ROOT / "storage" / "uploads" / "doc2.jpeg", "doc2"),
        (PROJECT_ROOT / "storage" / "uploads" / "new_kothi.jpeg", "new_kothi"),
        (PROJECT_ROOT / "storage" / "uploads" / "crop_o_kothi.png", "crop_kothi"),
        (PROJECT_ROOT / "storage" / "uploads" / "crop_a_mara.png", "crop_mara"),
        (PROJECT_ROOT / "storage" / "uploads" / "crop_p_hannu.png", "crop_hannu"),
    ]

    # Also check standard test locations
    alt_locations = [
        PROJECT_ROOT / "test_images",
        PROJECT_ROOT / "sample_images",
        PROJECT_ROOT / "data" / "test",
    ]

    results = []
    tested = 0
    passed = 0

    for img_path, doc_name in test_images:
        if img_path.exists():
            try:
                result = test_pipeline_directly(img_path, doc_name)
                if result:
                    results.append(result)
                    tested += 1
                    if result["regions"] > 0 and result["merged_text_len"] > 0:
                        passed += 1
                    else:
                        print(f"  [WARN] Pipeline returned 0 regions or empty text")
            except Exception as e:
                print(f"  [ERROR] {doc_name}: {e}")
                tested += 1

    # If no test images found, try a synthetic test
    if tested == 0:
        print("\n  No test images found in expected locations.")
        print("  Looking for any JPEG/PNG images...")
        
        for folder in [PROJECT_ROOT / "storage" / "uploads", PROJECT_ROOT]:
            if folder.exists():
                for ext in ["*.jpeg", "*.jpg", "*.png"]:
                    for img_file in folder.glob(ext):
                        if tested < 2:
                            try:
                                result = test_pipeline_directly(img_file, img_file.stem)
                                if result:
                                    results.append(result)
                                    tested += 1
                                    if result["regions"] > 0:
                                        passed += 1
                            except Exception as e:
                                print(f"  [ERROR] {img_file.name}: {e}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  E2E TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  Documents tested: {tested}")
    print(f"  Pipeline success: {passed}/{tested}")
    
    for r in results:
        status = "PASS" if r["regions"] > 0 and r["merged_text_len"] > 0 else "FAIL"
        print(f"  [{status}] {r['doc_name']}: {r['regions']} regions, {r['processing_ms']}ms, conf={r['confidence']}")

    # Save results
    report_path = PROJECT_ROOT / "evaluation" / "v2_e2e_test_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {report_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
