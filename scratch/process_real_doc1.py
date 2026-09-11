"""End-to-End Real Processing and Evaluation Script for doc1.jpeg.

Executes the complete pipeline:
1. Input doc1.jpeg
2. Preprocessing & automatic orientation correction (sideways 90deg photo -> landscape 1600x900)
3. Line segmentation (DocumentLineSegmenter)
4. Line-level multi-pass Kannada TrOCR recognition
5. Reading-order reconstruction
6. Translation to English via domain glossary + neural fallback
7. Person C structured land record extraction (Survey number, owner, village, etc.)
8. PDF and DOCX digital document generation
9. Full audit report
"""

import io
import json
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output encoding
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KANNADA_HANDWRITING_MODEL_PATH"] = r"models\trocr\kannada_full_checkpoints\checkpoint-step-55390"

# Fix torch DLL loading on Windows
import site
for sp in site.getsitepackages() + [site.getusersitepackages()]:
    tlib = os.path.join(sp, 'torch', 'lib')
    if os.path.isdir(tlib):
        try:
            os.add_dll_directory(tlib)
        except Exception:
            pass
        os.environ['PATH'] = tlib + ';' + os.environ.get('PATH', '')

repo_root = Path(r"c:\Land Record")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from PIL import Image
from src.integration.document_pipeline import DocumentProcessingPipeline
from src.integration.person_c_adapter import PersonCAdapter
from backend.app.services.export import build_pdf_export, build_docx_export

def main():
    print("=" * 80, flush=True)
    print("  LAND RECORD DIGITIZATION — END-TO-END PROCESSING OF DOC1.JPEG", flush=True)
    print("=" * 80, flush=True)

    doc_path = repo_root / "doc1.jpeg"
    if not doc_path.exists():
        print(f"Error: {doc_path} not found!", flush=True)
        sys.exit(1)

    print(f"1. Loading original document: {doc_path}", flush=True)
    raw_img = Image.open(doc_path)
    print(f"   Original dimensions: {raw_img.size} | Mode: {raw_img.mode}", flush=True)

    with open(doc_path, "rb") as f:
        file_bytes = f.read()

    print("\n2. Executing DocumentProcessingPipeline...", flush=True)
    start_time = time.perf_counter()

    pipeline = DocumentProcessingPipeline(
        default_language="kannada",
        confidence_threshold=0.60,
        apply_preprocessing=True,
        apply_normalization=True,
    )

    resp = pipeline.process_document(
        image=raw_img,
        document_id="doc1_real_test",
        image_path=str(doc_path),
        apply_preprocessing=True,
        apply_normalization=True,
    )
    pipeline_duration = time.perf_counter() - start_time
    print(f"   Pipeline completed in {pipeline_duration:.2f}s", flush=True)

    # Summary of OCR
    regions_count = len(resp.ordered_regions)
    avg_conf = resp.document_confidence or 0.0
    engine_bd = resp.engine_breakdown
    kannada_text = resp.original_kannada_text or resp.merged_text or ""
    english_trans = resp.translated_text or ""
    requires_review = resp.requires_human_review
    review_warnings = resp.warnings

    print(f"\n3. OCR Results Summary:", flush=True)
    print(f"   Total segmented regions/lines : {regions_count}", flush=True)
    print(f"   Overall confidence            : {avg_conf:.4f} ({avg_conf*100:.1f}%)", flush=True)
    print(f"   Engine breakdown              : {engine_bd}", flush=True)
    print(f"   Requires human review         : {requires_review}", flush=True)
    print(f"   Warnings count                : {len(review_warnings)}", flush=True)

    print("\n4. Running Person C Structured Land Record Extraction...", flush=True)
    c_result = None
    extracted_fields = {}
    try:
        c_result = PersonCAdapter.execute_person_c(
            b_response=resp,
            file_bytes=file_bytes,
        )
        for fname, fval in c_result.fields.items():
            extracted_fields[fname] = str(fval.normalized_value or fval.raw_value or "")
        print(f"   Extracted {len(extracted_fields)} cadastral fields via Person C", flush=True)
        for k, v in extracted_fields.items():
            print(f"     • {k}: {v}", flush=True)
    except Exception as exc:
        print(f"   Person C extraction warning: {exc}", flush=True)

    print("\n5. Generating Real Digital Documents (PDF & Word DOCX)...", flush=True)
    export_dir = repo_root / "storage" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    export_payload = {
        "document_id": "doc1_hackathon",
        "filename": "doc1.jpeg",
        "overall_confidence": avg_conf,
        "status": "COMPLETED",
        "fields": extracted_fields,
        "kannada_text": kannada_text,
        "english_translation": english_trans,
        "requires_review": requires_review,
        "review_reasons": review_warnings,
    }

    # Generate PDF
    pdf_bytes = build_pdf_export(**export_payload)
    pdf_path = export_dir / "doc1_digitized_report.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"   PDF generated: {pdf_path} ({len(pdf_bytes):,} bytes)", flush=True)

    # Generate DOCX
    docx_bytes = build_docx_export(**export_payload)
    docx_path = export_dir / "doc1_digitized_report.docx"
    with open(docx_path, "wb") as f:
        f.write(docx_bytes)
    print(f"   DOCX generated: {docx_path} ({len(docx_bytes):,} bytes)", flush=True)

    # Generate Kannada & English text files
    kn_path = export_dir / "doc1_transcription_kannada.txt"
    with open(kn_path, "w", encoding="utf-8") as f:
        f.write(kannada_text)
    print(f"   Kannada TXT generated: {kn_path}", flush=True)

    en_path = export_dir / "doc1_translation_english.txt"
    with open(en_path, "w", encoding="utf-8") as f:
        f.write(english_trans)
    print(f"   English TXT generated: {en_path}", flush=True)

    # Save complete JSON result
    json_path = export_dir / "doc1_full_result.json"
    full_json = {
        "document_id": "doc1_hackathon",
        "filename": "doc1.jpeg",
        "overall_confidence": avg_conf,
        "regions_count": regions_count,
        "engine_breakdown": engine_bd,
        "requires_human_review": requires_review,
        "review_warnings": review_warnings,
        "extracted_fields": extracted_fields,
        "kannada_transcription": kannada_text,
        "english_translation": english_trans,
        "pdf_path": str(pdf_path),
        "docx_path": str(docx_path),
        "timing_ms": int(pipeline_duration * 1000),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False)
    print(f"   Full JSON result saved to: {json_path}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("  KANNADA DIGITAL TRANSCRIPTION (PREVIEW):", flush=True)
    print("=" * 80, flush=True)
    print(kannada_text[:600] if kannada_text else "[Empty]", flush=True)
    if len(kannada_text) > 600:
        print(f"... ({len(kannada_text) - 600} more characters)", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("  ENGLISH TRANSLATION (PREVIEW):", flush=True)
    print("=" * 80, flush=True)
    print(english_trans[:600] if english_trans else "[Empty]", flush=True)
    if len(english_trans) > 600:
        print(f"... ({len(english_trans) - 600} more characters)", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("  REAL DOC1 PROCESSING COMPLETE", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
