import os
import sys
import traceback
from PIL import Image
import numpy as np

repo_root = r"c:\Land Record"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.preprocessing.image_enhancement import preprocess_document_image
from src.integration.document_pipeline import DocumentProcessingPipeline
from src.handwriting.paddle_recognizer import PaddleKannadaRecognizer

doc1_path = r"c:\Land Record\storage\uploads\48afaad54543e31e_doc1.jpeg"
doc2_path = r"c:\Land Record\storage\uploads\6779632e1e2a023e_doc2.jpeg"

for label, p in [("DOC1 (attached in prompt)", doc1_path), ("DOC2", doc2_path)]:
    print(f"\n==================== DEBUGGING {label}: {p} ====================", flush=True)
    if not os.path.exists(p):
        print(f"File not found: {p}", flush=True)
        continue

    img = Image.open(p)
    print(f"1. Original image format={img.format}, mode={img.mode}, size={img.size}", flush=True)
    
    # Preprocessing
    try:
        prep_res = preprocess_document_image(img, apply_thresholding=False)
        prep_img = prep_res.image
        print(f"2. Preprocessed image size={prep_img.size}, mode={prep_img.mode}", flush=True)
        print(f"   Audit metadata: {prep_res.audit_metadata}", flush=True)
        
        # Check if preprocessed image is blank
        arr = np.array(prep_img)
        print(f"3. Image array: shape={arr.shape}, min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}, std={arr.std():.2f}", flush=True)
    except Exception as e:
        print("Preprocessing error:", e, flush=True)
        traceback.print_exc()

    # Test PaddleKannadaRecognizer directly on original and preprocessed
    print("\n4. Testing PaddleKannadaRecognizer on preprocessed image:", flush=True)
    rec = PaddleKannadaRecognizer()
    try:
        paddle_res = rec.recognize_handwriting(prep_img)
        print(f"   PaddleOCR result on prep_img: text_len={len(paddle_res.text or '')}, conf={paddle_res.confidence}", flush=True)
        print(f"   Sample text: {repr((paddle_res.text or '')[:200])}", flush=True)
        meta = paddle_res.metadata.get("metadata", {})
        line_details = meta.get("line_details", [])
        print(f"   Detected lines count: {len(line_details)}", flush=True)
        for i, l in enumerate(line_details[:5]):
            print(f"     Line {i}: {l.get('text')} (conf={l.get('confidence')}) bbox={l.get('bbox')}", flush=True)
    except Exception as e:
        print("PaddleOCR error:", e, flush=True)
        traceback.print_exc()

    # Also test on original raw image directly
    print("\n4b. Testing PaddleKannadaRecognizer on RAW UNPROCESSED image:", flush=True)
    try:
        paddle_raw = rec.recognize_handwriting(img)
        print(f"   PaddleOCR result on RAW img: text_len={len(paddle_raw.text or '')}, conf={paddle_raw.confidence}", flush=True)
        print(f"   Sample text: {repr((paddle_raw.text or '')[:200])}", flush=True)
        meta = paddle_raw.metadata.get("metadata", {})
        line_details = meta.get("line_details", [])
        print(f"   Detected lines count: {len(line_details)}", flush=True)
    except Exception as e:
        print("PaddleOCR raw error:", e, flush=True)
        traceback.print_exc()

    # Also test with rotation 90, 180, 270 degrees
    print("\n4c. Testing PaddleKannadaRecognizer with Rotations (90, 180, 270):", flush=True)
    for rot in [90, 180, 270]:
        rot_img = img.rotate(rot, expand=True)
        try:
            paddle_rot = rec.recognize_handwriting(rot_img)
            meta = paddle_rot.metadata.get("metadata", {})
            lines = meta.get("line_details", [])
            print(f"   Rotation {rot} deg: detected_lines={len(lines)}, text_len={len(paddle_rot.text or '')}, conf={paddle_rot.confidence}", flush=True)
            if lines:
                print(f"     First line: {lines[0].get('text')} ({lines[0].get('confidence')})", flush=True)
        except Exception as e:
            print(f"   Rotation {rot} error: {e}", flush=True)
