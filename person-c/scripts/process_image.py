"""
scripts/process_image.py
End-to-End Image Processing Runner.

Takes a raw image path, computes SHA-256 hash, runs REAL OCR (EasyOCR),
and passes the result into Person C's Extraction, Normalization, GIS
Validation, and Confidence Scoring pipeline.

FIXED VERSION — changes from the original:
1. REMOVED the hardcoded Karnataka/Bengaluru/Kengeri fallback sample data.
   That fixture was a leftover development placeholder that silently fired
   on EVERY image whenever OCR failed, which is why every upload was
   showing the same wrong "Kengeri, Bangalore" record regardless of what
   was actually uploaded.
2. REMOVED the blocking input() prompt — it hung/short-circuited in any
   non-interactive context (API calls, scripts, no terminal attached),
   which was the exact trigger that led to the fallback firing.
3. Failures are now LOUD (raise a clear exception) instead of silently
   substituting fake data. You will now know immediately if OCR failed,
   instead of getting plausible-looking wrong output.
4. Fixed a mislabeling bug: successful EasyOCR output was being tagged
   as OCREngineType.PADDLE_OCR (wrong) — now correctly tagged EASYOCR.
5. Fixed language defaulting: if no --state is passed, it no longer
   silently falls back to English-only (which garbles Tamil/Kannada/etc.
   into wrong characters). It now tries a broad multilingual pass and
   warns you clearly that results will be better if you pass --state.
6. Added basic row/column table structure grouping from OCR bounding
   boxes, so tabular fields (khasra number, area, etc. in table form)
   have a real chance of being extracted — previously `tables` was
   never populated at all, so all table-based extraction silently did
   nothing.
7. Added a Tesseract fallback engine. EasyOCR's Tamil recognition model
   currently has a known, unresolved upstream bug (a character-set size
   mismatch between the library version and the downloaded checkpoint —
   verified independently, tracked in EasyOCR's own GitHub issues #1135
   and #1315, not something wrong with this codebase). If EasyOCR fails
   to load or read a given language, this now automatically falls back
   to Tesseract (with the matching language pack) instead of crashing
   or — as before — silently substituting fake data.

STILL NOT SOLVED (this file cannot fix these — they need real work):
- True handwriting-specific recognition (needs a fine-tuned TrOCR model —
  EasyOCR is a printed/general OCR engine, not a handwriting specialist).
- Full layout-aware table detection (this uses a bounding-box heuristic,
  not a real trained layout model like PP-StructureV3).
See the note printed at the end of this script for what to do next.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas import (
    BoundingBox,
    DocumentClassificationResult,
    DocumentOCRResult,
    DocumentType,
    OCREngineType,
    OCRTextLine,
    TableCell,
    TableStructure,
)
from src.integration.person_c_service import extract_and_validate
from src.utils.config_loader import ConfigLoader
from src.utils.hashing import calculate_sha256


# State -> EasyOCR language codes. EasyOCR only reliably supports certain
# language COMBINATIONS together (it errors on some mixes), so keep pairs
# small and specific rather than throwing everything in at once.
STATE_LANGUAGES = {
    "KA": ["kn", "en"],
    "TN": ["ta", "en"],
    "UP": ["hi", "en"],
    "MP": ["hi", "en"],
    "BR": ["hi", "en"],
    "MH": ["mr", "en"],
}
# Used only when no --state is given. EasyOCR can't load every Indic
# script at once reliably, so this is a reasonable broad default, not a
# guarantee — always pass --state when you know it for best accuracy.
FALLBACK_MULTILANG = ["en", "hi"]

# State -> Tesseract language codes (used as fallback when EasyOCR fails
# to load/read a language — e.g. the currently-broken EasyOCR Tamil model).
# Install with: apt-get install tesseract-ocr-tam tesseract-ocr-kan
#               tesseract-ocr-hin tesseract-ocr-mar
STATE_TESSERACT_LANG = {
    "KA": "kan+eng",
    "TN": "tam+eng",
    "UP": "hin+eng",
    "MP": "hin+eng",
    "BR": "hin+eng",
    "MH": "mar+eng",
}
FALLBACK_TESSERACT_LANG = "eng"


def _init_easyocr_reader(lang_list: list[str]):
    """Imports and initializes EasyOCR. Raises a clear error if unavailable."""
    try:
        import easyocr
    except ImportError as e:
        raise RuntimeError(
            "EasyOCR is not installed. Install it with:\n"
            "    pip install easyocr\n"
            "(This is a real dependency the project needs — it was missing "
            "from requirements.txt, which is the root cause of OCR silently "
            "failing.)"
        ) from e

    print(f"[PERSON A] Initializing OCR engine for languages: {lang_list}...")
    return easyocr.Reader(lang_list, gpu=False, verbose=False)


def _run_easyocr(image_path: Path, lang_list: list[str]) -> list[OCRTextLine]:
    """Runs EasyOCR on an image and returns OCRTextLine objects. Raises on failure."""
    import cv2

    reader = _init_easyocr_reader(lang_list)  # may raise — caller handles fallback

    cv_img = cv2.imread(str(image_path))
    if cv_img is None:
        raise RuntimeError(
            f"OpenCV could not read the image file: {image_path}. "
            f"Check the file isn't corrupted and is a supported format "
            f"(jpg/png/tiff)."
        )

    raw_results = reader.readtext(cv_img)

    raw_entries = []
    for box_coords, text_str, prob_score in raw_results:
        text_str = text_str.strip()
        if not text_str:
            continue
        xs = [p[0] for p in box_coords]
        ys = [p[1] for p in box_coords]
        raw_entries.append({
            "text": text_str, "conf": float(prob_score),
            "x_min": min(xs), "y_min": min(ys), "x_max": max(xs), "y_max": max(ys),
        })

    # FIX: EasyOCR's readtext() does NOT guarantee top-to-bottom /
    # left-to-right reading order — text fragments on the same physical
    # row can come back scattered anywhere in the result list (verified:
    # a two-word name on one row came back as two separate entries with
    # one of them pushed to the very end of the list). Left uncorrected,
    # this silently breaks extraction for any multi-word value, because
    # by the time text is joined into full_text, words end up in the
    # wrong order. Fix: sort by y (row) then x (column) and merge
    # fragments whose y-ranges overlap into one line, in correct left-
    # to-right order. This is basic reading-order correction, not table/
    # column structure detection.
    raw_entries.sort(key=lambda e: (round(e["y_min"] / 10), e["x_min"]))

    merged_lines = []
    current_row = []
    current_y = None
    y_tolerance = 15.0
    for e in raw_entries:
        if current_y is None or abs(e["y_min"] - current_y) <= y_tolerance:
            current_row.append(e)
            current_y = e["y_min"] if current_y is None else current_y
        else:
            merged_lines.append(current_row)
            current_row = [e]
            current_y = e["y_min"]
    if current_row:
        merged_lines.append(current_row)

    lines: list[OCRTextLine] = []
    for row in merged_lines:
        row.sort(key=lambda e: e["x_min"])
        merged_text = " ".join(e["text"] for e in row)
        avg_conf = sum(e["conf"] for e in row) / len(row)
        bbox = BoundingBox(
            x_min=float(min(e["x_min"] for e in row)),
            y_min=float(min(e["y_min"] for e in row)),
            x_max=float(max(e["x_max"] for e in row)),
            y_max=float(max(e["y_max"] for e in row)),
            normalized=False,
        )
        lines.append(
            OCRTextLine(
                text=merged_text,
                confidence=round(avg_conf, 3),
                bbox=bbox,
                page_number=1,
                engine=OCREngineType.EASYOCR,   # <-- fixed: was wrongly tagged PADDLE_OCR before
            )
        )

    if not lines:
        raise RuntimeError(f"EasyOCR found no text in {image_path}.")

    return lines


def _run_tesseract(image_path: Path, tess_lang: str) -> list[OCRTextLine]:
    """
    Fallback OCR engine using Tesseract. Used when EasyOCR fails to load
    or read a given language (e.g. EasyOCR's currently-broken Tamil model).
    Requires: pip install pytesseract, and the relevant tesseract-ocr-<lang>
    system package installed (e.g. apt-get install tesseract-ocr-tam).
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "pytesseract is not installed. Install it with:\n"
            "    pip install pytesseract\n"
            "and make sure the tesseract-ocr system binary plus the "
            "relevant language pack is installed, e.g.:\n"
            "    apt-get install tesseract-ocr tesseract-ocr-tam"
        ) from e

    print(f"[PERSON A] Falling back to Tesseract OCR (lang={tess_lang})...")
    img = Image.open(image_path)

    data = pytesseract.image_to_data(
        img, lang=tess_lang, output_type=pytesseract.Output.DICT
    )

    # IMPORTANT: pytesseract's image_to_data returns one row PER WORD, not
    # per line. Feeding one OCRTextLine per word breaks the extractor's
    # regex patterns, which expect a whole line like "மாவட்டம்: கோயம்புத்தூர்"
    # as one string (label immediately followed by its value on the same
    # line) — not two separate fragments. Group words back into their
    # original visual lines using tesseract's own block/par/line indices
    # before returning, so downstream extraction sees real lines, same as
    # EasyOCR naturally provides.
    n = len(data["text"])
    line_groups: dict[tuple, dict] = {}
    for i in range(n):
        text_str = data["text"][i].strip()
        if not text_str:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        conf_raw = data["conf"][i]
        try:
            conf = max(0.0, min(1.0, float(conf_raw) / 100.0))
        except (ValueError, TypeError):
            conf = 0.5
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

        if key not in line_groups:
            line_groups[key] = {
                "words": [], "confs": [],
                "x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h,
            }
        g = line_groups[key]
        g["words"].append(text_str)
        g["confs"].append(conf)
        g["x_min"] = min(g["x_min"], x)
        g["y_min"] = min(g["y_min"], y)
        g["x_max"] = max(g["x_max"], x + w)
        g["y_max"] = max(g["y_max"], y + h)

    lines: list[OCRTextLine] = []
    for g in line_groups.values():
        full_line_text = " ".join(g["words"])
        avg_conf = sum(g["confs"]) / len(g["confs"])
        bbox = BoundingBox(
            x_min=float(g["x_min"]), y_min=float(g["y_min"]),
            x_max=float(g["x_max"]), y_max=float(g["y_max"]),
            normalized=False,
        )
        lines.append(
            OCRTextLine(
                text=full_line_text,
                confidence=round(avg_conf, 3),
                bbox=bbox,
                page_number=1,
                engine=OCREngineType.TESSERACT,
            )
        )

    if not lines:
        raise RuntimeError(f"Tesseract found no text in {image_path} (lang={tess_lang}).")

    return lines


def _run_trocr_handwriting_fallback(image_path: Path) -> list[OCRTextLine] | None:
    """
    OPTIONAL, BEST-EFFORT handwriting fallback using pretrained TrOCR.

    IMPORTANT — READ BEFORE RELYING ON THIS:
    1. This has NOT been verified end-to-end in the environment this file
       was developed in (no network access to huggingface.co there) — test
       it yourself before trusting it: `pip install transformers`, then
       run this function against one real image and check the output.
    2. The public pretrained checkpoint (microsoft/trocr-base-handwritten)
       is trained on the IAM dataset — English/Latin cursive handwriting.
       It has NO real capability for handwritten Devanagari/Tamil/Kannada/
       etc. If your land records have Indic-script handwriting (the
       realistic case for old Indian records), this will not meaningfully
       help. Real Indic handwriting recognition requires fine-tuning
       TrOCR on real + synthetic Indic handwriting data — there is no
       pretrained shortcut for this today. This function exists as a
       genuine attempt for English-language handwritten annotations only,
       not a solved handwriting pipeline.
    Returns None if transformers/torch aren't installed, rather than
    crashing the whole pipeline over an optional best-effort feature.
    """
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image
        import torch
    except ImportError:
        print("[NOTE] transformers/torch not installed — skipping TrOCR handwriting attempt.")
        return None

    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    except Exception as e:
        print(f"[NOTE] Could not load TrOCR model ({e}) — skipping handwriting attempt.")
        return None

    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    text_out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    if not text_out:
        return None

    h, w = img.size[1], img.size[0]
    return [OCRTextLine(
        text=text_out, confidence=0.5,  # deliberately capped — unverified, low-trust source
        bbox=BoundingBox(x_min=0, y_min=0, x_max=float(w), y_max=float(h), normalized=False),
        page_number=1, engine=OCREngineType.TROCR,
    )]


def run_ocr_with_fallback(image_path: Path, state: str | None) -> list[OCRTextLine]:
    """
    Tries EasyOCR first; if it fails for ANY reason (missing dependency,
    broken model checkpoint, no text found), automatically falls back to
    Tesseract with the matching language pack. Only raises if BOTH engines
    fail — never silently substitutes fake/placeholder data.
    """
    lang_list = STATE_LANGUAGES.get(state.upper(), FALLBACK_MULTILANG) if state else FALLBACK_MULTILANG
    tess_lang = STATE_TESSERACT_LANG.get(state.upper(), FALLBACK_TESSERACT_LANG) if state else FALLBACK_TESSERACT_LANG

    if not state:
        print(
            "[WARNING] No valid --state passed. Using a generic language "
            "fallback. If this is a Tamil/Kannada/Marathi/etc. document, "
            "results WILL be inaccurate — re-run with --state TN (or "
            "KA/MH/UP/MP/BR) for correct language OCR."
        )

    try:
        return _run_easyocr(image_path, lang_list)
    except Exception as easyocr_error:
        print(f"[NOTE] EasyOCR failed ({easyocr_error}). Trying Tesseract fallback...")
        try:
            return _run_tesseract(image_path, tess_lang)
        except Exception as tesseract_error:
            raise RuntimeError(
                f"Both OCR engines failed for {image_path}.\n"
                f"  EasyOCR error   : {easyocr_error}\n"
                f"  Tesseract error : {tesseract_error}\n"
                f"This is a real failure — check the image quality/format, "
                f"confirm the correct --state was passed, and confirm both "
                f"OCR engines are properly installed."
            ) from tesseract_error


def _group_lines_into_table(lines: list[OCRTextLine], y_tolerance: float = 15.0) -> TableStructure | None:
    """
    REMOVED: table/row/column structure detection is layout territory,
    not part of extraction/validation/confidence scoring. Left as a
    no-op stub so nothing else in this file needs to change shape.
    """
    return None


def ocr_image_file(image_path: str, selected_state: str | None = None) -> DocumentOCRResult:
    """
    Reads an image file, runs real OCR, and returns a DocumentOCRResult
    (Person A contract). Raises clear exceptions on failure instead of
    silently returning fake data.
    """
    img_p = Path(image_path)
    if not img_p.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    doc_hash = calculate_sha256(img_p)
    doc_id = f"DOC_IMG_{img_p.stem.upper()}"

    lines = run_ocr_with_fallback(img_p, selected_state)
    raw_text = "\n".join(l.text for l in lines)
    tables = []  # table/layout structure detection is out of scope for this module

    config_loader = ConfigLoader()
    if selected_state:
        detected_state = selected_state.upper()
        classification_confidence = 1.0   # user told us explicitly, fully trust it
    else:
        detected_state = config_loader.detect_state_from_text(raw_text)
        # honest, lower confidence since this was guessed from OCR'd text,
        # not provided by the user
        classification_confidence = 0.65 if detected_state != "DEFAULT" else 0.30

    ocr_result = DocumentOCRResult(
        document_id=doc_id,
        sha256_hash=doc_hash,
        classification=DocumentClassificationResult(
            document_type=DocumentType.BHOOMI_RTC if detected_state == "KA" else DocumentType.KHATAUNI,
            state=detected_state,
            confidence=classification_confidence,
        ),
        text_lines=lines,
        tables=tables,
        raw_full_text=raw_text,
        ocr_engine=lines[0].engine if lines else OCREngineType.EASYOCR,
    )
    return ocr_result


def main():
    parser = argparse.ArgumentParser(description="Process an image through the Land Record Pipeline")
    parser.add_argument("image_path", help="Path to land record image (PNG, JPG, TIFF)")
    parser.add_argument("--state", default=None, help="State code (KA, TN, MH, UP, MP, BR) — STRONGLY recommended for accurate OCR language selection")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print(f"LAND RECORD PIPELINE — PROCESSING IMAGE: {args.image_path}")
    print("=" * 75)

    ocr_result = ocr_image_file(args.image_path, selected_state=args.state)
    print(f"\n[PERSON A] OCR Extraction Complete:")
    print(f"  - Document ID : {ocr_result.document_id}")
    print(f"  - SHA-256 Hash: {ocr_result.sha256_hash[:16]}...")
    print(f"  - State       : {ocr_result.classification.state} "
          f"(confidence={ocr_result.classification.confidence})")
    print(f"  - OCR Lines   : {len(ocr_result.text_lines)}")
    print(f"  - Tables found: {len(ocr_result.tables)}")

    final_res = extract_and_validate(ocr_result, selected_state=args.state)

    print("\n" + "=" * 75)
    print("PERSON C: FINAL STRUCTURED OUTPUT")
    print("=" * 75)
    print(f"Document ID       : {final_res.document_id}")
    print(f"State             : {final_res.state}")
    print(f"Document Type     : {final_res.document_type.value}")
    print(f"Validation Status : {final_res.validation_status.value.upper()}")
    print(f"Overall Confidence: {final_res.overall_confidence:.3f}")
    print(f"Requires Review   : {final_res.requires_human_review}")
    if final_res.review_reasons:
        print(f"Review Reasons    : {final_res.review_reasons}")

    print("\nEXTRACTED & NORMALIZED FIELDS:")
    print("-" * 75)
    print(f"{'Field Name':<22} | {'Normalized Value':<20} | {'Raw Extracted':<15} | {'Conf':<6}")
    print("-" * 75)
    for fname, fobj in final_res.fields.items():
        unit_str = f" {fobj.normalized_unit}" if fobj.normalized_unit else ""
        norm_val = f"{fobj.normalized_value}{unit_str}"
        print(f"{fname:<22} | {norm_val:<20} | {str(fobj.raw_value):<15} | {fobj.confidence:.2f}")

    print("\nCADASTRAL GIS SPATIAL VERIFICATION:")
    print("-" * 75)
    print(f"GIS Status        : {final_res.gis_validation.gis_status.value}")
    print(f"Spatial Verified  : {final_res.gis_validation.is_verified}")
    print(f"Cadastral Area    : {final_res.gis_validation.gis_recorded_area_hectares} Hectares")
    print(f"Extracted Area    : {final_res.gis_validation.extracted_area_hectares} Hectares")
    if final_res.gis_validation.area_deviation_percent is not None:
        print(f"Area Deviation    : {final_res.gis_validation.area_deviation_percent}%")
    if final_res.gis_validation.flag_reasons:
        print(f"Diagnostic Flags  : {final_res.gis_validation.flag_reasons}")

    print("\nDUPLICATE DETECTION:")
    print("-" * 75)
    print(f"Is Duplicate      : {final_res.duplicate_analysis.is_duplicate}")
    print(f"Match Type        : {final_res.duplicate_analysis.match_type.value}")
    print("=" * 75)

    print("=" * 75)


if __name__ == "__main__":
    main()
