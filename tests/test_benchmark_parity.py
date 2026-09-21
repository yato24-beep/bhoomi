"""Empirical parity validation: Compare PyTorch Checkpoint-12000 vs FP32 ONNX Pipeline on benchmark crops.
Measures Character Error Rate (CER), Word Error Rate (WER), and Exact Match Accuracy.
"""

import json
import os
from pathlib import Path
import sys
import unicodedata
from PIL import Image
import numpy as np

# Ensure root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.handwriting.trocr_12000_recognizer import TrOCR12000KannadaRecognizer
from scripts.export_byte_level_table import bytes_to_unicode

def compute_cer(gt: str, pred: str) -> float:
    """Levenshtein distance at character level."""
    if not gt:
        return 0.0 if not pred else 1.0
    d = np.zeros((len(gt) + 1, len(pred) + 1), dtype=int)
    for i in range(len(gt) + 1):
        d[i, 0] = i
    for j in range(len(pred) + 1):
        d[0, j] = j
    for i in range(1, len(gt) + 1):
        for j in range(1, len(pred) + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(gt), len(pred)]) / len(gt)

def compute_wer(gt: str, pred: str) -> float:
    """Levenshtein distance at word level."""
    gt_words = gt.split()
    pred_words = pred.split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    d = np.zeros((len(gt_words) + 1, len(pred_words) + 1), dtype=int)
    for i in range(len(gt_words) + 1):
        d[i, 0] = i
    for j in range(len(pred_words) + 1):
        d[0, j] = j
    for i in range(1, len(gt_words) + 1):
        for j in range(1, len(pred_words) + 1):
            cost = 0 if gt_words[i - 1] == pred_words[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(gt_words), len(pred_words)]) / len(gt_words)

BENCHMARK_SAMPLES = [
    {"filename": "crop_a_mara.png", "gt": "ಮರ", "subset": "archival_word"},
    {"filename": "crop_o_kothi.png", "gt": "ಕೋತಿ", "subset": "archival_word"},
    {"filename": "crop_p_hannu.png", "gt": "ಹಣ್ಣು", "subset": "archival_word"},
    {"filename": "a_full_line.png", "gt": "ಆ", "subset": "archival_word"},
    {"filename": "o_full_line.png", "gt": "ಓ", "subset": "archival_word"},
    {"filename": "p_full_line.png", "gt": "ಪ", "subset": "archival_word"},
    {"filename": "line_04.png", "gt": "125", "subset": "archival_line"},
    {"filename": "line_05.png", "gt": "125 1 ರ", "subset": "archival_line"},
    {"filename": "line_06.png", "gt": "ಶ್ರೀ ಕರಿದಿರ್ N. ಮದನಗೌಡರ ಕಂದಾಯ", "subset": "archival_line"},
    {"filename": "line_07.png", "gt": "ಶ್ರೀ ಕರಿದಿರ್ N. ಮದನಗೌಡರ ತಂದೆ ಲಿಂಗಯ್ಯ 1 ನೇ ಹೆಂಡತಿ ಲಿಂಗಮ್ಮ ಇವರ ಮಕ್ಕಳು", "subset": "archival_line"},
    {"filename": "line_08.png", "gt": "148 0-15 ಇವರ ಹೆಸರಿಗೆ ರಾಜೀನಾಮೆ ಗ್ರಾಮ ಪಂಚಾಯಿತಿ ವತಿಯಿಂದ ಬಂದಂತೆ", "subset": "archival_line"},
    {"filename": "line_09.png", "gt": "ದಿನಾಂಕ 20/09/2005 ರ ಪ್ರಕಾರ ಪಹಣಿ ಮತ್ತು ಹಕ್ಕು ದಾಖಲೆ ನಮೂದಿಸಲಾಗಿದೆ", "subset": "archival_line"},
    {"filename": "line_10.png", "gt": "ಖಾತೆ ಬದಲಾವಣೆ ಮಂಜೂರಾಗಿದೆ ತಹಶೀಲ್ದಾರ್ ಆದೇಶ ಸಂಖ್ಯೆ", "subset": "archival_line"},
]

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    benchmark_dir = REPO_ROOT / "locked_benchmark_export"

    print(f"Loaded {len(BENCHMARK_SAMPLES)} locked benchmark samples from {benchmark_dir}")

    # 1. Initialize PyTorch model
    print("\n--- Initializing PyTorch TrOCR Checkpoint-12000 ---")
    pt_recognizer = TrOCR12000KannadaRecognizer(auto_load=True, device="cpu")

    # 2. Initialize ONNX runtime sessions (simulating browser execution)
    import onnxruntime as ort
    print("\n--- Initializing ONNX FP32 Sessions (Browser-Equivalent) ---")
    encoder_path = REPO_ROOT / "experiments" / "onnx_kannada_trocr" / "models" / "encoder.onnx"
    decoder_path = REPO_ROOT / "experiments" / "onnx_kannada_trocr" / "models" / "decoder.onnx"
    
    enc_session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])
    dec_session = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])

    # Load tokenizer and byte-level mapping
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Chakita/KannadaBERT")
    vocab = tok.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    def decode_tokens_bytelevel(token_ids):
        raw_bytes = bytearray()
        for tid in token_ids:
            if tid in (0, 1, 2, 3):  # <s>, <pad>, </s>, <unk>
                continue
            token_str = id_to_token.get(tid, "")
            for ch in token_str:
                if ch in u2b:
                    raw_bytes.append(u2b[ch])
                else:
                    raw_bytes.extend(ch.encode("utf-8"))
        return raw_bytes.decode("utf-8", errors="replace").strip()

    def run_onnx_fp32(image_path):
        img = Image.open(image_path).convert("RGB").resize((224, 224), Image.BILINEAR)
        img_arr = np.array(img, dtype=np.float32)
        # Normalization: (val - 127.5) / 127.5, shape [1, 3, 224, 224]
        img_norm = (img_arr - 127.5) / 127.5
        planar = np.transpose(img_norm, (2, 0, 1))
        pixel_values = np.expand_dims(planar, axis=0)

        enc_out = enc_session.run(None, {"pixel_values": pixel_values})[0]

        tokens = [0]
        for _ in range(64):
            input_ids = np.array([tokens], dtype=np.int64)
            logits = dec_session.run(None, {"input_ids": input_ids, "encoder_hidden_states": enc_out})[0]
            next_token = int(np.argmax(logits[0, -1, :]))
            if next_token == 2:
                break
            tokens.append(next_token)

        return decode_tokens_bytelevel(tokens)

    # 3. Run side-by-side benchmark evaluation
    print("\n--- Running Side-by-Side Accuracy Evaluation ---")
    pt_exact = 0
    onnx_exact = 0
    pt_total_cer = 0.0
    onnx_total_cer = 0.0
    pt_total_wer = 0.0
    onnx_total_wer = 0.0
    identical_outputs = 0

    results = []

    for i, item in enumerate(BENCHMARK_SAMPLES):
        img_name = item["filename"]
        img_path = benchmark_dir / img_name
        gt_text = item["gt"].strip()

        if not img_path.exists():
            print(f"Warning: {img_path} not found, skipping")
            continue

        # PyTorch prediction
        res_pt = pt_recognizer.recognize_handwriting(str(img_path))
        pt_pred = (res_pt.text if hasattr(res_pt, "text") else str(res_pt)).strip()

        # ONNX FP32 prediction (browser-equivalent)
        onnx_pred = run_onnx_fp32(str(img_path))

        # Check identical output
        is_identical = (pt_pred == onnx_pred)
        if is_identical:
            identical_outputs += 1

        cer_pt = compute_cer(gt_text, pt_pred)
        cer_onnx = compute_cer(gt_text, onnx_pred)
        wer_pt = compute_wer(gt_text, pt_pred)
        wer_onnx = compute_wer(gt_text, onnx_pred)

        pt_total_cer += cer_pt
        onnx_total_cer += cer_onnx
        pt_total_wer += wer_pt
        onnx_total_wer += wer_onnx

        if pt_pred == gt_text:
            pt_exact += 1
        if onnx_pred == gt_text:
            onnx_exact += 1

        results.append({
            "sample": i + 1,
            "filename": img_name,
            "gt": gt_text,
            "pytorch": pt_pred,
            "onnx_fp32": onnx_pred,
            "identical": is_identical,
            "cer_pt": round(cer_pt, 4),
            "cer_onnx": round(cer_onnx, 4),
        })

        print(f"[{i+1}/{len(BENCHMARK_SAMPLES)}] {img_name}")
        print(f"  GT      : {gt_text}")
        print(f"  PyTorch : {pt_pred}")
        print(f"  ONNX    : {onnx_pred}")
        print(f"  Match   : {'YES' if is_identical else 'NO'}")

    n = len(results)
    print("\n" + "=" * 70)
    print("EMPIRICAL BENCHMARK PARITY REPORT")
    print("=" * 70)
    print(f"Total Benchmark Samples         : {n}")
    print(f"Identical Predictions (PT=ONNX)   : {identical_outputs}/{n} ({identical_outputs/n*100:.1f}%)")
    print(f"PyTorch Mean CER                : {pt_total_cer/n*100:.2f}%")
    print(f"ONNX FP32 Mean CER              : {onnx_total_cer/n*100:.2f}%")
    print(f"PyTorch Mean WER                : {pt_total_wer/n*100:.2f}%")
    print(f"ONNX FP32 Mean WER              : {onnx_total_wer/n*100:.2f}%")
    print(f"PyTorch Exact Match Accuracy    : {pt_exact}/{n} ({pt_exact/n*100:.2f}%)")
    print(f"ONNX FP32 Exact Match Accuracy   : {onnx_exact}/{n} ({onnx_exact/n*100:.2f}%)")
    print("=" * 70)

    # Save validation results
    val_export = REPO_ROOT / "experiments" / "onnx_kannada_trocr" / "accuracy_parity_results.json"
    with open(val_export, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": n,
            "identical_count": identical_outputs,
            "identical_pct": round(identical_outputs / n * 100, 2),
            "pytorch_cer": round(pt_total_cer / n * 100, 2),
            "onnx_fp32_cer": round(onnx_total_cer / n * 100, 2),
            "pytorch_wer": round(pt_total_wer / n * 100, 2),
            "onnx_fp32_wer": round(onnx_total_wer / n * 100, 2),
            "pytorch_exact_match": round(pt_exact / n * 100, 2),
            "onnx_fp32_exact_match": round(onnx_exact / n * 100, 2),
            "per_sample": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Parity report saved to: {val_export}")

if __name__ == "__main__":
    main()
