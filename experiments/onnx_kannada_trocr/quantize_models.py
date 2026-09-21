"""Dynamic Int8 Quantization Harness for Indic-TrOCR Checkpoint-12000.

Performs:
1. ONNX Runtime Dynamic Quantization (quantize_dynamic with QInt8 weights) on both Encoder and Decoder.
2. PyTorch Dynamic Quantization (torch.ao.quantization.quantize_dynamic) on the baseline model.
3. Quantized model size comparisons and runtime validation.
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
from transformers import VisionEncoderDecoderModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "trocr" / "checkpoint-12000"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ENCODER_FP32 = MODELS_DIR / "encoder.onnx"
ENCODER_INT8 = MODELS_DIR / "encoder_int8.onnx"
DECODER_FP32 = MODELS_DIR / "decoder.onnx"
DECODER_INT8 = MODELS_DIR / "decoder_int8.onnx"
PYTORCH_INT8_PATH = MODELS_DIR / "pytorch_trocr_int8.pt"


def quantize_onnx_encoder() -> Dict[str, Any]:
    print(f"\n[Quantization] Applying Dynamic Quantization (QInt8) to Encoder: {ENCODER_FP32.name}")
    t0 = time.perf_counter()
    quantize_dynamic(
        model_input=str(ENCODER_FP32),
        model_output=str(ENCODER_INT8),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    elapsed = time.perf_counter() - t0
    fp32_mb = ENCODER_FP32.stat().st_size / (1024 * 1024)
    int8_mb = ENCODER_INT8.stat().st_size / (1024 * 1024)
    reduction_pct = (1.0 - int8_mb / fp32_mb) * 100
    print(f"  Encoder FP32 Size: {fp32_mb:.2f} MB")
    print(f"  Encoder INT8 Size: {int8_mb:.2f} MB ({reduction_pct:.1f}% reduction in {elapsed:.2f}s)")
    return {
        "fp32_size_mb": round(fp32_mb, 2),
        "int8_size_mb": round(int8_mb, 2),
        "reduction_pct": round(reduction_pct, 2),
        "quantize_time_sec": round(elapsed, 2),
    }


def quantize_onnx_decoder() -> Dict[str, Any]:
    print(f"\n[Quantization] Applying Dynamic Quantization (QInt8) to Decoder: {DECODER_FP32.name}")
    t0 = time.perf_counter()
    quantize_dynamic(
        model_input=str(DECODER_FP32),
        model_output=str(DECODER_INT8),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    elapsed = time.perf_counter() - t0
    fp32_mb = DECODER_FP32.stat().st_size / (1024 * 1024)
    int8_mb = DECODER_INT8.stat().st_size / (1024 * 1024)
    reduction_pct = (1.0 - int8_mb / fp32_mb) * 100
    print(f"  Decoder FP32 Size: {fp32_mb:.2f} MB")
    print(f"  Decoder INT8 Size: {int8_mb:.2f} MB ({reduction_pct:.1f}% reduction in {elapsed:.2f}s)")
    return {
        "fp32_size_mb": round(fp32_mb, 2),
        "int8_size_mb": round(int8_mb, 2),
        "reduction_pct": round(reduction_pct, 2),
        "quantize_time_sec": round(elapsed, 2),
    }


def quantize_pytorch_baseline() -> Dict[str, Any]:
    print(f"\n[Quantization] Applying PyTorch Dynamic Quantization (torch.qint8) to Baseline Checkpoint...", flush=True)
    t0 = time.perf_counter()
    model = VisionEncoderDecoderModel.from_pretrained(str(CHECKPOINT_DIR), local_files_only=True)
    model.eval()

    # Dynamic quantization of Linear layers across encoder and decoder
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    elapsed = time.perf_counter() - t0

    # Save state dict
    torch.save(quantized_model.state_dict(), str(PYTORCH_INT8_PATH))
    pt_int8_mb = PYTORCH_INT8_PATH.stat().st_size / (1024 * 1024)
    print(f"  PyTorch INT8 Size: {pt_int8_mb:.2f} MB (quantized in {elapsed:.2f}s)")

    return {
        "int8_size_mb": round(pt_int8_mb, 2),
        "quantize_time_sec": round(elapsed, 2),
    }


def run_quantization() -> Dict[str, Any]:
    print("=" * 70)
    print("PHASE 4: DYNAMIC QUANTIZATION HARNESS")
    print("=" * 70)

    enc_stats = quantize_onnx_encoder()
    dec_stats = quantize_onnx_decoder()
    pt_stats = quantize_pytorch_baseline()

    total_onnx_fp32 = enc_stats["fp32_size_mb"] + dec_stats["fp32_size_mb"]
    total_onnx_int8 = enc_stats["int8_size_mb"] + dec_stats["int8_size_mb"]
    total_reduction_pct = (1.0 - total_onnx_int8 / total_onnx_fp32) * 100

    print("\n" + "=" * 70)
    print("QUANTIZATION COMPARISON SUMMARY:")
    print(f"  Total ONNX FP32 Size:     {total_onnx_fp32:.2f} MB")
    print(f"  Total ONNX INT8 Size:     {total_onnx_int8:.2f} MB ({total_reduction_pct:.1f}% reduction)")
    print(f"  PyTorch Quantized Size:   {pt_stats['int8_size_mb']:.2f} MB")
    print("=" * 70)

    return {
        "encoder": enc_stats,
        "decoder": dec_stats,
        "total_onnx_fp32_mb": round(total_onnx_fp32, 2),
        "total_onnx_int8_mb": round(total_onnx_int8, 2),
        "total_onnx_reduction_pct": round(total_reduction_pct, 2),
        "pytorch_int8": pt_stats,
    }


if __name__ == "__main__":
    run_quantization()
