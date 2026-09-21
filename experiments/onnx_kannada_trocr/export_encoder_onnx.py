"""Export ViT Vision Encoder of Indic-TrOCR to ONNX and verify numerical parity.

Input:
- pixel_values: float32 tensor of shape (batch_size, 3, 224, 224)
Output:
- last_hidden_state: float32 tensor of shape (batch_size, 197, 768)
"""

import os
from pathlib import Path
import sys
import time
from typing import Dict, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from transformers import AutoImageProcessor, VisionEncoderDecoderModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "trocr" / "checkpoint-12000"
MODELS_OUT_DIR = Path(__file__).resolve().parent / "models"
MODELS_OUT_DIR.mkdir(parents=True, exist_ok=True)
ENCODER_ONNX_PATH = MODELS_OUT_DIR / "encoder.onnx"


class ViTEncoderWrapper(torch.nn.Module):
    """Clean wrapper around ViTModel encoder to ensure explicit output tensor."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.encoder(pixel_values=pixel_values)
        # encoder_outputs.last_hidden_state is (batch_size, 197, 768)
        return encoder_outputs.last_hidden_state


def export_vision_encoder() -> Tuple[bool, Dict[str, float]]:
    print("=" * 70)
    print("PHASE 2: EXPORTING VISION ENCODER TO ONNX")
    print("=" * 70)

    print(f"Loading checkpoint from: {CHECKPOINT_DIR}", flush=True)
    model = VisionEncoderDecoderModel.from_pretrained(str(CHECKPOINT_DIR), local_files_only=True)
    encoder = model.get_encoder()
    encoder.eval()

    wrapper = ViTEncoderWrapper(encoder)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"Exporting encoder to: {ENCODER_ONNX_PATH}...")
    t0 = time.perf_counter()

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(ENCODER_ONNX_PATH),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
        },
    )
    export_time = time.perf_counter() - t0
    total_bytes = ENCODER_ONNX_PATH.stat().st_size
    data_path = Path(str(ENCODER_ONNX_PATH) + ".data")
    if data_path.exists():
        total_bytes += data_path.stat().st_size
    file_size_mb = total_bytes / (1024 * 1024)
    print(f"Encoder ONNX export completed in {export_time:.2f}s. Total Size: {file_size_mb:.2f} MB", flush=True)

    # Verify ONNX model
    import onnx
    print("Validating ONNX model schema with onnx.checker...")
    onnx_model = onnx.load(str(ENCODER_ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("ONNX model check PASSED!")

    # Verify numerical consistency with ONNX Runtime
    import onnxruntime as ort
    print("Testing numerical parity between PyTorch and ONNX Runtime...")
    ort_session = ort.InferenceSession(
        str(ENCODER_ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    test_input = torch.randn(2, 3, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        pt_out = wrapper(test_input).numpy()

    ort_inputs = {"pixel_values": test_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    mean_diff = np.mean(np.abs(pt_out - ort_out))
    print(f"Max Absolute Error:  {max_diff:.6e}")
    print(f"Mean Absolute Error: {mean_diff:.6e}")

    assert max_diff < 1e-4, f"Encoder parity check failed: max_diff {max_diff} exceeds 1e-4"
    print("Encoder numerical parity check PASSED!")

    return True, {
        "export_time_sec": round(export_time, 2),
        "disk_size_mb": round(file_size_mb, 2),
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
    }


if __name__ == "__main__":
    export_vision_encoder()
