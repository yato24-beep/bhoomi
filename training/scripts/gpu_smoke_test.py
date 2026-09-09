"""GPU Smoke Test for TrOCR Kannada Handwriting Recognition.

Verifies:
1. CUDA availability and device properties (NVIDIA RTX 4050 Laptop GPU).
2. VisionEncoderDecoderModel architecture with XLM-RoBERTa multilingual tokenizer on CUDA.
3. Full forward pass with Loss calculation on synthetic batch [B=2, 3, 384, 384].
4. Backward pass (loss.backward()) and gradient computation.
5. Autoregressive text generation step on CUDA (model.generate()).
6. Memory allocation, peak VRAM measurement, and CUDA cache release.
"""

import sys
import time
from pathlib import Path

# Ensure UTF-8 output encoding
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    XLMRobertaTokenizer,
)


def run_gpu_smoke_test():
    print("=" * 75, flush=True)
    print("  PHASE 1: GPU ENVIRONMENT & HARDWARE SMOKE TEST", flush=True)
    print("=" * 75, flush=True)

    print(f"PyTorch Version       : {torch.__version__}", flush=True)
    print(f"CUDA Available        : {torch.cuda.is_available()}", flush=True)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is NOT available to PyTorch! Aborting GPU smoke test.", flush=True)
        sys.exit(1)

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    total_vram_gb = device_props.total_memory / (1024 ** 3)
    major_cc, minor_cc = device_props.major, device_props.minor

    print(f"CUDA Device Count     : {device_count}", flush=True)
    print(f"Active Device (cuda:0): {device_name}", flush=True)
    print(f"Compute Capability    : {major_cc}.{minor_cc}", flush=True)
    print(f"Total Dedicated VRAM  : {total_vram_gb:.2f} GB ({device_props.total_memory:,} bytes)", flush=True)
    print("-" * 75, flush=True)

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    initial_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f"[1/5] Initial Allocated VRAM: {initial_alloc:.2f} MB", flush=True)

    # 1. Load Tokenizer & Processor
    model_id = "microsoft/trocr-small-handwritten"
    tokenizer_id = "xlm-roberta-base"
    print(f"[2/5] Loading Model ({model_id}) & Tokenizer ({tokenizer_id})...", flush=True)

    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_id)
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

    model = VisionEncoderDecoderModel.from_pretrained(model_id)

    # Resize decoder token embeddings for multilingual tokenizer
    vocab_len = len(tokenizer)
    curr_embeddings = model.decoder.get_input_embeddings().num_embeddings
    print(f"      Original Decoder Vocab : {curr_embeddings:,}", flush=True)
    print(f"      Resizing to Multilingual: {vocab_len:,} (XLM-RoBERTa)", flush=True)
    model.decoder.resize_token_embeddings(vocab_len)

    # Configure special token IDs
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
    model.config.pad_token_id = tokenizer.pad_token_id or 1
    model.config.eos_token_id = tokenizer.eos_token_id or 2
    model.config.vocab_size = vocab_len

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
        model.generation_config.pad_token_id = model.config.pad_token_id
        model.generation_config.eos_token_id = model.config.eos_token_id

    # 2. Move model to CUDA
    print("[3/5] Transferring model to GPU (cuda:0)...", flush=True)
    start_load = time.time()
    model.to(device)
    load_time = time.time() - start_load
    model_vram = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f"      Transfer Time : {load_time:.2f}s", flush=True)
    print(f"      Model VRAM    : {model_vram:.2f} MB", flush=True)

    # 3. Create dummy batch
    batch_size = 2
    dummy_pixel_values = torch.randn(batch_size, 3, 384, 384, device=device)
    # Synthetic Kannada labels: "ಕನ್ನಡ" -> token IDs
    sample_text = "ಕನ್ನಡ ಭೂ ದಾಖಲೆ"
    labels_tensor = tokenizer(
        [sample_text] * batch_size,
        padding="max_length",
        max_length=32,
        return_tensors="pt",
    ).input_ids.to(device)
    # Mask padding tokens with -100
    labels_tensor[labels_tensor == tokenizer.pad_token_id] = -100

    print(f"[4/5] Executing Forward & Backward Optimization Pass (Batch Size: {batch_size})...", flush=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    start_step = time.time()
    outputs = model(pixel_values=dummy_pixel_values, labels=labels_tensor)
    loss = outputs.loss
    loss_val = loss.item()
    print(f"      Forward Loss  : {loss_val:.4f}", flush=True)

    loss.backward()
    optimizer.step()
    step_time = (time.time() - start_step) * 1000

    train_vram = torch.cuda.memory_allocated(device) / (1024 ** 2)
    peak_train_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"      Step Duration : {step_time:.2f} ms", flush=True)
    print(f"      Allocated VRAM: {train_vram:.2f} MB (Peak: {peak_train_vram:.2f} MB)", flush=True)

    # 4. Test Autoregressive Generation on CUDA
    print("[5/5] Testing Autoregressive Generation (model.generate) on CUDA...", flush=True)
    model.eval()
    with torch.no_grad():
        gen_start = time.time()
        generated_ids = model.generate(
            dummy_pixel_values[:1],
            max_new_tokens=16,
        )
        gen_time = (time.time() - gen_start) * 1000
        decoded_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(f"      Generation Time : {gen_time:.2f} ms", flush=True)
        print(f"      Decoded Output  : '{decoded_text[0]}'", flush=True)

    final_peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print("\n" + "=" * 75, flush=True)
    print("  GPU SMOKE TEST PASSED SUCCESSFULLY!", flush=True)
    print("=" * 75, flush=True)
    print(f"  Target Device       : {device_name}", flush=True)
    print(f"  Total GPU Memory    : {total_vram_gb:.2f} GB", flush=True)
    print(f"  Peak VRAM Used      : {final_peak_vram:.2f} MB ({final_peak_vram / (1024 * total_vram_gb) * 100:.1f}% of total)", flush=True)
    print(f"  CUDA Compute Status : FULLY OPERATIONAL & READY FOR TRAINING", flush=True)
    print("=" * 75, flush=True)

    return True


if __name__ == "__main__":
    run_gpu_smoke_test()
