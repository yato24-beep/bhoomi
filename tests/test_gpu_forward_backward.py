"""GPU Forward/Backward Micro-Test for TrOCR Kannada OCR."""

import time
import pytest
import torch
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoProcessor

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_gpu_forward_backward():
    print("=== Testing GPU Forward/Backward Pass ===")
    assert torch.cuda.is_available(), "CUDA is not available!"
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    total_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"Target GPU: {gpu_name} ({total_mem_mb:.1f} MB Total VRAM)")
    
    # 1. Load VisionEncoderDecoder & tokenizer
    print("Loading TrOCR model and XLM-RoBERTa tokenizer...")
    model_name = "microsoft/trocr-small-handwritten"
    tokenizer_name = "xlm-roberta-base"
    
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Resize decoder token embeddings to match tokenizer
    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = len(tokenizer)
    
    # Move to GPU
    model = model.to(device)
    
    # 2. Setup synthetic test batch (Batch size = 4)
    batch_size = 4
    dummy_pixel_values = torch.randn(batch_size, 3, 384, 384, device=device)
    dummy_labels = torch.randint(0, len(tokenizer), (batch_size, 16), device=device)
    
    # 3. Test Forward & Backward with AMP (FP16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler("cuda")
    
    torch.cuda.reset_peak_memory_stats(0)
    start_time = time.time()
    
    optimizer.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(pixel_values=dummy_pixel_values, labels=dummy_labels)
        loss = outputs.loss
        
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    alloc_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    peak_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    
    print(f"Step Loss: {loss.item():.4f}")
    print(f"Step Elapsed Time: {elapsed:.3f}s")
    print(f"VRAM Allocated: {alloc_mb:.1f} MB | Peak VRAM: {peak_mb:.1f} MB")
    print("=== GPU Forward/Backward Test Passed Successfully ===")

if __name__ == "__main__":
    test_gpu_forward_backward()
