"""
Evaluate saved medium checkpoint on validation Kannada crops.
"""
import json
from pathlib import Path
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, XLMRobertaTokenizer, TrOCRProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CKPT_DIR = PROJECT_ROOT / "models" / "trocr" / "kannada_medium_checkpoints" / "best_checkpoint"
VAL_MANIFEST = PROJECT_ROOT / "training" / "datasets" / "iiit_kannada_val.jsonl"

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    print(f"Loading checkpoint from: {CKPT_DIR}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = VisionEncoderDecoderModel.from_pretrained(str(CKPT_DIR)).to(device)
    tokenizer = XLMRobertaTokenizer.from_pretrained(str(CKPT_DIR))
    processor = TrOCRProcessor.from_pretrained(str(CKPT_DIR))
    model.eval()

    samples = []
    with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= 15:
                break

    print("\n" + "=" * 80)
    print(f"{'#':<3} | {'Ground Truth (Kannada)':<25} | {'Predicted (Model Output)':<30} | {'Match'}")
    print("=" * 80)

    for idx, sample in enumerate(samples, start=1):
        rel_path = sample.get("image") or sample.get("image_path")
        img_path = PROJECT_ROOT / rel_path
        gt_text = sample["text"]
        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=32,
                decoder_start_token_id=model.config.decoder_start_token_id,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=model.config.eos_token_id,
            )
        
        pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        match = "YES" if pred_text == gt_text else "NO"
        print(f"{idx:<3} | {gt_text:<25} | {pred_text:<30} | {match}")
    print("=" * 80)

if __name__ == "__main__":
    main()
