import os
import sys
import json
import torch
from PIL import Image

sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import src

from src.training.config import TrainingConfig
from src.training.dataset import HandwritingDataset
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, XLMRobertaTokenizer

CONFIG_PATH = r"c:\Land Record\training\configs\personal_trial.yaml"

print("=" * 60)
print("  PERSONAL HANDWRITING FINE-TUNING TRIAL: DRY RUN VERIFICATION")
print("=" * 60)

# 1. Config Loading
config = TrainingConfig.from_yaml(CONFIG_PATH)
print("[✓] 1. Configuration loaded successfully")

# 2. Check Image Files & Crops
crop_paths = [
    r"c:\Land Record\training\datasets\personal_trial\crops\crop_a_mara.png",
    r"c:\Land Record\training\datasets\personal_trial\crops\crop_o_kothi.png",
    r"c:\Land Record\training\datasets\personal_trial\crops\crop_p_hannu.png",
]
for p in crop_paths:
    assert os.path.exists(p), f"Crop not found: {p}"
    im = Image.open(p)
    print(f"[✓] 2. Crop verified: {os.path.basename(p)} size={im.size} mode={im.mode}")

# 3. Check Kannada Tokenizer & Labels
CKPT = r"c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint"
tokenizer = XLMRobertaTokenizer.from_pretrained(CKPT)
processor = TrOCRProcessor.from_pretrained(CKPT)

test_labels = ["ಮರ", "ಕೋತಿ", "ಹಣ್ಣು"]
for lbl in test_labels:
    tokens = tokenizer(lbl, return_tensors="pt").input_ids
    decoded = tokenizer.decode(tokens[0], skip_special_tokens=True)
    assert decoded == lbl, f"Tokenizer mismatch for {lbl}: got {decoded}"
    print(f"[✓] 3. Tokenizer verified: '{lbl}' -> token IDs: {tokens[0].tolist()} -> decoded: '{decoded}'")

# 4. Check Dataset loading
dataset = HandwritingDataset(
    manifest_path=r"c:\Land Record\training\datasets\personal_trial\train.jsonl",
    processor=processor,
    max_target_length=32,
    root_dir=r"c:\Land Record",
)
print(f"[✓] 4. Dataset loaded with {len(dataset)} samples")

# 5. Model Loading on CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VisionEncoderDecoderModel.from_pretrained(CKPT).to(device)
print(f"[✓] 5. Model loaded from {CKPT} onto device: {device}")

# 6. Loss Calculation Forward Pass
pixel_values_list = []
labels_list = []
for i in range(len(dataset)):
    item = dataset[i]
    pixel_values_list.append(item["pixel_values"])
    labels_list.append(item["labels"])

batch_pixels = torch.stack(pixel_values_list).to(device)
# Pad labels
max_len = max(len(l) for l in labels_list)
padded_labels = []
for l in labels_list:
    pad_len = max_len - len(l)
    padded = torch.cat([l, torch.full((pad_len,), -100, dtype=torch.long)]) if pad_len > 0 else l
    padded_labels.append(padded)
batch_labels = torch.stack(padded_labels).to(device)

model.train()
outputs = model(pixel_values=batch_pixels, labels=batch_labels)
loss = outputs.loss
print(f"[✓] 6. Loss calculation forward pass successful: loss = {loss.item():.4f}")

# Backward pass check
loss.backward()
print("[✓] 7. Gradient backward pass successful!")
print("\n[DRY RUN PASSED 100%] Ready for trial fine-tuning.")
