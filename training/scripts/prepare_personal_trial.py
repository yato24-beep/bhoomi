import os
import sys
import json
from PIL import Image

sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAW_DIR = r"c:\Land Record\training\datasets\personal_trial\raw"
CROPS_DIR = r"c:\Land Record\training\datasets\personal_trial\crops"
OUT_DIR = r"c:\Land Record\training\datasets\personal_trial"

os.makedirs(CROPS_DIR, exist_ok=True)

# Exact bounding boxes on 90-deg CCW rotated images
# a: rotated 90 CCW (1280x720) -> 'ಮರ' is at x=[530..675], y=[120..230]
# o: rotated 90 CCW (1600x900) -> 'ಕೋತಿ' is at x=[595..830], y=[145..275]
# p: rotated 90 CCW (1280x720) -> 'ಹಣ್ಣು' is at x=[535..680], y=[175..240]

samples = [
    {
        "id": "sample_a_mara",
        "raw_image": "a.jpeg",
        "text": "ಮರ",
        "english_meaning": "TREE",
        "bbox": (530, 120, 675, 230),
        "crop_filename": "crop_a_mara.png",
    },
    {
        "id": "sample_o_kothi",
        "raw_image": "o.jpeg",
        "text": "ಕೋತಿ",
        "english_meaning": "MONKEY",
        "bbox": (595, 145, 830, 275),
        "crop_filename": "crop_o_kothi.png",
    },
    {
        "id": "sample_p_hannu",
        "raw_image": "p.jpeg",
        "text": "ಹಣ್ಣು",
        "english_meaning": "FRUIT",
        "bbox": (535, 175, 680, 240),
        "crop_filename": "crop_p_hannu.png",
    },
]

records = []
for s in samples:
    raw_path = os.path.join(RAW_DIR, s["raw_image"])
    rot = Image.open(raw_path).rotate(90, expand=True)
    crop = rot.crop(s["bbox"])
    
    crop_path = os.path.join(CROPS_DIR, s["crop_filename"])
    crop.save(crop_path)
    
    rel_crop_path = f"training/datasets/personal_trial/crops/{s['crop_filename']}"
    
    record = {
        "image": rel_crop_path,
        "text": s["text"],
        "language": "kannada",
        "script": "Kannada",
        "metadata": {
            "source": "personal_handwritten_trial",
            "source_type": "real_handwriting",
            "raw_image": s["raw_image"],
            "english_meaning": s["english_meaning"],
            "crop_bbox": s["bbox"],
            "raw_text": s["text"],
        }
    }
    records.append(record)
    print(f"Created crop: {rel_crop_path} for label: {s['text']} ({s['english_meaning']}), size: {crop.size}")

# Write trial JSONL manifests
train_jsonl = os.path.join(OUT_DIR, "train.jsonl")
val_jsonl = os.path.join(OUT_DIR, "val.jsonl")
trial_all_jsonl = os.path.join(OUT_DIR, "trial_all.jsonl")

for path in [train_jsonl, val_jsonl, trial_all_jsonl]:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Successfully wrote {len(records)} samples to {train_jsonl} and {val_jsonl}")
