import json
import os
import sys
import random
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

random.seed(42)

VOWELS = ["ಅ", "ಆ", "ಇ", "ಈ", "ಉ", "ಊ", "ಋ", "ಎ", "ಏ", "ಐ", "ಒ", "ಓ", "ಔ"]
CONSONANTS = [
    "ಕ", "ಖ", "ಗ", "ಘ", "ಙ",
    "ಚ", "ಛ", "ಜ", "ಝ", "ಞ",
    "ಟ", "ಠ", "ಡ", "ಢ", "ಣ",
    "ತ", "ಥ", "ದ", "ಧ", "ನ",
    "ಪ", "ಫ", "ಬ", "ಭ", "ಮ",
    "ಯ", "ರ", "ಲ", "ವ",
    "ಶ", "ಷ", "ಸ", "ಹ",
    "ಳ"
]
MATRAS = ["ಾ", "ಿ", "ೀ", "ು", "ೂ", "ೃ", "ೆ", "ೇ", "ೈ", "ೊ", "ೋ", "ೌ"]
NUMERALS = ["೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯"]
TOP_CONJUNCTS = [
    "ಲ್ಲ", "ತ್ತ", "ದ್ದ", "ನ್ನ", "ಕ್ಕ", "ಪ್ರ", "ಟ್ಟ", "ಳ್ಳ", "ತ್ರ", "ಕ್ಷ",
    "ಷ್ಟ", "ಸ್ತ", "ಪ್ಪ", "ವ್ಯ", "ಸ್ಥ", "ಬ್ಬ", "ತ್ಯ", "ಕ್ತ", "ಮ್ಮ", "ಚ್ಚ"
]

VAL_MANIFEST = r"c:\Land Record\training\datasets\iiit_kannada_val.jsonl"
OUT_DIR = r"c:\Land Record\training\datasets\kannada_diagnostic_val"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "diagnostic_full.jsonl")

def build_diagnostic_dataset():
    print(f"Reading validation samples from: {VAL_MANIFEST}...", flush=True)
    all_val_samples = []
    with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rel_img = item.get("image", "")
            img_p = os.path.join(r"c:\Land Record", rel_img)
            if os.path.exists(img_p):
                all_val_samples.append(item)

    print(f"Total valid image samples in IIIT Val: {len(all_val_samples):,}", flush=True)

    pools = defaultdict(list)
    for s in all_val_samples:
        txt = s.get("text", "")
        # Check vowels
        for v in VOWELS:
            if v in txt:
                pools[f"vowel_{v}"].append(s)
        # Check consonants
        for c in CONSONANTS:
            if c in txt:
                pools[f"cons_{c}"].append(s)
        # Check matras
        for m in MATRAS:
            if m in txt:
                pools[f"matra_{m}"].append(s)
        # Check numerals
        for n in NUMERALS:
            if n in txt:
                pools[f"num_{n}"].append(s)
        # Check conjuncts
        for cj in TOP_CONJUNCTS:
            if cj in txt:
                pools[f"conj_{cj}"].append(s)

    selected_samples = {}  # key by image path to avoid duplicates
    diagnostic_metadata = []

    # Sample for each category
    def add_from_pool(pool_key, max_k, tag):
        pool = pools.get(pool_key, [])
        if not pool:
            return
        sample_k = random.sample(pool, min(max_k, len(pool)))
        for s in sample_k:
            img_k = s["image"]
            if img_k not in selected_samples:
                item_copy = dict(s)
                item_copy["diag_tags"] = [tag]
                selected_samples[img_k] = item_copy
            else:
                if tag not in selected_samples[img_k]["diag_tags"]:
                    selected_samples[img_k]["diag_tags"].append(tag)

    # 1. Sample vowels (up to 8 each)
    for v in VOWELS:
        add_from_pool(f"vowel_{v}", 8, f"vowel:{v}")

    # 2. Sample consonants (up to 8 each)
    for c in CONSONANTS:
        add_from_pool(f"cons_{c}", 8, f"cons:{c}")

    # 3. Sample matras (up to 8 each)
    for m in MATRAS:
        add_from_pool(f"matra_{m}", 8, f"matra:{m}")

    # 4. Sample numerals (up to 8 each)
    for n in NUMERALS:
        add_from_pool(f"num_{n}", 8, f"num:{n}")

    # 5. Sample conjuncts (up to 6 each)
    for cj in TOP_CONJUNCTS:
        add_from_pool(f"conj_{cj}", 6, f"conj:{cj}")

    # 6. Add 50 general anchor samples
    anchor_candidates = [s for s in all_val_samples if s["image"] not in selected_samples]
    sample_anchors = random.sample(anchor_candidates, min(50, len(anchor_candidates)))
    for s in sample_anchors:
        item_copy = dict(s)
        item_copy["diag_tags"] = ["anchor"]
        selected_samples[s["image"]] = item_copy

    diag_list = list(selected_samples.values())
    random.shuffle(diag_list)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for item in diag_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[✓] Diagnostic Full Validation Dataset Built: {len(diag_list)} samples saved to {OUT_PATH}", flush=True)

if __name__ == "__main__":
    build_diagnostic_dataset()
